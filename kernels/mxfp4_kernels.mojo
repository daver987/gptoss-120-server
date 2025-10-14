# -----------------------------------------------
# MXFP4 Quantization + Fused-Dequant Kernels
# -----------------------------------------------
from math import floor, log2, ldexp
import compiler
from layout import Layout, LayoutTensor
from max.tensor import InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from gpu.host import DeviceContext
from gpu import block_idx, thread_idx, block_dim, syncthreads, shared_array

alias QK_MXFP4: Int = 32

@always_inline
fn min_int(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@always_inline
fn max_int(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

@always_inline
fn clamp_f32(x: Float32, lo: Float32, hi: Float32) -> Float32:
    if x < lo: return lo
    if x > hi: return hi
    return x

@always_inline
fn ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b

@always_inline
fn e8m0_to_fp32(e: UInt8) -> Float32:
    return ldexp(Float32(1.0), Int(e) - 127)

@always_inline
fn fp32_to_e8m0_from_block_max(max_val: Float32) -> UInt8:
    if max_val <= 0.0: return 0
    if max_val < 1e-38: return 0
    var e_est: Float32 = floor(log2(max_val)) - 2.5850000 + 127.0
    return UInt8(clamp_f32(e_est, 1.0, 254.0))

@always_inline
fn unit_from_code(code: UInt8) -> Float32:
    var c: Int = Int(code)
    if c == 0:  return 0.0
    elif c == 1:  return 0.5
    elif c == 2:  return 1.0
    elif c == 3:  return 1.5
    elif c == 4:  return 2.0
    elif c == 5:  return 3.0
    elif c == 6:  return 4.0
    elif c == 7:  return 6.0
    elif c == 8:  return -0.0
    elif c == 9:  return -0.5
    elif c == 10: return -1.0
    elif c == 11: return -1.5
    elif c == 12: return -2.0
    elif c == 13: return -3.0
    elif c == 14: return -4.0
    else:        return -6.0

@always_inline
fn encode_mxfp4(val: Float32, d: Float32) -> UInt8:
    if d == 0.0: return 0
    var a = val / d
    var sign: UInt8 = 0
    if a < 0.0:
        sign = 0x8
        a = -a
    var k: UInt8 = 0
    if a >= 0.25: k = 1
    if a >= 0.75: k = 2
    if a >= 1.25: k = 3
    if a >= 1.75: k = 4
    if a >= 2.5:  k = 5
    if a >= 3.5:  k = 6
    if a >= 5.0:  k = 7
    return sign | k

# -----------------------------------------------------------------------------
# QUANTIZE: X[H,W] -> Q[H,W/2] + E[H,W/32]
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_quantize_exq")
struct MXFP4QuantizeEXQ:
    @staticmethod
    fn execute[in_dtype: DType, rank: Int, BN: Int, BD: Int, target: StaticString](
        out_q: OutputTensor[dtype = DType.uint8, rank=rank],
        out_e: OutputTensor[dtype = DType.uint8, rank=rank],
        x: InputTensor[dtype=in_dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        constrained[in_dtype == DType.float32, "quantize expects float32 input"]()
        var X = x.to_layout_tensor()
        var Q = out_q.to_layout_tensor()
        var E = out_e.to_layout_tensor()
        alias W = X.shape[1]()
        constrained[W % QK_MXFP4 == 0, "W must be divisible by 32"]()
        @parameter
        if target == "cpu":
            _mxfp4_quantize_cpu(X, Q, E)
        else:
            var dev = ctx.get_device_context()
            _mxfp4_quantize_gpu[BN, BD](dev, X, Q, E)

fn _mxfp4_quantize_cpu(X: LayoutTensor, mut Q: LayoutTensor, mut E: LayoutTensor):
    alias H = X.shape[0]()
    alias W = X.shape[1]()
    let blocks_per_row = W // QK_MXFP4
    alias XW = X.element_layout.size()
    alias QW = Q.element_layout.size()
    alias EW = E.element_layout.size()
    for r in range(H):
        for b in range(blocks_per_row):
            let c0 = b * QK_MXFP4
            var m: Float32 = 0.0
            for j in range(QK_MXFP4):
                let f: Float32 = X[r, c0 + j].cast[DType.float32]()[0]
                let af = (f if f >= 0.0 else -f)
                if af > m: m = af
            let e: UInt8 = fp32_to_e8m0_from_block_max(m)
            E[r, b] = SIMD[DType.uint8, EW](e)
            let d: Float32 = e8m0_to_fp32(e)
            let q_base = b * (QK_MXFP4 // 2)
            for j in range(QK_MXFP4 // 2):
                let v0: Float32 = X[r, c0 + j].cast[DType.float32]()[0]
                let v1: Float32 = X[r, c0 + j + QK_MXFP4 // 2].cast[DType.float32]()[0]
                let i0: UInt8 = encode_mxfp4(v0, d)
                let i1: UInt8 = encode_mxfp4(v1, d)
                let packed: UInt8 = (UInt8(i1) << 4) | (i0 & 0x0F)
                Q[r, q_base + j] = SIMD[DType.uint8, QW](packed)

fn _mxfp4_quantize_kernel[
    x_dtype: DType, x_layout: Layout,
    q_dtype: DType, q_layout: Layout,
    e_dtype: DType, e_layout: Layout,
    BN: Int, BD: Int,
](X: LayoutTensor[x_dtype, x_layout, MutableAnyOrigin],
  Q: LayoutTensor[q_dtype, q_layout, MutableAnyOrigin],
  E: LayoutTensor[e_dtype, e_layout, MutableAnyOrigin]):
    var tile_x = X.tile[BN, BD](block_idx.y, block_idx.x)
    var tile_q = Q.tile[BN, BD // 2](block_idx.y, block_idx.x)
    var tile_e = E.tile[BN, BD // QK_MXFP4](block_idx.y, block_idx.x)
    let Ht = min_int(BN, X.shape[0]() - block_idx.y * BN)
    let Wt = min_int(BD, X.shape[1]() - block_idx.x * BD)
    let CBLK = Wt // QK_MXFP4
    if CBLK <= 0: return
    alias XW = X.element_layout.size()
    alias QW = Q.element_layout.size()
    alias EW = E.element_layout.size()
    let T = Int(block_dim.x)
    for r in range(Ht):
        var t = Int(thread_idx.x)
        while t < CBLK:
            let c0 = t * QK_MXFP4
            var m: Float32 = 0.0
            for j in range(QK_MXFP4):
                let f: Float32 = tile_x[r, c0 + j].cast[DType.float32]()[0]
                let af = (f if f >= 0.0 else -f)
                if af > m: m = af
            let e: UInt8 = fp32_to_e8m0_from_block_max(m)
            tile_e[r, t] = SIMD[DType.uint8, EW](e)
            let d: Float32 = e8m0_to_fp32(e)
            let q_base = t * (QK_MXFP4 // 2)
            for j in range(QK_MXFP4 // 2):
                let v0: Float32 = tile_x[r, c0 + j].cast[DType.float32]()[0]
                let v1: Float32 = tile_x[r, c0 + j + QK_MXFP4 // 2].cast[DType.float32]()[0]
                let i0: UInt8 = encode_mxfp4(v0, d)
                let i1: UInt8 = encode_mxfp4(v1, d)
                let packed: UInt8 = (UInt8(i1) << 4) | (i0 & 0x0F)
                tile_q[r, q_base + j] = SIMD[DType.uint8, QW](packed)
            t += T

def _mxfp4_quantize_gpu[BN: Int, BD: Int](
    ctx: DeviceContext, X: LayoutTensor, Q: LayoutTensor, E: LayoutTensor):
    alias kernel = _mxfp4_quantize_kernel[
        X.dtype, X.layout, Q.dtype, Q.layout, E.dtype, E.layout, BN, BD
    ]
    let cblk_per_tile = BD // QK_MXFP4
    let tpb = max_int(1, min_int(32, cblk_per_tile))
    ctx.enqueue_function[kernel](
        X, Q, E,
        grid_dim=(ceil_div(X.shape[1](), BD), ceil_div(X.shape[0](), BN)),
        block_dim=tpb,
    )

# -----------------------------------------------------------------------------
# DEQUANTIZE: (Q,E) -> X
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_dequantize_exq")
struct MXFP4DequantizeEXQ:
    @staticmethod
    fn execute[out_dtype: DType, rank: Int, BN: Int, BD: Int, target: StaticString](
        out_x: OutputTensor[dtype=out_dtype, rank=rank],
        q: InputTensor[dtype = DType.uint8, rank=rank],
        e: InputTensor[dtype = DType.uint8, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        constrained[out_dtype == DType.float32, "dequantize outputs float32"]()
        var Q = q.to_layout_tensor()
        var E = e.to_layout_tensor()
        var X = out_x.to_layout_tensor()
        alias W = X.shape[1]()
        constrained[W % QK_MXFP4 == 0, "W must be divisible by 32"]()
        @parameter
        if target == "cpu":
            _mxfp4_dequantize_cpu(Q, E, X)
        else:
            var dev = ctx.get_device_context()
            _mxfp4_dequantize_gpu[BN, BD](dev, Q, E, X)

fn _mxfp4_dequantize_cpu(Q: LayoutTensor, E: LayoutTensor, mut X: LayoutTensor):
    alias H = X.shape[0]()
    alias W = X.shape[1]()
    let blocks_per_row = W // QK_MXFP4
    alias XD = X.dtype
    alias XW = X.element_layout.size()
    alias QW = Q.element_layout.size()
    alias EW = E.element_layout.size()
    for r in range(H):
        for b in range(blocks_per_row):
            let c0 = b * QK_MXFP4
            let e_lane0: UInt8 = E[r, b].cast[DType.uint8]()[0]
            let d: Float32 = e8m0_to_fp32(e_lane0)
            let q_base = b * (QK_MXFP4 // 2)
            for j in range(QK_MXFP4 // 2):
                let byte_val0: UInt8 = Q[r, q_base + j].cast[DType.uint8]()[0]
                let i0: UInt8 = byte_val0 & UInt8(0x0F)
                let i1: UInt8 = byte_val0 >> 4
                let v0c: SIMD[XD, 1] = (unit_from_code(i0) * d).cast[XD]()
                let v1c: SIMD[XD, 1] = (unit_from_code(i1) * d).cast[XD]()
                X[r, c0 + j] = SIMD[XD, XW](v0c)
                X[r, c0 + j + QK_MXFP4 // 2] = SIMD[XD, XW](v1c)

fn _mxfp4_dequantize_kernel[
    q_dtype: DType, q_layout: Layout,
    e_dtype: DType, e_layout: Layout,
    x_dtype: DType, x_layout: Layout,
    BN: Int, BD: Int,
](Q: LayoutTensor[q_dtype, q_layout, MutableAnyOrigin],
  E: LayoutTensor[e_dtype, e_layout, MutableAnyOrigin],
  X: LayoutTensor[x_dtype, x_layout, MutableAnyOrigin]):
    var tile_q = Q.tile[BN, BD // 2](block_idx.y, block_idx.x)
    var tile_e = E.tile[BN, BD // QK_MXFP4](block_idx.y, block_idx.x)
    var tile_x = X.tile[BN, BD](block_idx.y, block_idx.x)
    let Ht = min_int(BN, X.shape[0]() - block_idx.y * BN)
    let Wt = min_int(BD, X.shape[1]() - block_idx.x * BD)
    let CBLK = Wt // QK_MXFP4
    if CBLK <= 0: return
    alias XD = X.dtype
    alias XW = X.element_layout.size()
    let T = Int(block_dim.x)
    for r in range(Ht):
        var t = Int(thread_idx.x)
        while t < CBLK:
            let d: Float32 = e8m0_to_fp32(tile_e[r, t].cast[DType.uint8]()[0])
            let q_base = t * (QK_MXFP4 // 2)
            let x_base = t * QK_MXFP4
            for j in range(QK_MXFP4 // 2):
                let byte0: UInt8 = tile_q[r, q_base + j].cast[DType.uint8]()[0]
                let i0: UInt8 = byte0 & UInt8(0x0F)
                let i1: UInt8 = byte0 >> 4
                let v0c: SIMD[XD, 1] = (unit_from_code(i0) * d).cast[XD]()
                let v1c: SIMD[XD, 1] = (unit_from_code(i1) * d).cast[XD]()
                tile_x[r, x_base + j] = SIMD[XD, XW](v0c)
                tile_x[r, x_base + j + QK_MXFP4 // 2] = SIMD[XD, XW](v1c)
            t += T

def _mxfp4_dequantize_gpu[BN: Int, BD: Int](
    ctx: DeviceContext, Q: LayoutTensor, E: LayoutTensor, X: LayoutTensor):
    alias kernel = _mxfp4_dequantize_kernel[
        Q.dtype, Q.layout, E.dtype, E.layout, X.dtype, X.layout, BN, BD
    ]
    let cblk_per_tile = BD // QK_MXFP4
    let tpb = max_int(1, min_int(32, cblk_per_tile))
    ctx.enqueue_function[kernel](
        Q, E, X,
        grid_dim=(ceil_div(X.shape[1](), BD), ceil_div(X.shape[0](), BN)),
        block_dim=tpb,
    )

# -----------------------------------------------------------------------------
# FUSED QMATVEC: (Q[H,W/2],E[H,W/32]) · x[W] -> y[H]
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_qmatvec_f32_exq")
struct MXFP4QMatVecF32EXQ:
    @staticmethod
    fn execute[in_dtype: DType, rank_q: Int, rank_e: Int, target: StaticString](
        out_y: OutputTensor[dtype = DType.float32, rank=1],
        q: InputTensor[dtype = DType.uint8, rank=rank_q],
        e: InputTensor[dtype = DType.uint8, rank=rank_e],
        x: InputTensor[dtype=in_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[in_dtype == DType.float32, "x must be float32"]()
        var Q = q.to_layout_tensor()
        var E = e.to_layout_tensor()
        var Xv = x.to_layout_tensor()
        var Y = out_y.to_layout_tensor()
        @parameter
        if target == "cpu":
            _mxfp4_matvec_cpu(Q, E, Xv, Y)
        else:
            var dev = ctx.get_device_context()
            _mxfp4_matvec_gpu(dev, Q, E, Xv, Y)

fn _mxfp4_matvec_cpu(Q: LayoutTensor, E: LayoutTensor, Xv: LayoutTensor, mut Y: LayoutTensor):
    alias H = Q.shape[0]()
    alias W2 = Q.shape[1]()
    let W = W2 * 2
    let blocks = W // QK_MXFP4
    for r in range(H):
        var acc: Float32 = 0.0
        for b in range(blocks):
            let d: Float32 = e8m0_to_fp32(E[r, b].cast[DType.uint8]()[0])
            let q_base = b * (QK_MXFP4 // 2)
            let x_base = b * QK_MXFP4
            for j in range(QK_MXFP4 // 2):
                let byte0: UInt8 = Q[r, q_base + j].cast[DType.uint8]()[0]
                let i0: UInt8 = byte0 & UInt8(0x0F)
                let i1: UInt8 = byte0 >> 4
                let v0 = unit_from_code(i0) * d
                let v1 = unit_from_code(i1) * d
                let x0: Float32 = Xv[x_base + j].cast[DType.float32]()[0]
                let x1: Float32 = Xv[x_base + j + QK_MXFP4 // 2].cast[DType.float32]()[0]
                acc += v0 * x0 + v1 * x1
        Y[r] = acc

# Tiled GPU matvec: stage x in shared, dequant on the fly.
fn _mxfp4_matvec_kernel[
    q_dtype: DType, q_layout: Layout,
    e_dtype: DType, e_layout: Layout,
    x_dtype: DType, x_layout: Layout,
    y_dtype: DType, y_layout: Layout,
](
    Q: LayoutTensor[q_dtype, q_layout],
    E: LayoutTensor[e_dtype, e_layout],
    X: LayoutTensor[x_dtype, x_layout],
    Y: LayoutTensor[y_dtype, y_layout],
):
    alias YD = Y.dtype
    alias YW = Y.element_layout.size()
    let r = Int(block_idx.y) * Int(block_dim.y) + Int(thread_idx.y)
    if r >= Q.shape[0](): return
    let W = Q.shape[1]() * 2
    let blocks = W // QK_MXFP4
    # BK tile along K (here: blocks of 32 * TILES)
    let TILE_K_BLKS = 8  # 8 * 32 = 256 elements per tile
    var acc: Float32 = 0.0
    # shared stage of x for a tile (256 floats)
    var xs = shared_array[Float32, TILE_K_BLKS * QK_MXFP4]()
    var b0 = 0
    while b0 < blocks:
        let bmax = min_int(blocks, b0 + TILE_K_BLKS)
        # load x tile
        var t = Int(thread_idx.x)
        let lane_cnt = Int(block_dim.x)
        var load_idx = t
        while load_idx < (bmax - b0) * QK_MXFP4:
            let g = b0 * QK_MXFP4 + load_idx
            xs[load_idx] = X[g].cast[DType.float32]()[0]
            load_idx += lane_cnt
        syncthreads()
        # consume tile
        for tb in range(b0, bmax):
            let d: Float32 = e8m0_to_fp32(E[r, tb].cast[DType.uint8]()[0])
            let q_base = tb * (QK_MXFP4 // 2)
            let x_base = (tb - b0) * QK_MXFP4
            for j in range(QK_MXFP4 // 2):
                let byte0: UInt8 = Q[r, q_base + j].cast[DType.uint8]()[0]
                let i0: UInt8 = byte0 & UInt8(0x0F)
                let i1: UInt8 = byte0 >> 4
                let v0 = unit_from_code(i0) * d
                let v1 = unit_from_code(i1) * d
                acc += v0 * xs[x_base + j] + v1 * xs[x_base + j + QK_MXFP4 // 2]
        syncthreads()
        b0 += TILE_K_BLKS
    Y[r] = SIMD[YD, YW](acc.cast[YD]())

def _mxfp4_matvec_gpu(ctx: DeviceContext, Q: LayoutTensor, E: LayoutTensor, X: LayoutTensor, Y: LayoutTensor):
    alias kernel = _mxfp4_matvec_kernel[
        Q.dtype, Q.layout, E.dtype, E.layout, X.dtype, X.layout, Y.dtype, Y.layout
    ]
    let grid_y = Q.shape[0]()
    ctx.enqueue_function[kernel](
        Q, E, X, Y,
        grid_dim=(1, grid_y),
        block_dim=(32, 1),  # 32 threads to load x tiles
    )

# -----------------------------------------------------------------------------
# FUSED QMATMUL: (Q[A=H,K/2],E[H,K/32]) × B[K,N] → Y[H,N], float32 accum
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_qmatmul_f32_exq")
struct MXFP4QMatMulF32EXQ:
    @staticmethod
    fn execute[in_dtype: DType, rank_q: Int, rank_e: Int, target: StaticString](
        out_y: OutputTensor[dtype = DType.float32, rank=2],
        q: InputTensor[dtype = DType.uint8, rank=rank_q],
        e: InputTensor[dtype = DType.uint8, rank=rank_e],
        b: InputTensor[dtype = in_dtype, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[in_dtype == DType.float32, "B must be float32"]()
        var Q = q.to_layout_tensor()
        var E = e.to_layout_tensor()
        var B = b.to_layout_tensor()
        var Y = out_y.to_layout_tensor()
        # Dim checks
        alias H = Q.shape[0]()
        alias K2 = Q.shape[1]()
        alias K = K2 * 2
        constrained[B.shape[0]() == K, "K mismatch"]()
        constrained[Y.shape[0]() == H, "H mismatch"]()
        constrained[Y.shape[1]() == B.shape[1](), "N mismatch"]()
        @parameter
        if target == "cpu":
            _mxfp4_matmul_cpu(Q, E, B, Y)
        else:
            var dev = ctx.get_device_context()
            _mxfp4_matmul_gpu(dev, Q, E, B, Y)

fn _mxfp4_matmul_cpu(Q: LayoutTensor, E: LayoutTensor, B: LayoutTensor, mut Y: LayoutTensor):
    alias H = Q.shape[0]()
    alias K2 = Q.shape[1]()
    alias K = K2 * 2
    alias N = B.shape[1]()
    let blks = K // QK_MXFP4
    for m in range(H):
        for n in range(N):
            var acc: Float32 = 0.0
            for b in range(blks):
                let d: Float32 = e8m0_to_fp32(E[m, b].cast[DType.uint8]()[0])
                let q_base = b * (QK_MXFP4 // 2)
                let k_base = b * QK_MXFP4
                for j in range(QK_MXFP4 // 2):
                    let byte0: UInt8 = Q[m, q_base + j].cast[DType.uint8]()[0]
                    let i0: UInt8 = byte0 & UInt8(0x0F)
                    let i1: UInt8 = byte0 >> 4
                    let v0: Float32 = unit_from_code(i0) * d
                    let v1: Float32 = unit_from_code(i1) * d
                    let b0: Float32 = B[k_base + j, n].cast[DType.float32]()[0]
                    let b1: Float32 = B[k_base + j + QK_MXFP4 // 2, n].cast[DType.float32]()[0]
                    acc += v0 * b0 + v1 * b1
            Y[m, n] = acc

# GPU tiled: BMxBN block of Y, tile B into shared, dequant Q on the fly.
fn _mxfp4_matmul_kernel[
    q_dtype: DType, q_layout: Layout,
    e_dtype: DType, e_layout: Layout,
    b_dtype: DType, b_layout: Layout,
    y_dtype: DType, y_layout: Layout,
](
    Q: LayoutTensor[q_dtype, q_layout],
    E: LayoutTensor[e_dtype, e_layout],
    B: LayoutTensor[b_dtype, b_layout],
    Y: LayoutTensor[y_dtype, y_layout],
):
    alias YD = Y.dtype
    alias YW = Y.element_layout.size()
    let H = Q.shape[0]()
    let K = Q.shape[1]() * 2
    let N = B.shape[1]()
    if (K == 0) or (N == 0) or (H == 0): return
    # Blocking
    let BM = 64
    let BN = 64
    let BK = 128   # must be multiple of 32
    let m0 = Int(block_idx.y) * BM
    let n0 = Int(block_idx.x) * BN
    # Thread mapping: 16x16 = 256 threads per block ≈ 4x4 accum per thread
    let tx = Int(thread_idx.x)  # [0,15]
    let ty = Int(thread_idx.y)  # [0,15]
    # Shared tile for B: BK x BN
    var sB = shared_array[Float32, BK * BN]()
    var acc = [Float32; 4, 4]()
    for ii in range(4):
        for jj in range(4):
            acc[ii][jj] = 0.0
    var k0 = 0
    while k0 < K:
        # Load B tile into shared (coalesced)
        var load = tx + ty * 16
        while load < BK * BN:
            let kk = load / BN
            let nn = load - kk * BN
            let gk = k0 + kk
            let gn = n0 + nn
            var val: Float32 = 0.0
            if (gk < K) and (gn < N):
                val = B[gk, gn].cast[DType.float32]()[0]
            sB[kk * BN + nn] = val
            load += 256
        syncthreads()
        # Dequant-and-dot for this m-tile
        for ib in range(4):    # 4 rows per thread
            let m = m0 + ty * 4 + ib
            if m >= H: continue
            var kk = 0
            while kk < BK:
                let k_abs = k0 + kk
                if k_abs >= K: break
                # compute block index in MXFP4 space
                let blk = k_abs // QK_MXFP4
                let d: Float32 = e8m0_to_fp32(E[m, blk].cast[DType.uint8]()[0])
                let q_base = blk * (QK_MXFP4 // 2)
                let off_in_blk = k_abs - blk * QK_MXFP4
                # pick up two codes i0/i1 and dequant to v0/v1 for k_abs and k_abs+16
                let half = off_in_blk // (QK_MXFP4 // 2)
                let j = off_in_blk % (QK_MXFP4 // 2)
                let byte0: UInt8 = Q[m, q_base + j].cast[DType.uint8]()[0]
                let i0: UInt8 = byte0 & UInt8(0x0F)
                let i1: UInt8 = byte0 >> 4
                let v = (unit_from_code((i0 if half == 0 else i1)) * d)
                # accumulate across BN in 4 columns per thread
                for jb in range(4):
                    let n = n0 + tx * 4 + jb
                    if n >= N: continue
                    let bval: Float32 = sB[kk * BN + (n - n0)]
                    acc[ib][jb] += v * bval
                kk += 1
            end
        end
        syncthreads()
        k0 += BK
    end
    # write back
    for ib in range(4):
        let m = m0 + ty * 4 + ib
        if m >= H: continue
        for jb in range(4):
            let n = n0 + tx * 4 + jb
            if n >= N: continue
            Y[m, n] = SIMD[YD, YW](acc[ib][jb].cast[YD]())
        end
    end

def _mxfp4_matmul_gpu(ctx: DeviceContext, Q: LayoutTensor, E: LayoutTensor, B: LayoutTensor, Y: LayoutTensor):
    alias kernel = _mxfp4_matmul_kernel[
        Q.dtype, Q.layout, E.dtype, E.layout, B.dtype, B.layout, Y.dtype, Y.layout
    ]
    let BM = 64
    let BN = 64
    let grid_x = ceil_div(Y.shape[1](), BN)
    let grid_y = ceil_div(Y.shape[0](), BM)
    ctx.enqueue_function[kernel](
        Q, E, B, Y,
        grid_dim=(grid_x, grid_y),
        block_dim=(16, 16),
    )
