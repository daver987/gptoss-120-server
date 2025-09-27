"""MXFP4 Quantization and Dequantization Operations"""
from math import floor, log2, ldexp
import compiler
from layout import Layout, LayoutTensor
from max.tensor import InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from gpu.host import DeviceContext
from gpu import block_idx, thread_idx, block_dim

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
alias QK_MXFP4: Int = 32


@always_inline
fn min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


@always_inline
fn max_int(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b


@always_inline
fn clamp_f32(x: Float32, lo: Float32, hi: Float32) -> Float32:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@always_inline
fn ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


@always_inline
fn e8m0_to_fp32(e: UInt8) -> Float32:
    return ldexp(Float32(1.0), Int(e) - 127)


@always_inline
fn fp32_to_e8m0_from_block_max(max_val: Float32) -> UInt8:
    if max_val <= 0.0:
        return 0
    if max_val < 1e-38:
        return 0
    var e_est: Float32 = floor(log2(max_val)) - 2.5850000 + 127.0
    return UInt8(clamp_f32(e_est, 1.0, 254.0))


@always_inline
fn unit_from_code(code: UInt8) -> Float32:
    var c: Int = Int(code)
    if c == 0:
        return 0.0
    elif c == 1:
        return 0.5
    elif c == 2:
        return 1.0
    elif c == 3:
        return 1.5
    elif c == 4:
        return 2.0
    elif c == 5:
        return 3.0
    elif c == 6:
        return 4.0
    elif c == 7:
        return 6.0
    elif c == 8:
        return -0.0
    elif c == 9:
        return -0.5
    elif c == 10:
        return -1.0
    elif c == 11:
        return -1.5
    elif c == 12:
        return -2.0
    elif c == 13:
        return -3.0
    elif c == 14:
        return -4.0
    else:
        return -6.0


@always_inline
fn encode_mxfp4(val: Float32, d: Float32) -> UInt8:
    if d == 0.0:
        return 0
    var a = val / d
    var sign: UInt8 = 0
    if a < 0.0:
        sign = 0x8
        a = -a
    var k: UInt8 = 0
    if a >= 0.25:
        k = 1
    if a >= 0.75:
        k = 2
    if a >= 1.25:
        k = 3
    if a >= 1.75:
        k = 4
    if a >= 2.5:
        k = 5
    if a >= 3.5:
        k = 6
    if a >= 5.0:
        k = 7
    return sign | k


# -----------------------------------------------------------------------------
# QUANTIZE: X[H,W] -> Q[H,W/2] + E[H,W/32]
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_quantize_exq")
struct MXFP4QuantizeEXQ:
    @staticmethod
    fn execute[
        in_dtype: DType, rank: Int, BN: Int, BD: Int, target: StaticString
    ](
        out_q: OutputTensor[dtype = DType.uint8, rank=rank],
        out_e: OutputTensor[dtype = DType.uint8, rank=rank],
        x: InputTensor[dtype=in_dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        constrained[
            in_dtype == DType.float32, "quantize expects float32 input"
        ]()
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


fn _mxfp4_quantize_cpu(
    X: LayoutTensor, mut Q: LayoutTensor, mut E: LayoutTensor
):
    alias H = X.shape[0]()
    alias W = X.shape[1]()
    let blocks_per_row = W // QK_MXFP4

    # Bind element widths (compile-time) for constructing SIMD values
    alias XW = X.element_layout.size()
    alias QW = Q.element_layout.size()
    alias EW = E.element_layout.size()

    for r in range(H):
        for b in range(blocks_per_row):
            let c0 = b * QK_MXFP4
            var m: Float32 = 0.0
            for j in range(QK_MXFP4):
                # READ: lane 0 of SIMD element → scalar (Float32 is SIMD[f32,1] alias)
                let f: Float32 = X[r, c0 + j].cast[DType.float32]()[0]
                let af = (f if f >= 0.0 else -f)
                if af > m:
                    m = af
            let e: UInt8 = fp32_to_e8m0_from_block_max(m)
            # WRITE: splat e as SIMD[u8, EW]
            E[r, b] = SIMD[DType.uint8, EW](e)
            let d: Float32 = e8m0_to_fp32(e)

            let q_base = b * (QK_MXFP4 // 2)
            for j in range(QK_MXFP4 // 2):
                let v0: Float32 = X[r, c0 + j].cast[DType.float32]()[0]
                let v1: Float32 = X[r, c0 + j + QK_MXFP4 // 2].cast[DType.float32]()[0]
                let i0: UInt8 = encode_mxfp4(v0, d)
                let i1: UInt8 = encode_mxfp4(v1, d)
                let packed: UInt8 = (UInt8(i1) << 4) | (i0 & 0x0F)
                # WRITE: splat into SIMD[u8, QW]
                Q[r, q_base + j] = SIMD[DType.uint8, QW](packed)


fn _mxfp4_quantize_kernel[
    x_dtype: DType,
    x_layout: Layout,
    q_dtype: DType,
    q_layout: Layout,
    e_dtype: DType,
    e_layout: Layout,
    BN: Int,
    BD: Int,
](
    X: LayoutTensor[x_dtype, x_layout, MutableAnyOrigin],
    Q: LayoutTensor[q_dtype, q_layout, MutableAnyOrigin],
    E: LayoutTensor[e_dtype, e_layout, MutableAnyOrigin],
):
    var tile_x = X.tile[BN, BD](block_idx.y, block_idx.x)
    var tile_q = Q.tile[BN, BD // 2](block_idx.y, block_idx.x)
    var tile_e = E.tile[BN, BD // QK_MXFP4](block_idx.y, block_idx.x)

    let Ht = min_int(BN, X.shape[0]() - block_idx.y * BN)
    let Wt = min_int(BD, X.shape[1]() - block_idx.x * BD)
    let CBLK = Wt // QK_MXFP4
    if CBLK <= 0:
        return

    # Bind element widths for SIMD construction
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
                if af > m:
                    m = af
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


def _mxfp4_quantize_gpu[
    BN: Int, BD: Int
](ctx: DeviceContext, X: LayoutTensor, Q: LayoutTensor, E: LayoutTensor):
    alias kernel = _mxfp4_quantize_kernel[
        X.dtype, X.layout, Q.dtype, Q.layout, E.dtype, E.layout, BN, BD
    ]
    let cblk_per_tile = BD // QK_MXFP4
    let tpb = max_int(1, min_int(32, cblk_per_tile))
    ctx.enqueue_function[kernel](
        X,
        Q,
        E,
        grid_dim=(ceil_div(X.shape[1](), BD), ceil_div(X.shape[0](), BN)),
        block_dim=tpb,
    )


# -----------------------------------------------------------------------------
# DEQUANTIZE: (Q[H,W/2], E[H,W/32]) -> X[H,W]
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_dequantize_exq")
struct MXFP4DequantizeEXQ:
    @staticmethod
    fn execute[
        out_dtype: DType, rank: Int, BN: Int, BD: Int, target: StaticString
    ](
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

    alias QW = Q.element_layout.size()
    alias EW = E.element_layout.size()
    alias XD = X.dtype
    alias XW = X.element_layout.size()

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
                let v0f: Float32 = unit_from_code(i0) * d
                let v1f: Float32 = unit_from_code(i1) * d
                # Cast Float32 -> XD, then splat to XW on write
                let v0c: SIMD[XD, 1] = v0f.cast[XD]()
                let v1c: SIMD[XD, 1] = v1f.cast[XD]()
                X[r, c0 + j] = SIMD[XD, XW](v0c)
                X[r, c0 + j + QK_MXFP4 // 2] = SIMD[XD, XW](v1c)


fn _mxfp4_dequantize_kernel[
    q_dtype: DType,
    q_layout: Layout,
    e_dtype: DType,
    e_layout: Layout,
    x_dtype: DType,
    x_layout: Layout,
    BN: Int,
    BD: Int,
](
    Q: LayoutTensor[q_dtype, q_layout, MutableAnyOrigin],
    E: LayoutTensor[e_dtype, e_layout, MutableAnyOrigin],
    X: LayoutTensor[x_dtype, x_layout, MutableAnyOrigin],
):
    var tile_q = Q.tile[BN, BD // 2](block_idx.y, block_idx.x)
    var tile_e = E.tile[BN, BD // QK_MXFP4](block_idx.y, block_idx.x)
    var tile_x = X.tile[BN, BD](block_idx.y, block_idx.x)

    let Ht = min_int(BN, X.shape[0]() - block_idx.y * BN)
    let Wt = min_int(BD, X.shape[1]() - block_idx.x * BD)
    let CBLK = Wt // QK_MXFP4
    if CBLK <= 0:
        return

    alias XD = X.dtype
    alias XW = X.element_layout.size()
    alias QW = Q.element_layout.size()
    alias EW = E.element_layout.size()

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


def _mxfp4_dequantize_gpu[
    BN: Int, BD: Int
](ctx: DeviceContext, Q: LayoutTensor, E: LayoutTensor, X: LayoutTensor):
    alias kernel = _mxfp4_dequantize_kernel[
        Q.dtype, Q.layout, E.dtype, E.layout, X.dtype, X.layout, BN, BD
    ]
    let cblk_per_tile = BD // QK_MXFP4
    let tpb = max_int(1, min_int(32, cblk_per_tile))
    ctx.enqueue_function[kernel](
        Q,
        E,
        X,
        grid_dim=(ceil_div(X.shape[1](), BD), ceil_div(X.shape[0](), BN)),
        block_dim=tpb,
    )


# -----------------------------------------------------------------------------
# FUSED DEQUANT-DOT: (Q[H,W/2], E[H,W/32]) · x[W] -> y[H]
# -----------------------------------------------------------------------------
@compiler.register("modular_ops::mxfp4_qmatvec_f32_exq")
struct MXFP4QMatVecF32EXQ:
    @staticmethod
    fn execute[
        in_dtype: DType, rank_q: Int, rank_e: Int, target: StaticString
    ](
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


fn _mxfp4_matvec_cpu(
    Q: LayoutTensor, E: LayoutTensor, Xv: LayoutTensor, mut Y: LayoutTensor
):
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
        # Y is rank-1 float32; when run on CPU path we can still assign scalar here;
        # (GPU kernel scope handles SIMD write explicitly.)
        Y[r] = acc


fn _mxfp4_matvec_kernel[
    q_dtype: DType,
    q_layout: Layout,
    e_dtype: DType,
    e_layout: Layout,
    x_dtype: DType,
    x_layout: Layout,
    y_dtype: DType,
    y_layout: Layout,
](
    Q: LayoutTensor[q_dtype, q_layout],
    E: LayoutTensor[e_dtype, e_layout],
    X: LayoutTensor[x_dtype, x_layout],
    Y: LayoutTensor[y_dtype, y_layout],
):
    let r = Int(block_idx.y) * Int(block_dim.y) + Int(thread_idx.y)
    if r >= Q.shape[0]():
        return

    alias YD = Y.dtype
    alias YW = Y.element_layout.size()

    let W = Q.shape[1]() * 2
    let blocks = W // QK_MXFP4
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
            let x0: Float32 = X[x_base + j].cast[DType.float32]()[0]
            let x1: Float32 = X[x_base + j + QK_MXFP4 // 2].cast[DType.float32]()[0]
            acc += v0 * x0 + v1 * x1
    # WRITE: splat acc into Y's element width/type
    Y[r] = SIMD[YD, YW](acc.cast[YD]())


def _mxfp4_matvec_gpu(
    ctx: DeviceContext,
    Q: LayoutTensor,
    E: LayoutTensor,
    X: LayoutTensor,
    Y: LayoutTensor,
):
    alias kernel = _mxfp4_matvec_kernel[
        Q.dtype,
        Q.layout,
        E.dtype,
        E.layout,
        X.dtype,
        X.layout,
        Y.dtype,
        Y.layout,
    ]
    let grid_y = Q.shape[0]()
    ctx.enqueue_function[kernel](
        Q,
        E,
        X,
        Y,
        grid_dim=(1, grid_y),
        block_dim=(1, 1),
    )
