from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, ops


class MXFP4Linear:
    """
    MAX-graph friendly linear: Y[M,N] = (Q[M,K/2], E[M,K/32]) x X[K,N].
    - If q/e are None, we quantize 'w_f32' once at build time on the GPU.
    - Bias is optional and kept in fp32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device,
        mojo_kernels_dir: Path,
        *,
        q: Optional[np.ndarray] = None,  # uint8 [M, K/2]
        e: Optional[np.ndarray] = None,  # uint8 [M, K/32]
        w_f32: Optional[np.ndarray] = None,  # float32 [M, K]
        bias_f32: Optional[
            np.ndarray
        ] = None,  # float32 [M] applied after matmul(X is [K,N])
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.device_ref = getattr(device, "device_ref", device)
        self.kernels_dir = Path(mojo_kernels_dir)
        assert (q is not None and e is not None) or (w_f32 is not None), (
            "Provide either (q,e) or (w_f32) for quantization."
        )
        self._q = q
        self._e = e
        self._w = w_f32
        self._bias = bias_f32

    def build(self):
        """
        Returns a callable closure that maps X[K,N] -> Y[M,N].
        """
        QK = 32
        M = self.out_features
        K = self.in_features
        # Prepare constant tensors/types for the graph
        out_type = TensorType(
            dtype=DType.float32,
            shape=[M, None],  # N is dynamic (batching-friendly)
            device=self.device_ref,
        )

        def forward(x):
            # x: [K,N] float32
            # quantize if needed (once) on the first call
            nonlocal self
            if self._q is None or self._e is None:
                # Quantize w_f32 -> q,e using custom op
                assert self._w is not None and self._w.shape == (M, K)
                # Construct outputs with shapes q[M,K/2], e[M,K/32]
                q_type = TensorType(DType.uint8, [M, K // 2], self.device_ref)
                e_type = TensorType(DType.uint8, [M, K // QK], self.device_ref)
                q, e = ops.custom(
                    name="modular_ops::mxfp4_quantize_exq",
                    device=self.device_ref,
                    values=[
                        ops.constant(
                            self._w.astype(np.float32), device=self.device_ref
                        )
                    ],
                    out_types=[q_type, e_type],
                )
                self._q, self._e = q.tensor, e.tensor
            else:
                # make constants once
                if isinstance(self._q, np.ndarray):
                    self._q = ops.constant(self._q, device=self.device_ref).tensor
                if isinstance(self._e, np.ndarray):
                    self._e = ops.constant(self._e, device=self.device_ref).tensor
            y = ops.custom(
                name="modular_ops::mxfp4_qmatmul_f32_exq",
                device=self.device_ref,
                values=[self._q, self._e, x],
                out_types=[out_type],
            )[0].tensor
            if self._bias is not None:
                b = ops.constant(
                    self._bias.astype(np.float32), device=self.device_ref
                ).tensor
                # y += b[:, None]
                y = ops.add(y, ops.broadcast_to(b, [M, ops.shape(y)[1]]))
            return y
        # encase in a small graph to allow reuse
        graph = Graph(
            name="mxfp4_linear",
            forward=forward,
            input_types=[TensorType(DType.float32, [K, None], self.device_ref)],
            custom_extensions=[self.kernels_dir],  # this compiles/loads our Mojo ops
        )
        return graph
