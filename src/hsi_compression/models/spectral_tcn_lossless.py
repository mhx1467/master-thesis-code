from __future__ import annotations

import json
import math
import struct
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional lossless dependency
    zstd = None


class ResidualCausalTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for a causal TCN block")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left padding makes the convolution causal: each band sees only previous bands.
        y = self.conv(F.pad(x, (self.left_padding, 0)))
        y = self.proj(self.dropout(self.act(y)))
        return x + y

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.left_padding == 0:
            return torch.empty(batch_size, self.channels, 0, device=device, dtype=dtype)
        # state stores the past hidden values needed for one-step decoding.
        return torch.zeros(batch_size, self.channels, self.left_padding, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_t_3d = x_t.unsqueeze(-1)
        if self.left_padding > 0:
            # keep only past values so sequential decoding stays causal
            window = torch.cat((state, x_t_3d), dim=-1)
            next_state = window[:, :, 1:]
        else:
            window = x_t_3d
            next_state = state

        y_t = F.conv1d(
            window,
            self.conv.weight,
            self.conv.bias,
            dilation=self.dilation,
        )
        y_t = self.proj(self.dropout(self.act(y_t)))
        return (x_t_3d + y_t).squeeze(-1), next_state


class SpectralTCNLossless(nn.Module):
    compression_mode = "lossless"
    supports_actual_compression = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 48,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
        output_activation: str | None = "sigmoid",
        symbol_scale: int = 10000,
        zlib_level: int = 9,
        raw_fallback: bool = True,
        pixels_per_patch: int | None = None,
        prediction_mode: str = "value",
        residual_backend: str = "zlib",
        residual_transform: str = "none",
    ):
        super().__init__()
        if in_channels <= 1:
            raise ValueError("in_channels must be > 1 for spectral prediction")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if symbol_scale <= 0:
            raise ValueError("symbol_scale must be > 0")
        if not (0 <= zlib_level <= 9):
            raise ValueError("zlib_level must be in [0, 9]")
        if pixels_per_patch is not None and pixels_per_patch <= 0:
            raise ValueError("pixels_per_patch must be positive or None")
        if prediction_mode not in {"value", "delta"}:
            raise ValueError("prediction_mode must be one of: 'value', 'delta'")
        if residual_backend not in {"zlib", "zstd"}:
            raise ValueError("residual_backend must be one of: 'zlib', 'zstd'")
        residual_transform = self._normalize_residual_transform(residual_transform)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.symbol_scale = int(symbol_scale)
        self.zlib_level = int(zlib_level)
        self.raw_fallback = raw_fallback
        self.pixels_per_patch = pixels_per_patch
        self.prediction_mode = prediction_mode
        self.residual_backend = residual_backend
        self.residual_transform = residual_transform

        # input projection turns the previous scalar band value into hidden tcn channels.
        self.input_proj = nn.Conv1d(1, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualCausalTCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
                for block_idx in range(num_blocks)
            ]
        )
        # dilations grow as powers of two so later blocks see a wider spectral history.
        self.head_act = nn.GELU()
        self.output_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        if output_activation == "sigmoid":
            # value mode predicts normalized values in 0..1.
            self.output_head = nn.Sigmoid()
        elif output_activation == "tanh":
            # delta mode can predict negative or positive spectral differences.
            self.output_head = nn.Tanh()
        elif output_activation in (None, "identity"):
            self.output_head = nn.Identity()
        else:
            raise ValueError(
                "output_activation must be one of: 'sigmoid', 'tanh', 'identity', None"
            )

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        symbols = self._to_symbols(x)
        # training predicts each band from previous known bands, then compares with target.
        teacher, x_target = self._teacher_and_target_from_symbols(symbols)

        mask_for_loss = valid_mask
        if self.pixels_per_patch is not None:
            # sampling limits memory because every pixel is a separate spectral sequence.
            teacher, x_target, mask_for_loss = self._sample_pixel_sequences(
                teacher=teacher,
                target=x_target,
                valid_mask=valid_mask,
                pixels_per_patch=self.pixels_per_patch,
            )

        x_hat = self._predict_from_teacher_values(teacher)
        return {
            "x_hat": x_hat,
            "x_target": x_target,
            "mask_for_loss": mask_for_loss,
        }

    def update(self, force: bool = False) -> bool:
        del force
        return False

    def compress(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        residual_backend: str | None = None,
        residual_transform: str | None = None,
        codec_backend: str | None = None,
        compression_level: int | None = None,
    ) -> dict[str, bytes | tuple[int, ...]]:
        del valid_mask
        x_float = x.detach().float()
        # lossless residual coding works on integer symbols, not directly on floats.
        symbols = self._to_symbols(x_float)

        is_exact_symbol_grid = self._is_exact_symbol_grid(x_float, symbols)
        if self.raw_fallback and not is_exact_symbol_grid:
            # raw fallback preserves exact float input when symbol grid coding is invalid
            strings = self._pack_array(
                header={
                    "compression_mode": self.compression_mode,
                    "codec_backend": "zlib_raw_float32",
                    "dtype": "float32",
                    "shape": list(x_float.shape),
                },
                array=np.ascontiguousarray(x_float.cpu().numpy().astype(np.float32)),
            )
            return {"strings": strings, "shape": tuple(x.shape)}
        if not is_exact_symbol_grid:
            raise ValueError(
                "Input is not exactly representable on the configured symbol grid "
                f"(symbol_scale={self.symbol_scale}). Enable raw_fallback for exact "
                "float32 coding or use symbol-grid inputs for predictive residual coding."
            )

        residuals = self._residuals_from_symbols(symbols)
        strings = self._pack_residuals(
            residuals=residuals,
            residual_backend=residual_backend or self.residual_backend,
            residual_transform=residual_transform or self.residual_transform,
            codec_backend=codec_backend,
            compression_level=compression_level,
        )
        return {"strings": strings, "shape": tuple(x.shape)}

    def decompress(
        self,
        strings,
        shape,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del shape, kwargs
        header, payload = self._unpack_payload(strings)
        device = next(self.parameters()).device

        if header["codec_backend"] == "zlib_raw_float32":
            # raw fallback restores the original float values exactly.
            array = np.frombuffer(payload, dtype=np.float32).copy().reshape(header["shape"])
            x_hat = torch.from_numpy(array).to(device=device, dtype=torch.float32)
            return {"x_hat": x_hat}

        residuals = self._decode_residual_payload(header=header, payload=payload, device=device)
        # residuals are turned back into symbols using the same causal prediction rule.
        symbols = self._decode_symbols_from_residuals(
            residuals,
            prediction_mode=str(header.get("prediction_mode", "value")),
        )
        x_hat = self._symbols_to_float(symbols)
        return {"x_hat": x_hat}

    @property
    def proxy_bpppc(self) -> None:
        return None

    @property
    def bpppc(self) -> None:
        return None

    def _predict_from_target_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        teacher, _ = self._teacher_and_target_from_symbols(symbols)
        return self._predict_from_teacher_values(teacher)

    def _teacher_and_target_from_symbols(
        self, symbols: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prediction_mode == "value":
            target = self._symbols_to_float(symbols)
        else:
            target = self._deltas_to_float(self._symbols_to_deltas(symbols))

        # teacher forcing shifts the known sequence by one band for causal prediction
        teacher = torch.zeros_like(target)
        teacher[:, 1:] = target[:, :-1]
        return teacher, target

    def _predict_from_teacher_values(self, teacher_values: torch.Tensor) -> torch.Tensor:
        n, c, h, w = teacher_values.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")

        # each pixel becomes a separate sequence over spectral bands.
        seq = teacher_values.permute(0, 2, 3, 1).reshape(n * h * w, 1, c)
        hidden = self.input_proj(seq)
        for block in self.blocks:
            hidden = block(hidden)
        out = self.output_proj(self.head_act(hidden))
        out = self.output_head(out)
        return out.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _sample_pixel_sequences(
        self,
        teacher: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None,
        pixels_per_patch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        n, c, h, w = target.shape
        num_pixels = h * w
        sample_count = min(pixels_per_patch, num_pixels)
        if sample_count == num_pixels:
            return teacher, target, valid_mask

        if self.training:
            # random pixel sampling gives cheaper but varied training batches.
            indices = torch.randperm(num_pixels, device=target.device)[:sample_count]
        else:
            # deterministic validation sampling avoids noisy metric selection
            indices = (
                torch.linspace(
                    0,
                    num_pixels - 1,
                    steps=sample_count,
                    device=target.device,
                )
                .round()
                .long()
            )

        sample_h = math.isqrt(sample_count)
        while sample_h > 1 and sample_count % sample_h != 0:
            sample_h -= 1
        sample_w = sample_count // sample_h

        # reshape sampled pixels back to a small grid because losses expect image tensors.
        teacher_flat = teacher.reshape(n, c, num_pixels)
        target_flat = target.reshape(n, c, num_pixels)
        sampled_teacher = teacher_flat.index_select(dim=2, index=indices).reshape(
            n, c, sample_h, sample_w
        )
        sampled_target = target_flat.index_select(dim=2, index=indices).reshape(
            n, c, sample_h, sample_w
        )

        sampled_mask = None
        if valid_mask is not None:
            mask_flat = valid_mask.reshape(n, c, num_pixels)
            sampled_mask = mask_flat.index_select(dim=2, index=indices).reshape(
                n, c, sample_h, sample_w
            )

        return sampled_teacher.contiguous(), sampled_target.contiguous(), sampled_mask

    def _residuals_from_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        teacher, _ = self._teacher_and_target_from_symbols(symbols)
        predicted = self._predict_from_teacher_values(teacher)
        if self.prediction_mode == "value":
            # value mode predicts the next absolute spectral symbol.
            predicted_symbols = self._to_symbols(predicted)
            return (symbols - predicted_symbols).to(torch.int32)

        # delta mode predicts changes between adjacent bands before coding residuals.
        deltas = self._symbols_to_deltas(symbols)
        predicted_deltas = self._to_delta_symbols(predicted)
        return (deltas - predicted_deltas).to(torch.int32)

    def _decode_symbols_from_residuals(
        self,
        residuals: torch.Tensor,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        mode = self.prediction_mode if prediction_mode is None else prediction_mode
        if mode == "delta":
            return self._decode_symbols_from_delta_residuals(residuals)
        if mode != "value":
            raise ValueError(f"Unknown prediction_mode in payload: {mode!r}")

        n, c, h, w = residuals.shape
        num_pixels = n * h * w
        device = residuals.device

        residuals_flat = residuals.permute(0, 2, 3, 1).reshape(num_pixels, c)
        decoded_flat = torch.zeros_like(residuals_flat, dtype=torch.int32)
        # the first prediction uses zero because no previous spectral band exists.
        teacher_t = torch.zeros(num_pixels, device=device, dtype=torch.float32)

        states = [
            block.init_state(num_pixels, device=device, dtype=torch.float32)
            for block in self.blocks
        ]

        for band_idx in range(c):
            # decode one band at a time so future bands are never used as context.
            predicted_t, states = self._predict_step(teacher_t, states)
            predicted_symbols = self._to_symbols(predicted_t)
            decoded_t = (predicted_symbols + residuals_flat[:, band_idx]).clamp(
                0, self.symbol_scale
            )
            decoded_flat[:, band_idx] = decoded_t
            teacher_t = decoded_t.to(torch.float32) / self.symbol_scale

        return decoded_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _decode_symbols_from_delta_residuals(self, residuals: torch.Tensor) -> torch.Tensor:
        n, c, h, w = residuals.shape
        num_pixels = n * h * w
        device = residuals.device

        residuals_flat = residuals.permute(0, 2, 3, 1).reshape(num_pixels, c)
        decoded_flat = torch.zeros_like(residuals_flat, dtype=torch.int32)
        # delta mode predicts spectral differences, then accumulates them back to values.
        teacher_t = torch.zeros(num_pixels, device=device, dtype=torch.float32)
        prev_symbol = torch.zeros(num_pixels, device=device, dtype=torch.int32)

        states = [
            block.init_state(num_pixels, device=device, dtype=torch.float32)
            for block in self.blocks
        ]

        for band_idx in range(c):
            predicted_t, states = self._predict_step(teacher_t, states)
            predicted_delta = self._to_delta_symbols(predicted_t)
            decoded_delta = predicted_delta + residuals_flat[:, band_idx]
            decoded_symbol = (prev_symbol + decoded_delta).clamp(0, self.symbol_scale)
            decoded_flat[:, band_idx] = decoded_symbol
            teacher_t = decoded_delta.to(torch.float32) / self.symbol_scale
            prev_symbol = decoded_symbol

        return decoded_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _predict_symbols_sequential_from_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        n, c, h, w = symbols.shape
        num_pixels = n * h * w
        device = symbols.device

        symbols_flat = symbols.permute(0, 2, 3, 1).reshape(num_pixels, c)
        predicted_flat = torch.zeros_like(symbols_flat, dtype=torch.int32)
        # teacher_t contains the previous true band during residual creation.
        teacher_t = torch.zeros(num_pixels, device=device, dtype=torch.float32)

        states = [
            block.init_state(num_pixels, device=device, dtype=torch.float32)
            for block in self.blocks
        ]

        for band_idx in range(c):
            predicted_t, states = self._predict_step(teacher_t, states)
            predicted_flat[:, band_idx] = self._to_symbols(predicted_t)
            teacher_t = symbols_flat[:, band_idx].to(torch.float32) / self.symbol_scale

        return predicted_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _predict_deltas_sequential_from_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        n, c, h, w = symbols.shape
        num_pixels = n * h * w
        device = symbols.device

        deltas_flat = self._symbols_to_deltas(symbols).permute(0, 2, 3, 1).reshape(num_pixels, c)
        predicted_flat = torch.zeros_like(deltas_flat, dtype=torch.int32)
        # here the model predicts previous true deltas instead of absolute values.
        teacher_t = torch.zeros(num_pixels, device=device, dtype=torch.float32)

        states = [
            block.init_state(num_pixels, device=device, dtype=torch.float32)
            for block in self.blocks
        ]

        for band_idx in range(c):
            predicted_t, states = self._predict_step(teacher_t, states)
            predicted_flat[:, band_idx] = self._to_delta_symbols(predicted_t)
            teacher_t = deltas_flat[:, band_idx].to(torch.float32) / self.symbol_scale

        return predicted_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _predict_step(
        self,
        teacher_t: torch.Tensor,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # one scalar per pixel is projected to hidden channels for causal tcn blocks.
        hidden = self.input_proj(teacher_t.view(-1, 1, 1)).squeeze(-1)
        next_states: list[torch.Tensor] = []
        for block, state in zip(self.blocks, states, strict=True):
            hidden, next_state = block.step(hidden, state)
            next_states.append(next_state)

        out = self.output_proj(self.head_act(hidden.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        out = self.output_head(out)
        return out, next_states

    def _to_symbols(self, x: torch.Tensor) -> torch.Tensor:
        # normalized values in 0..1 become integer symbols in 0..symbol_scale.
        return torch.round(x.clamp(0.0, 1.0) * self.symbol_scale).to(torch.int32)

    def _to_delta_symbols(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x.clamp(-1.0, 1.0) * self.symbol_scale).to(torch.int32)

    def _symbols_to_float(self, symbols: torch.Tensor) -> torch.Tensor:
        return symbols.to(torch.float32) / float(self.symbol_scale)

    def _symbols_to_deltas(self, symbols: torch.Tensor) -> torch.Tensor:
        # first band is stored as a value, following bands are differences.
        deltas = torch.empty_like(symbols, dtype=torch.int32)
        deltas[:, 0] = symbols[:, 0]
        deltas[:, 1:] = symbols[:, 1:] - symbols[:, :-1]
        return deltas

    def _deltas_to_float(self, deltas: torch.Tensor) -> torch.Tensor:
        return deltas.to(torch.float32) / float(self.symbol_scale)

    def _is_exact_symbol_grid(self, x: torch.Tensor, symbols: torch.Tensor) -> bool:
        # residual coding is only truly lossless when input floats are exactly on the symbol grid.
        x_float = x.to(torch.float32)
        scaled = x_float * float(self.symbol_scale)
        finite = torch.isfinite(scaled).all()
        in_range = (x_float >= -1e-7).all() and (x_float <= 1.0 + 1e-7).all()
        close_to_symbols = torch.allclose(
            scaled,
            symbols.to(dtype=torch.float32),
            rtol=0.0,
            atol=1e-3,
        )
        return bool(finite and in_range and close_to_symbols)

    def _pack_residuals(
        self,
        residuals: torch.Tensor,
        residual_backend: str,
        residual_transform: str,
        codec_backend: str | None,
        compression_level: int | None,
    ) -> bytes:
        if residual_backend not in {"zlib", "zstd"}:
            raise ValueError("residual_backend must be one of: 'zlib', 'zstd'")
        transform = self._normalize_residual_transform(residual_transform)

        residual_min = int(residuals.min().item())
        residual_max = int(residuals.max().item())
        residuals_np = np.ascontiguousarray(residuals.cpu().numpy().astype(np.int32, copy=False))

        if transform == "zigzag+bitplane":
            if residual_min < -32768 or residual_max > 32767:
                raise ValueError(
                    "TCN residuals do not fit int16 zigzag bitplane coding "
                    f"(min={residual_min}, max={residual_max})"
                )
            payload = self._uint16_to_bitplane_bytes(self._zigzag_encode_int32(residuals_np))
            payload_dtype = "uint16"
        else:
            residual_dtype = (
                np.int16 if residual_min >= -32768 and residual_max <= 32767 else np.int32
            )
            payload = residuals_np.astype(residual_dtype).tobytes(order="C")
            payload_dtype = np.dtype(residual_dtype).name

        backend_name = codec_backend or self._default_codec_backend(residual_backend, transform)
        header: dict[str, object] = {
            "codec_backend": backend_name,
            "dtype": payload_dtype,
            "entropy_backend": residual_backend,
            "shape": list(residuals.shape),
            "symbol_scale": self.symbol_scale,
            "prediction_mode": self.prediction_mode,
        }
        if transform != "none":
            header["transform"] = transform
        if codec_backend is None and residual_backend == "zlib" and transform == "none":
            # Preserve the historical public payload shape for the default model codec.
            header["compression_mode"] = self.compression_mode

        return self._pack_payload(
            header=header,
            payload=payload,
            backend=residual_backend,
            compression_level=compression_level,
        )

    def _decode_residual_payload(
        self,
        header: dict[str, object],
        payload: bytes,
        device: torch.device,
    ) -> torch.Tensor:
        transform = str(header.get("transform", "none"))
        if transform == "zigzag+bitplane":
            residual_array = self._zigzag_decode_uint16(
                self._bitplane_bytes_to_uint16(payload, header["shape"])
            )
        elif transform == "none":
            residual_array = np.frombuffer(payload, dtype=np.dtype(header["dtype"])).copy()
            residual_array = residual_array.reshape(header["shape"])
        else:
            raise ValueError(f"Unknown residual transform in payload: {transform!r}")
        return torch.from_numpy(residual_array).to(device=device, dtype=torch.int32)

    def _pack_array(self, header: dict[str, object], array: np.ndarray) -> bytes:
        return self._pack_payload(
            header=header,
            payload=array.tobytes(order="C"),
            backend="zlib",
            compression_level=None,
        )

    def _pack_payload(
        self,
        header: dict[str, object],
        payload: bytes,
        backend: str,
        compression_level: int | None,
    ) -> bytes:
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        compressed = self._compress_payload(payload, backend, compression_level)
        return struct.pack("<I", len(header_bytes)) + header_bytes + compressed

    def _unpack_payload(self, strings) -> tuple[dict[str, object], bytes]:
        if not isinstance(strings, (bytes, bytearray)):
            raise TypeError(f"Expected raw bytes for strings, got {type(strings)!r}")
        # first four bytes store the header length as little-endian unsigned int.
        header_len = struct.unpack("<I", strings[:4])[0]
        header_start = 4
        header_end = header_start + header_len
        header = json.loads(strings[header_start:header_end].decode("utf-8"))
        backend = self._backend_from_header(header)
        payload = self._decompress_payload(strings[header_end:], backend)
        return header, payload

    def _compress_payload(
        self,
        payload: bytes,
        backend: str,
        compression_level: int | None,
    ) -> bytes:
        level = self.zlib_level if compression_level is None else int(compression_level)
        if backend == "zlib":
            return zlib.compress(payload, level=level)
        if backend == "zstd":
            if zstd is None:
                raise RuntimeError(
                    "Optional dependency 'zstandard' is required for residual_backend='zstd'."
                )
            return zstd.ZstdCompressor(level=level).compress(payload)
        raise ValueError(f"Unknown residual backend: {backend!r}")

    @staticmethod
    def _decompress_payload(payload: bytes, backend: str) -> bytes:
        if backend == "zlib":
            return zlib.decompress(payload)
        if backend == "zstd":
            if zstd is None:
                raise RuntimeError(
                    "Optional dependency 'zstandard' is required to decode zstd TCN payloads."
                )
            return zstd.ZstdDecompressor().decompress(payload)
        raise ValueError(f"Unknown residual backend: {backend!r}")

    @staticmethod
    def _backend_from_header(header: dict[str, object]) -> str:
        entropy_backend = header.get("entropy_backend")
        if entropy_backend is not None:
            return str(entropy_backend)
        codec_backend = str(header.get("codec_backend", ""))
        if codec_backend.endswith("_zstd"):
            return "zstd"
        return "zlib"

    @staticmethod
    def _default_codec_backend(residual_backend: str, transform: str) -> str:
        if transform == "zigzag+bitplane":
            return f"bitplane_tcn_residual_{residual_backend}"
        if residual_backend == "zlib":
            return "zlib_residual"
        return "tcn_residual_zstd"

    @staticmethod
    def _normalize_residual_transform(transform: str) -> str:
        if transform in {"none", ""}:
            return "none"
        if transform in {"bitplane", "zigzag_bitplane", "zigzag+bitplane"}:
            return "zigzag+bitplane"
        raise ValueError(
            "residual_transform must be one of: 'none', 'bitplane', 'zigzag_bitplane', "
            "'zigzag+bitplane'"
        )

    @staticmethod
    def _uint16_to_bitplane_bytes(values: np.ndarray) -> bytes:
        flat = np.ascontiguousarray(values.astype(np.uint16, copy=False)).reshape(-1)
        bytes_view = flat.view(np.uint8).reshape(-1, 2)
        bits = np.unpackbits(bytes_view, axis=1, bitorder="little")
        return np.packbits(bits.T.reshape(-1), bitorder="little").tobytes()

    @staticmethod
    def _bitplane_bytes_to_uint16(data: bytes, shape: object) -> np.ndarray:
        num_values = int(np.prod(np.asarray(shape, dtype=np.int64)))
        packed = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(packed, bitorder="little")[: num_values * 16]
        bits = bits.reshape(16, num_values).T
        bytes_view = np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little"))
        return bytes_view.reshape(num_values, 2).view(np.uint16).reshape(shape).copy()

    @staticmethod
    def _zigzag_encode_int32(values: np.ndarray) -> np.ndarray:
        values_i32 = values.astype(np.int32, copy=False)
        mapped = np.where(values_i32 >= 0, values_i32 * 2, (-values_i32 * 2) - 1)
        return mapped.astype(np.uint16)

    @staticmethod
    def _zigzag_decode_uint16(values: np.ndarray) -> np.ndarray:
        values_i32 = np.array(values, dtype=np.int32, copy=True)
        half = np.right_shift(values_i32, 1)
        sign = -np.bitwise_and(values_i32, 1)
        return np.bitwise_xor(half, sign).astype(np.int32)
