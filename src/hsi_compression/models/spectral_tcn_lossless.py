from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        y = self.conv(F.pad(x, (self.left_padding, 0)))
        y = self.proj(self.dropout(self.act(y)))
        return x + y

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.left_padding == 0:
            return torch.empty(batch_size, self.channels, 0, device=device, dtype=dtype)
        return torch.zeros(batch_size, self.channels, self.left_padding, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_t_3d = x_t.unsqueeze(-1)
        if self.left_padding > 0:
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

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.symbol_scale = int(symbol_scale)
        self.zlib_level = int(zlib_level)
        self.raw_fallback = raw_fallback

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
        self.head_act = nn.GELU()
        self.output_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        if output_activation == "sigmoid":
            self.output_head = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_head = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del valid_mask
        symbols = self._to_symbols(x)
        x_target = self._symbols_to_float(symbols)
        teacher = torch.zeros_like(x_target)
        teacher[:, 1:] = x_target[:, :-1]
        x_hat = self._predict_from_teacher_values(teacher)
        return {
            "x_hat": x_hat,
            "x_target": x_target,
        }

    def update(self, force: bool = False) -> bool:
        del force
        return False

    def compress(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, bytes | tuple[int, ...]]:
        del valid_mask
        x_float = x.detach().float()
        symbols = self._to_symbols(x_float)

        if self.raw_fallback and not self._is_exact_symbol_grid(x_float, symbols):
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

        predicted = self._predict_from_target_symbols(symbols)
        predicted_symbols = self._to_symbols(predicted)
        residuals = (symbols - predicted_symbols).to(torch.int32)

        residual_min = int(residuals.min().item())
        residual_max = int(residuals.max().item())
        residual_dtype = np.int16 if -32768 <= residual_min and residual_max <= 32767 else np.int32
        residual_array = np.ascontiguousarray(residuals.cpu().numpy().astype(residual_dtype))

        strings = self._pack_array(
            header={
                "compression_mode": self.compression_mode,
                "codec_backend": "zlib_residual",
                "dtype": np.dtype(residual_dtype).name,
                "shape": list(symbols.shape),
                "symbol_scale": self.symbol_scale,
            },
            array=residual_array,
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
            array = np.frombuffer(payload, dtype=np.float32).copy().reshape(header["shape"])
            x_hat = torch.from_numpy(array).to(device=device, dtype=torch.float32)
            return {"x_hat": x_hat}

        residual_array = np.frombuffer(payload, dtype=np.dtype(header["dtype"])).copy()
        residuals = torch.from_numpy(residual_array.reshape(header["shape"])).to(
            device=device, dtype=torch.int32
        )
        symbols = self._decode_symbols_from_residuals(residuals)
        x_hat = self._symbols_to_float(symbols)
        return {"x_hat": x_hat}

    @property
    def proxy_bpppc(self) -> None:
        return None

    @property
    def bpppc(self) -> None:
        return None

    def _predict_from_target_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        teacher = torch.zeros_like(symbols, dtype=torch.float32)
        teacher[:, 1:] = symbols[:, :-1].to(torch.float32) / self.symbol_scale
        return self._predict_from_teacher_values(teacher)

    def _predict_from_teacher_values(self, teacher_values: torch.Tensor) -> torch.Tensor:
        n, c, h, w = teacher_values.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")

        seq = teacher_values.permute(0, 2, 3, 1).reshape(n * h * w, 1, c)
        hidden = self.input_proj(seq)
        for block in self.blocks:
            hidden = block(hidden)
        out = self.output_proj(self.head_act(hidden))
        out = self.output_head(out)
        return out.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _decode_symbols_from_residuals(self, residuals: torch.Tensor) -> torch.Tensor:
        n, c, h, w = residuals.shape
        num_pixels = n * h * w
        device = residuals.device

        residuals_flat = residuals.permute(0, 2, 3, 1).reshape(num_pixels, c)
        decoded_flat = torch.zeros_like(residuals_flat, dtype=torch.int32)
        teacher_t = torch.zeros(num_pixels, device=device, dtype=torch.float32)

        states = [
            block.init_state(num_pixels, device=device, dtype=torch.float32) for block in self.blocks
        ]

        for band_idx in range(c):
            predicted_t, states = self._predict_step(teacher_t, states)
            predicted_symbols = self._to_symbols(predicted_t)
            decoded_t = (predicted_symbols + residuals_flat[:, band_idx]).clamp(0, self.symbol_scale)
            decoded_flat[:, band_idx] = decoded_t
            teacher_t = decoded_t.to(torch.float32) / self.symbol_scale

        return decoded_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _predict_step(
        self,
        teacher_t: torch.Tensor,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self.input_proj(teacher_t.view(-1, 1, 1)).squeeze(-1)
        next_states: list[torch.Tensor] = []
        for block, state in zip(self.blocks, states, strict=True):
            hidden, next_state = block.step(hidden, state)
            next_states.append(next_state)

        out = self.output_proj(self.head_act(hidden.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        out = self.output_head(out)
        return out, next_states

    def _to_symbols(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x.clamp(0.0, 1.0) * self.symbol_scale).to(torch.int32)

    def _symbols_to_float(self, symbols: torch.Tensor) -> torch.Tensor:
        return symbols.to(torch.float32) / float(self.symbol_scale)

    def _is_exact_symbol_grid(self, x: torch.Tensor, symbols: torch.Tensor) -> bool:
        reconstructed = self._symbols_to_float(symbols).to(device=x.device)
        return torch.equal(reconstructed, x.to(torch.float32))

    def _pack_array(self, header: dict[str, object], array: np.ndarray) -> bytes:
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        payload = zlib.compress(array.tobytes(order="C"), level=self.zlib_level)
        return struct.pack("<I", len(header_bytes)) + header_bytes + payload

    def _unpack_payload(self, strings) -> tuple[dict[str, object], bytes]:
        if not isinstance(strings, (bytes, bytearray)):
            raise TypeError(f"Expected raw bytes for strings, got {type(strings)!r}")
        header_len = struct.unpack("<I", strings[:4])[0]
        header_start = 4
        header_end = header_start + header_len
        header = json.loads(strings[header_start:header_end].decode("utf-8"))
        payload = zlib.decompress(strings[header_end:])
        return header, payload
