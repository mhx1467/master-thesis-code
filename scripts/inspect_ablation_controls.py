import argparse
from collections.abc import Mapping
from pathlib import Path

from hsi_compression.utils import load_config

IGNORED_DIFF_PATHS = {
    ("experiment", "name"),
    ("model", "model_kwargs", "num_summary_tokens"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect model/control differences for an ablation pair."
    )
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--ablation-config", type=str, required=True)
    parser.add_argument("--input-height", type=int, default=128)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument(
        "--check-forward",
        action="store_true",
        help="Run a no-grad CPU forward pass to report actual latent shapes.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only compare config controls. This does not import model dependencies.",
    )
    return parser.parse_args()


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_diff(left, right, path=()):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        for key in keys:
            yield from flatten_diff(left.get(key), right.get(key), path + (str(key),))
        return

    if left != right:
        yield path, left, right


def build_from_config(cfg: dict):
    from hsi_compression.models.registry import build_model

    model_cfg = cfg["model"]
    kwargs = dict(model_cfg.get("model_kwargs", {}))
    in_channels = int(kwargs.pop("in_channels"))
    return build_model(model_name=model_cfg["model_name"], in_channels=in_channels, **kwargs)


def model_summary(label: str, cfg_path: Path, cfg: dict, check_forward: bool, h: int, w: int):
    import torch

    model = build_from_config(cfg)
    model.eval()
    model_kwargs = cfg["model"].get("model_kwargs", {})
    in_channels = int(model_kwargs["in_channels"])
    n_params = count_trainable_parameters(model)
    proxy_bpppc = getattr(model, "proxy_bpppc", None)

    print(f"\n[{label}] {cfg_path}")
    print(f"experiment.name: {cfg.get('experiment', {}).get('name')}")
    print(f"model.name: {cfg['model']['model_name']}")
    print(f"num_summary_tokens: {model_kwargs.get('num_summary_tokens')}")
    print(f"latent_channels: {model_kwargs.get('latent_channels')}")
    print(f"trainable_parameters: {n_params:,}")
    if proxy_bpppc is not None:
        print(f"proxy_bpppc: {float(proxy_bpppc):.6f}")

    if check_forward:
        with torch.no_grad():
            x = torch.zeros(1, in_channels, h, w)
            out = model(x)
        print(f"forward.z_shape: {tuple(out['z'].shape)}")
        print(f"forward.x_hat_shape: {tuple(out['x_hat'].shape)}")

    return {
        "parameters": n_params,
        "proxy_bpppc": float(proxy_bpppc) if proxy_bpppc is not None else None,
    }


def main():
    args = parse_args()
    base_path = Path(args.base_config)
    ablation_path = Path(args.ablation_config)
    base_cfg = load_config(base_path)
    ablation_cfg = load_config(ablation_path)

    if not args.config_only:
        base = model_summary(
            "base",
            base_path,
            base_cfg,
            args.check_forward,
            args.input_height,
            args.input_width,
        )
        ablation = model_summary(
            "ablation",
            ablation_path,
            ablation_cfg,
            args.check_forward,
            args.input_height,
            args.input_width,
        )

        print("\n[comparison]")
        print(f"parameter_delta: {ablation['parameters'] - base['parameters']:,}")
        if base["proxy_bpppc"] is not None and ablation["proxy_bpppc"] is not None:
            print(f"proxy_bpppc_delta: {ablation['proxy_bpppc'] - base['proxy_bpppc']:.6f}")

    unexpected_diffs = [
        (path, left, right)
        for path, left, right in flatten_diff(base_cfg, ablation_cfg)
        if path not in IGNORED_DIFF_PATHS
    ]
    if unexpected_diffs:
        print("\n[unexpected config differences]")
        for path, left, right in unexpected_diffs:
            dotted = ".".join(path)
            print(f"{dotted}: base={left!r} | ablation={right!r}")
    else:
        print("\nConfig control check: OK")
        print("Only experiment.name and model.model_kwargs.num_summary_tokens differ.")


if __name__ == "__main__":
    main()
