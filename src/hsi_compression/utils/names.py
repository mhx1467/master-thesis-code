from __future__ import annotations

SAFE_FILENAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def safe_path_component(value: object, *, fallback: str = "item") -> str:
    component = "".join(char if char in SAFE_FILENAME_CHARS else "_" for char in str(value)).strip(
        "._-"
    )
    return component or fallback


def safe_sample_stem(sample_id: object) -> str:
    value = str(sample_id)
    return f"{int(value):04d}" if value.isdigit() else safe_path_component(value, fallback="sample")
