WATER_VAPOR_BANDS: list[int] = list(range(126, 141)) + list(range(160, 167))

FULL_BAND_COUNT: int = 224
CLEAN_BAND_COUNT: int = 202

NODATA_VALUE: int = -32768

GLOBAL_MIN: float = 0.0
GLOBAL_MAX: float = 10000.0

PATCH_SIZE: int = 128

DEFAULT_DIFFICULTY: str = "easy"
DEFAULT_NUM_WORKERS: int = 8

DEFAULT_LR: float = 1e-4
HEAVY_MODEL_LR: float = 1e-5
GRAD_CLIP_MAX_NORM: float = 1.0
