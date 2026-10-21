"""Time split of every file into train / val / reserve regions (design D11)."""

_TOL = 1e-9


# Region bounds in seconds: train [0, (1-v-r)D), val [(1-v-r)D, (1-r)D), reserve [(1-r)D, D].
def region_bounds(duration_s: float, val_ratio: float, reserve_ratio: float) -> dict[str, tuple[float, float]]:
    train_end = (1.0 - val_ratio - reserve_ratio) * duration_s
    val_end = (1.0 - reserve_ratio) * duration_s
    return {"train": (0.0, train_end), "val": (train_end, val_end), "reserve": (val_end, duration_s)}


# Name of the region that contains the whole span (start, end) in seconds, or None when it straddles a boundary.
def assign_region(span: tuple[float, float], bounds: dict[str, tuple[float, float]]) -> str | None:
    start, end = span
    for name, (lo, hi) in bounds.items():
        if lo - _TOL <= start and end <= hi + _TOL:
            return name
    return None
