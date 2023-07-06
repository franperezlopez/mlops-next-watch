"""
Training Parameters
"""

# Raw train/prod data percentage
RAW: float = 0.05  # 0.10

# Training data percentage
TRAIN: float = 0.60  # 0.80

SEED: int = 42


class ALS:
    """
    ALS Hyper-Parameters
    """

    RANK: list[int] = [8, 10, 12, 15, 18, 20]
    REG_INTERVAL: list[float] = [0.001, 0.2]
    COLD_START_STRATEGY: list[str] = ["drop"]
    MAX_ITER: list[int] = [5]
