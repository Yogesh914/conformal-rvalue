import os
from pathlib import Path

import torch


def _optional_path_from_env(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _path_from_env(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


class Config:
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = _path_from_env("CONFORMAL_RVALUE_DATA_DIR", ROOT_DIR / "data")
    IMAGENET_DATASET_DIR = _path_from_env("IMAGENET_DATASET_DIR", DATA_DIR / "imagenet")
    HF_DATASETS_CACHE = _optional_path_from_env("HF_DATASETS_CACHE")
    ADAPTER_SAVE_DIR = _path_from_env("VIT_ADAPTER_SAVE_DIR", DATA_DIR / "adaptors" / "vit_l" / "epoch_3")
    EVAL_OUTPUT_DIR = _path_from_env(
        "VIT_EVAL_OUTPUT_DIR",
        DATA_DIR / "adaptor_info" / "vit_l" / "epoch_3",
    )
    INFERENCE_OUTPUT_PATH = _path_from_env("VIT_INFERENCE_OUTPUT_PATH", ROOT_DIR / "shark_predictions.pkl")

    # Model settings
    NUM_MODELS = 1000
    BATCH_SIZE = 512
    NUM_CLASSES = 1000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training settings
    EPOCHS = 2
    LEARNING_RATE = 0.001

    # Random seed
    SEED = 769

    @classmethod
    def apply_hf_datasets_cache(cls):
        if cls.HF_DATASETS_CACHE is None:
            return

        from datasets import config as datasets_config

        datasets_config.HF_DATASETS_CACHE = str(cls.HF_DATASETS_CACHE)

    @classmethod
    def require_imagenet_dataset_dir(cls):
        if cls.IMAGENET_DATASET_DIR.exists():
            return cls.IMAGENET_DATASET_DIR

        raise FileNotFoundError(
            "ImageNet dataset not found at "
            f"{cls.IMAGENET_DATASET_DIR}. Set IMAGENET_DATASET_DIR to a local datasets.load_from_disk path."
        )
