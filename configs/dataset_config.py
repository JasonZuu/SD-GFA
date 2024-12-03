from dataclasses import dataclass


@dataclass
class DatasetConfig:
    train_metadata_path: str = "data/CelebA-HQ/celebahq_256_train.csv"
    val_metadata_path: str = "data/CelebA-HQ/celebahq_256_valid.csv"
    test_metadata_path: str = "data/CelebA-HQ/celebahq_256_test.csv"
    img_size: int = 256
    norm_type: str = "imagenet"

