"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Tuân thủ nghiêm ngặt quy tắc chia dữ liệu bắt buộc (S1-S6) ở README.md mục 2.1.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Mỗi file có cột Filename, Label, Species. Trả về (train_df, val_df, test_df).
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        raise FileNotFoundError(f"Không tìm thấy đủ file split fold {fold} trong {labels_dir}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> Dict[str, Any]:
    """Kiểm tra bắt buộc trước khi train (README.md mục 2.1).
    1. Số ảnh mỗi tập và mỗi lớp xấp xỉ 60/20/20.
    2. Giao từng cặp tập theo Filename là RỖNG.
    3. Hợp ba tập bằng đúng 17.509 ảnh.
    4. Mọi Filename tồn tại trong images_dir.
    """
    images_path = Path(images_dir)
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    n_train, n_val, n_test = len(train_files), len(val_files), len(test_files)
    n_total = n_train + n_val + n_test

    # 1. Giao giữa các tập phải rỗng
    tv_overlap = train_files & val_files
    tt_overlap = train_files & test_files
    vt_overlap = val_files & test_files
    assert len(tv_overlap) == 0, f"Lỗi S4: Train và Val trùng {len(tv_overlap)} ảnh!"
    assert len(tt_overlap) == 0, f"Lỗi S4: Train và Test trùng {len(tt_overlap)} ảnh!"
    assert len(vt_overlap) == 0, f"Lỗi S4: Val và Test trùng {len(vt_overlap)} ảnh!"

    # 2. Hợp ba tập đúng 17.509 ảnh
    union_files = train_files | val_files | test_files
    assert len(union_files) == 17509, f"Lỗi: Tổng số ảnh {len(union_files)} != 17,509!"

    # 3. Mọi file tồn tại trên đĩa nếu images_dir có sẵn
    if images_path.exists():
        existing_images = set(os.listdir(images_path))
        missing_files = union_files - existing_images
        assert len(missing_files) == 0, f"Lỗi: Thiếu {len(missing_files)} file ảnh trong {images_dir}!"

    # 4. Thống kê theo lớp
    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": n_total},
        "ratio": {
            "train": n_train / n_total,
            "val": n_val / n_total,
            "test": n_test / n_total,
        },
        "per_class": {
            "train": train_df.groupby("Label")["Filename"].count().to_dict(),
            "val": val_df.groupby("Label")["Filename"].count().to_dict(),
            "test": test_df.groupby("Label")["Filename"].count().to_dict(),
        },
        "overlap": {"train_val": len(tv_overlap), "train_test": len(tt_overlap), "val_test": len(vt_overlap)},
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> transforms.Compose:
    """Tạo torchvision transform cho train hoặc eval."""
    if train:
        tf_list = []
        # Random resized crop
        tf_list.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        # Horizontal flip (lật ngang hoàn toàn hợp lệ với thực vật tự nhiên)
        tf_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if aug == "color":
            tf_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        elif aug == "randaug":
            tf_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif aug == "trivial":
            tf_list.append(transforms.TrivialAugmentWide())
        elif aug == "basic":
            pass
        else:
            raise ValueError(f"Không hỗ trợ augmentation: {aug}")

        tf_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transforms.Compose(tf_list)
    else:
        # Eval: Deterministic resize & center crop
        if img_size == 256:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh DeepWeeds theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].values
        self.labels = self.df["Label"].values.astype(int)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        fn = self.filenames[idx]
        lbl = int(self.labels[idx])
        img_path = self.images_dir / fn
        
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)

        return image, lbl, fn


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: Optional[str] = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader hỗ trợ cân bằng mẫu (sampler='balanced')."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train and sampler == "balanced":
        # WeightedRandomSampler: trọng số nghịch đảo số lượng mẫu mỗi lớp
        class_counts = df["Label"].value_counts().to_dict()
        weights = [1.0 / class_counts[lbl] for lbl in df["Label"]]
        weights_tensor = torch.DoubleTensor(weights)
        sampler_obj = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if len(dataset) % batch_size == 1 else False,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if (train and len(dataset) % batch_size == 1) else False,
        )
    return loader
