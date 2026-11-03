"""PyTorch dataset over data_build/dataset.csv (packet-level splits come from the CSV)."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1] / "data_build"
NUM_CLASSES = 6          # strand count 0..5
GOOD_CLASS = 4
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_table(variant="enh"):
    df = pd.read_csv(ROOT / "dataset.csv")
    df["path"] = df[f"path_{variant}"].map(lambda p: str(ROOT / p))
    df["label"] = df["strand_count"].astype(int)
    return df


def get_split(df, val_fold, use_test=False):
    """Packet-level: test = split=='test'; val = one cv_fold; train = remaining cv folds."""
    cv = df[df.split == "cv"]
    if use_test:  # final eval: train on all CV, evaluate on frozen test
        return cv, None, df[df.split == "test"]
    return cv[cv.cv_fold != val_fold], cv[cv.cv_fold == val_fold], None


def make_transform(train, size=(160, 288)):
    """size=(H,W); crops are ~1.8:1 landscape. Light augmentation only."""
    aug = []
    if train:
        aug = [T.RandomHorizontalFlip(),
               T.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05)),
               T.ColorJitter(brightness=0.15, contrast=0.15)]
    return T.Compose([T.Resize(size), *aug, T.ToTensor(), T.Normalize(MEAN, STD)])


class StrandDataset(Dataset):
    def __init__(self, df, train=False, size=(160, 288)):
        self.df = df.reset_index(drop=True)
        self.tf = make_transform(train, size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), int(r["label"])


def class_weights(labels, n=NUM_CLASSES):
    """Inverse-frequency, normalised; absent classes get weight 0."""
    c = torch.bincount(torch.tensor(labels), minlength=n).float()
    return torch.where(c > 0, c.sum() / (n * c.clamp(min=1)), torch.zeros_like(c))
