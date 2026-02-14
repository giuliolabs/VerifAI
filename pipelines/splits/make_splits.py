"""
Create train / val / test splits for FaceForensics++ C23
Splits are done at VIDEO level
"""

from pathlib import Path
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

FAKE_FOLDERS = {
    "deepfakes",
    "face2face",
    "faceswap",
    "faceshifter",
    "neuraltextures",
}

def infer_label(video_path: Path) -> int:
    parts = [p.lower() for p in video_path.parts]

    if "original" in parts:
        return 0  # real

    for name in FAKE_FOLDERS:
        if name in parts:
            return 1  # fake

    return -1  # unknown


def collect_videos(root: Path):
    records = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label = infer_label(p)
            records.append({
                "video_path": str(p).replace("\\", "/"),
                "label": label,
                "video_id": p.stem
            })
    return records


def main():
    ffpp_root = Path("data/raw/FaceForensics++_C23")
    if not ffpp_root.exists():
        raise FileNotFoundError(ffpp_root)

    records = collect_videos(ffpp_root)
    df = pd.DataFrame(records)

    # Remove unknown labels
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    print("Total FF++ videos:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts())

    # Shuffle (reproducible)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Split ratios
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]

    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "faceforensics++_train.csv", index=False)
    val_df.to_csv(out_dir / "faceforensics++_val.csv", index=False)
    test_df.to_csv(out_dir / "faceforensics++_test.csv", index=False)

    print("Splits written:")
    print(" train:", len(train_df))
    print(" val:", len(val_df))
    print(" test:", len(test_df))


if __name__ == "__main__":
    main()
