import shutil
from pathlib import Path
from PIL import Image

DATA_DIR = Path(r"C:\Users\Kelvin Agbonde\Documents\Carla Traffic Signs dataset\Carla Traffic Signs\traffic_signs")
MIN_SIDE = 32

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    moved = 0

    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue

        for class_dir in [d for d in split_dir.iterdir() if d.is_dir()]:
            small_dir = DATA_DIR / "removed_small" / split / class_dir.name
            small_dir.mkdir(parents=True, exist_ok=True)

            for p in class_dir.iterdir():
                if not p.is_file() or not is_image_file(p):
                    continue

                w, h = Image.open(p).size
                if min(w, h) < MIN_SIDE:
                    shutil.move(str(p), str(small_dir / p.name))
                    moved += 1

    print(f"Done. Moved {moved} tiny images to removed_small/")

if __name__ == "__main__":
    main()
