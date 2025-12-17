from pathlib import Path
from PIL import Image

# Dataset Directory
DATA_DIR = Path(r"C:\Users\Kelvin Agbonde\Documents\Carla Traffic Signs dataset\Carla Traffic Signs\traffic_signs")

MIN_SIDE = 40  # images with width or height smaller than this will be reported

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    tiny = []
    total = 0

    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"Skipping missing folder: {split_dir}")
            continue

        for p in split_dir.rglob("*"):
            if not p.is_file() or not is_image_file(p):
                continue

            total += 1
            try:
                w, h = Image.open(p).size
                if min(w, h) < MIN_SIDE:
                    tiny.append((split, w, h, str(p)))
            except Exception as e:
                tiny.append((split, -1, -1, f"{p}  (ERROR: {e})"))

    print("Total images scanned:", total)
    print(f"Images with min side < {MIN_SIDE}:", len(tiny))
    print("\nExamples (up to 50):")
    for item in tiny[:50]:
        split, w, h, path = item
        print(f"{split:5}  {w:4}x{h:4}  {path}")

if __name__ == "__main__":
    main()
