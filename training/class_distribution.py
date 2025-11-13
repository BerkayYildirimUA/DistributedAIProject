# from collections import Counter
# from pathlib import Path
#
# def count_yolo_classes(root_dir):
#     root = Path(root_dir)
#     class_counts: dict[int, Counter] = {}
#
#     # find every .txt under train/, valid/, test/
#     for lbl in root.rglob("*.txt"):
#         # file names like Town02_000660.txt -> "Town02" -> 2
#         town_tag = lbl.stem.split("_")[0]            # 'Town02'
#         try:
#             town = int(town_tag.replace("Town0", ""))  # 2
#         except ValueError:
#             continue  # skip any unexpected files
#
#         class_counts.setdefault(town, Counter())
#
#         with lbl.open("r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 cls_id = int(parts[0])  # YOLO format: class cx cy w h
#                 class_counts[town][cls_id] += 1
#
#     return class_counts
#
#
# if __name__ == "__main__":
#     directory = r"C:\Users\micha\Documents\semester9\Master\1-Distributed AI\project\code\DistributedAIProject\training\dataset\labels"
#     counts = count_yolo_classes(directory)
#
#     print("Class instance counts:")
#     for town, cmap in sorted(counts.items()):
#         print(f"{town}:")
#         for cls_id, cnt in sorted(cmap.items()):
#             print(f"\t{cls_id}: {cnt}")
#



from collections import Counter
from pathlib import Path

# Edit if your class list differs
CLASS_NAMES = ["vehicle", "motorbike", "bike", "traffic light", "traffic sign", "pedestrian"]

def count_all_classes(labels_root: str | Path, class_names=CLASS_NAMES) -> Counter:
    root = Path(labels_root)
    if not root.exists():
        raise FileNotFoundError(f"Labels folder not found: {root}")

    counts = Counter()
    label_files = list(root.rglob("*.txt"))  # recurse into train/valid/test

    for lbl in label_files:
        with lbl.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # YOLO label format: class cx cy w h
                # (cast via float->int to be robust to "0.0" etc.)
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue
                counts[cls_id] += 1

    # Print results
    total = sum(counts.values())
    print(f"Scanned {len(label_files)} files • total objects: {total}\n")

    max_id = max(counts.keys(), default=-1)
    for cid in range(max_id + 1):
        name = class_names[cid] if class_names and cid < len(class_names) else f"class_{cid}"
        n = counts.get(cid, 0)
        print(f"{cid:>2} ({name}): {n}")

    return counts


if __name__ == "__main__":
    LABEL_DIR = r"C:\Users\micha\Documents\semester9\Master\1-Distributed AI\project\code\DistributedAIProject\training\dataset\labels"
    count_all_classes(LABEL_DIR)

