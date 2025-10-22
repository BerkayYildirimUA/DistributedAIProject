import os
from collections import Counter


def count_yolo_classes(directory):
    class_counts = {}
    for i in range(1,6):
        class_counts[i]=Counter()

    # Loop over all txt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            town=int(filename.split("_")[0].replace("Town0",""))
            with open(filepath, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 0:  # valid line
                        class_id = int(parts[0])  # first number is class id
                        # class_counts[class_id] += 1
                        class_counts[town][class_id] += 1

    return class_counts


if __name__ == "__main__":
    directory = "archive"  # change to your directory path
    counts = count_yolo_classes(directory)

    print("Class instance counts:")
    for town, count_map in sorted(counts.items()):
        print(f"{town}:")
        for class_id, count in sorted(count_map.items()):
            print(f"\t{class_id}: {count}")
