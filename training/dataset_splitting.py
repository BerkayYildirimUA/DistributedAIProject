import os
import shutil

def create_folders():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/labels", exist_ok=True)

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)

    os.makedirs("dataset/images/valid", exist_ok=True)
    os.makedirs("dataset/labels/valid", exist_ok=True)

    os.makedirs("dataset/images/test", exist_ok=True)
    os.makedirs("dataset/labels/test", exist_ok=True)



def move_to_folders(directory):
    valid_class="Town02"
    test_class="Town03"

    # Loop over all txt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            image_filename=filename.replace(".txt",".png")
            filepath = os.path.join(directory, filename)
            image_filepath=filepath.replace(".txt",".png")
            town_name = filename.split("_")[0]

            if town_name == test_class:
                shutil.copy2(image_filepath, "dataset/images/test/" + image_filename)
                shutil.copy2(filepath, "dataset/labels/test/" + filename)
            elif town_name == valid_class:
                shutil.copy2(image_filepath, "dataset/images/valid/" + image_filename)
                shutil.copy2(filepath, "dataset/labels/valid/" + filename)
            else:
                shutil.copy2(image_filepath, "dataset/images/train/" + image_filename)
                shutil.copy2(filepath, "dataset/labels/train/" + filename)


if __name__ == "__main__":
    directory = "archive"  # change to your directory path
    create_folders()
    move_to_folders(directory)
