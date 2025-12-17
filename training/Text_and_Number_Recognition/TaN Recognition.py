import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import random
import shutil


#Directory of the dataset
DATA_DIR = Path(r"C:\Users\Kelvin Agbonde\Downloads\Masters Docs\Distributed AI\Project\Traffic Signs Data set\Carla Traffic Signs\traffic_signs")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4

#To use gpu if available, if not use cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Function to create a test set by moving a % of train images
def create_test_split_from_train(test_ratio=0.10, seed=42):
    train_dir = DATA_DIR / "train"
    test_dir  = DATA_DIR / "test"
    test_dir.mkdir(exist_ok=True) #This creates test folder since it does not exist

    random.seed(seed)

    def is_image_file(p: Path) -> bool: #This is a helper function to check if a file is an image
        return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} #This is the allowed image extensions to be used

    for class_dir in train_dir.iterdir(): #Skips anything that is not a directory
        if not class_dir.is_dir():
            continue

        (test_dir / class_dir.name).mkdir(parents=True, exist_ok=True) #Creates matching class foolder inside test

        images = [p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)]
        if len(images) == 0:
            continue

        k = max(1, int(len(images) * test_ratio))
        chosen = random.sample(images, k)

        for src in chosen:
            dst = test_dir / class_dir.name / src.name
            shutil.move(str(src), str(dst))

    print(f"Created test split at: {test_dir}") #Prints where all tests split is created

#To create dataloaders for training and validation
def get_dataloaders():
    """
    Loads images from disk and prepares them for training.

    - Applies random augmentations to training images
    - Resizes images to 128×128
    - Normalizes pixel values
    """

# Transformations for TRAINING data
    train_tfms = transforms.Compose([
        transforms.Resize((32, 32)),         # Resize images
        transforms.RandomRotation(10),         # Random rotate for variability
        transforms.ColorJitter(0.2, 0.2),      # Random brightness/contrast
        transforms.RandomHorizontalFlip(),     # Random flip
        transforms.ToTensor(),                 # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Transformations for VALIDATION data (NO randomness here)
    val_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Loads images labeled by folder name
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=val_tfms)

    # Wrap datasets in DataLoader for batching and GPU transfer
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)

    # Returns data + list of class labels (e.g., ["20", "30", "STOP"])
    return train_loader, val_loader, test_loader, train_ds.classes


# FUNCTION: Create a ResNet model for classification

def build_model(num_classes):
    """
    Loads a pre-trained ResNet18 model and replaces the last layer
    to match the number of traffic-sign classes.
    """

    # Load the training model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final classification layer with a new one
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    return model

#print("Script started running...")
#This is the main training loop

def train():
    """
    Runs the full training process:
    - loads dataset
    - trains the model
    - evaluates on validation data
    - saves graphs and best model
    """

    # Load data and class names
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    print("Detected classes:", class_names)

    # Build model and send it to GPU (if available)
    model = build_model(num_classes).to(DEVICE)

    # Loss function: This is how wrong the predictions are
    criterion = nn.CrossEntropyLoss()

    # Optimizer: This adjusts weights to reduce loss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #   This store the best accuracy to save the best model
    best_val_acc = 0.0

    # This is the lists to store graph data
    history_train_loss = []
    history_train_acc  = []
    history_val_acc    = []


    # This runs once for each epoch loop
    for epoch in range(NUM_EPOCHS):


        # Training mode section

        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        # Loop through all training batches
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Reset gradients
            optimizer.zero_grad()

            # Run forward pass
            outputs = model(imgs)

            # Compute loss (how wrong predictions are)
            loss = criterion(outputs, labels)

            # Backpropagation: compute gradients
            loss.backward()

            # Update network weights
            optimizer.step()

            # Accumulate accuracy and loss stats
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        # Calculate final training metrics for the epoch
        train_loss = running_loss / total
        train_acc = running_correct / total

    # Validation Mode
        model.eval()
        val_correct = 0
        val_total = 0

        # Disable gradient calculation during validation
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total

        # This is to save the metrics for the graphs...I think
        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        # Print epoch results to console
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.3f}  "
              f"Val Acc: {val_acc:.3f}")

        # This is to save the best model so far that is the one with the highest val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, "sign_text_classifier_best.pth")

            print("Saved new best-performing model")



    # PLOT TRAINING RESULTS (GRAPHS)
        # X-axis = epoch numbers
    epochs = range(1, len(history_train_loss) + 1)

    # ---- Graph 1: Training Loss ----
    plt.figure()
    plt.plot(epochs, history_train_loss, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss_curve.png")  # save graph to file

    # ---- Graph 2: Training + Validation Accuracy ----
    plt.figure()
    plt.plot(epochs, history_train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, history_val_acc, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curves.png")

    print("Saved graphs as:")
    print("   training_loss_curve.png")
    print("   accuracy_curves.png")

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += imgs.size(0)

    test_acc = test_correct / test_total
    print(f"Final TEST Accuracy: {test_acc:.3f}")

#This will give from image to label to speed value
inference_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_trained_model(model_path):
    """
    Loads the saved .pth traffic sign classifier and returns:
      - model (ResNet18 with correct head)
      - class_names (list of sign labels)
    What will call from CARLA startup code.
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    # Recreate the exact same model architecture
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def label_to_speed(label: str):
    """
    Converts a class label ('speed_limit_30')
     and then converts into an integer speed (30).
    Returns None for non-speed signs (e.g., 'stop', 'back').
    """
    nums = re.findall(r"\d+", label)
    if not nums:
        return None
    return int(nums[0])

#preprocessing as during training
def get_speed_from_image(img, model, class_names):
    """
    MAIN FUNCTION YOU CARE ABOUT.

    Input:
      - img: traffic sign image (NumPy array from CARLA OR PIL.Image)
      - model: loaded trained model
      - class_names: list of labels from training

    Output:
      - predicted_label: e.g., 'speed_limit_30'
      - speed_value:    e.g., 30  (or None if it's 'stop', 'back', etc.)

    This is what you'll call inside CARLA, after cropping a sign
    from the camera frame.
    """

    # If img is a NumPy array (e.g. from CARLA camera), convert to PIL
    if isinstance(img, np.ndarray):
        # Assume BGR from OpenCV/CARLA - convert to RGB
        img = Image.fromarray(img[:, :, ::-1]).convert("RGB")
    else:
        # Ensure it's RGB
        img = img.convert("RGB")

    # Preprocess the image for the network
    x = inference_tfms(img).unsqueeze(0).to(DEVICE)

    # Forward pass (no gradients needed)
    with torch.no_grad():
        outputs = model(x)                # shape: [1, num_classes]
        preds = outputs.argmax(dim=1)     # index of highest score
        class_idx = preds.item()

    predicted_label = class_names[class_idx]
    speed_value = label_to_speed(predicted_label)

    return predicted_label, speed_value

# The main entry point
# if __name__ == "__main__":
#     print("Script started, beginning training...")
#
# #This trains and loads the trained model for inference...uses the already ,pth file I already saved
#
#
#     model_path = "sign_text_classifier_best.pth"
#     model, class_names = load_trained_model(model_path)
#     print("Loaded trained model with classes:", class_names)
#
# #Using a test example from a single image in any of the folder
#     test_image_path = r"C:/Users/Kelvin Agbonde/Downloads/Traffic Signs Data set/Carla Traffic Signs/traffic_signs/val/speed_limit_30/speed_limit_30_5.JPG"
#     if os.path.exists(test_image_path):
#         test_img = Image.open(test_image_path)
#
#         predicted_label, speed_value = get_speed_from_image(test_img, model, class_names)
#
#         print("Predicted label:", predicted_label)
#         print("Predicted speed value:", speed_value)
#     else:
#         print("No test image found at:", test_image_path)
#         print("Skip test inference; just model loading is verified.")
#
#     print("Script finished.")



#Run Training
if __name__ == "__main__":
    print("Script started, beginning training...")

    # This should run ONCE to create the test folder from train
    create_test_split_from_train(test_ratio=0.10, seed=42) #Commenting this out so it doesn't run again

    train()
    print("Training finished.")
