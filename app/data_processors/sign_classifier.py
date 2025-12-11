import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import re

import os

class SignClassifier:
    def __init__(self, use_tracking = True):
        # Initialize model
        print("CUDA:", torch.cuda.is_available())

        # TODO: init in your model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet18(weights=None)
        in_feats = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_feats, 8)

        # TODO load checkpoint into model
        # Directory where THIS script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "sign_text_classifier_best.pth")

        checkpoint = torch.load(model_path, map_location=self.device)
        # checkpoint = torch.load("./data_processors/model/sign_text_classifier_best.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # TODO place model in eval mode
        self.model.to(self.device)
        self.model.eval()

        # TODO: define classess
        self.classes = checkpoint["class_names"]

        # TODO define transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # def cropped_traffic_signs(self, frame, boxes, class_ids):
    #     # I should assume that the boxes is a list of yolo coordinates.
    #     # Now I should extract the boxes of the frames and return a list of those new images.
    #     crops_img = []
    #
    #     #putting the  frame i a numpy array
    #     for box, class_id in zip(boxes, class_ids):
    #         if class_id == 4:
    #
    #             x1, y1, x2, y2 = box
    #
    #     #cropping from the numpy array
    #             crop = frame[y1:y2, x1:x2]
    #
    #             crops_img.append(crop)
    #
    #     return crops_img

    def cropped_traffic_signs(self, frame, boxes, class_ids):
        crops_img = []

        h, w = frame.shape[:2]

        for box, class_id in zip(boxes, class_ids):
            if class_id == 4:
                print("SIGN FOUND")
                # Ensure all coordinates are ints
                x1, y1, x2, y2 = [int(v) for v in box]

                # Clip to valid bounds
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                # Valid crop check
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crops_img.append(crop)

        return crops_img

    def label_to_speed(self, label):
        nums = re.findall(r"\d+", label)
        if not nums:
            return None
        return int(nums[0])

    # TODO: this function should take image from carla as input and return a tensor
    # you will need to apply your defined transformations here as well
    def preprocess_frame(self,frame) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            frame = frame[:, :, ::-1]
            frame = Image.fromarray(frame)
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)
        tensor = self.transform(frame).unsqueeze(0).to(self.device)
        return tensor
    @torch.no_grad()
    def read_sign(self, x):

        # TODO: call the preprocess fucntion here
        #x = self.preprocess_frame(frame)

        # TODO: Pass the return tensor through the model
        #with torch.no_grad():
        outputs = self.model(x)
        pred_index = outputs.argmax(dim=1).item()

        # TODO: extract label based on index from classes
        label = self.classes[pred_index]
        speed_value = self.label_to_speed(label)

        return label, speed_value

    def signal_classifier(self, frame, boxes, class_ids):
        frame = self.preprocess_frame(frame)
        images = self.cropped_traffic_signs(frame, boxes, class_ids)
        print(len(images))
        labels=[]
        for image in images:
            label, speed_value = self.read_sign(image)
            print(label, speed_value)
            labels.append(label)

        if len(labels) == 0:
            return -1
        return labels[0]





