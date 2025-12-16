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

        # Init model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet18(weights=None)
        in_feats = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_feats, 8)

        # Load checkpoint into model
        # Directory where THIS script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "sign_text_classifier_best.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Place model in eval mode
        self.model.to(self.device)
        self.model.eval()

        # TODO: define classess

        # classes: ['back', 'speed_30', 'speed_60', 'speed_90', 'speed_limit_30', 'speed_limit_40',
        #           'speed_limit_60', 'stop']
        # self.classes= ['back',30,60,90,30,40,60,'stop']
        self.classes = [-1, 30, 60, 90, 30, 40, 60, -1]
        # self.classes = checkpoint["class_names"]

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _yolo_to_xyxy(self,box):
        cx, cy, bw, bh = box

        # compute half sizes
        half_w = bw / 2.0
        half_h = bh / 2.0

        # compute xyxy (float)
        x1_f = cx - half_w
        y1_f = cy - half_h
        x2_f = cx + half_w
        y2_f = cy + half_h

        # round and convert to int (use round to reduce off-by-one)
        x1 = int(x1_f)
        y1 = int(y1_f)
        x2 = int(x2_f)
        y2 = int(y2_f)

        return x1, y1, x2, y2

    def cropped_traffic_signs(self, frame, boxes, class_ids):
        crops_img = []

        # h, w = frame.shape[:2]
        frame = np.array(frame)

        for box, class_id in zip(boxes, class_ids):
            if class_id == 4:
                # Ensure all coordinates are ints
                x1, y1, x2, y2 = self._yolo_to_xyxy(box)

                # Valid crop check
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crop_pil = Image.fromarray(crop)
                    crops_img.append(crop_pil)
                else:
                    print("INVALID CROP")

        return crops_img

    @torch.no_grad()
    def read_sign(self, image_pil):
        x = self.transform(image_pil).unsqueeze(0).to(self.device)

        outputs = self.model(x)
        pred_index = outputs.argmax(dim=1).item()

        value = self.classes[pred_index]

        return value, outputs[0,pred_index]

    def signal_classifier(self, frame, boxes, class_ids):
        frame=Image.fromarray(frame)
        images = self.cropped_traffic_signs(frame, boxes, class_ids)

        most_conf=-1
        speed=-1
        for image in images:
            label, conf = self.read_sign(image)
            if conf > most_conf:
                most_conf = conf
                speed = label

        return speed





