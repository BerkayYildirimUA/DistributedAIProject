import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from .model.model import parsingNet
import torch.nn.functional as F

class LaneDetector:
    def __init__(self, model_path='./data_processors/model/tusimple_18.pth'):
        self.cls_num_per_lane = 56

        # CREATE GRID
        # Fixed row anchors: on what locations is there a row
        self.row_anchors = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]

        # Number of grid cells in a row
        self.griding_num = 100


        # Input size
        self.model_input_height = 288
        self.model_input_width = 800

        # Size of incoming carla frame
        self.frame_w = 640
        self.frame_h = 480

        self.model_path=model_path

        # Create grid cells within row
        self.calculate_grid()


        # --- Load model ---
        self.load_model()

        # --- Define transform ---
        self.transform = transforms.Compose([
            transforms.Resize((self.model_input_height, self.model_input_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    # Create all grid samples for a row
    def calculate_grid(self):
        # Precompute horizontal sample points
        self.col_sample = np.linspace(0, self.model_input_width - 1, self.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]

    def load_model(self):
        self.net = parsingNet(
            pretrained=False,
            backbone='18',
            cls_dim=(self.griding_num + 1, self.cls_num_per_lane, 4),
            use_aux=False
        ).cuda()

        state_dict = torch.load(self.model_path, map_location='cuda')
        # Fix potential "module." prefix mismatch
        state_dict = state_dict.get('model', state_dict)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.net.load_state_dict(new_state_dict, strict=False)
        self.net.eval()

    def preprocess(self, img):
        img_pil = Image.fromarray(img)
        return self.transform(img_pil).unsqueeze(0).cuda()

    @torch.no_grad()
    def get_lane_scores(self, img_tensor):
        img_tensor=self.preprocess(img_tensor)

        # Shape output (batch_size, num_grid_cells+1,num_row_anchors,num_lanes)
        # Batch_size=1, we only pass one sample -> take it
        output = self.net(img_tensor)[0].data
        # Model predicts from top -> bottom, we change order
        output = torch.flip(output, dims=[1])

        # Calculate the probability of a lane in that cell with softmax
        # Drop last element in num_grid_cells dimension: it used to indicate if there is no lane detected, we don't need a prob
        prob = F.softmax(output[:-1, :, :], dim=0)
        # Soft localization: calculate continuous value for lane position
        # We take weighted average of all bins, outcome is a continuous value indicating the lane position
        # Shape (number_rows, lanes)
        bins = torch.arange(self.griding_num, device=prob.device).float() + 1

        # outputs an x coordinate for each row for each lane
        # soft_loc = np.sum(prob * bins.reshape(-1, 1, 1), axis=0)
        soft_loc = torch.sum(prob * bins[:, None, None], dim=0)

        # Hard classification, find max bin
        # Outputs bin number for each row for each lane
        output = torch.argmax(output, axis=0)
        # In case the max bin is the one reserved for no lane detected, we set it to zero -> ignore x-coor
        soft_loc[output == self.griding_num] = 0
        output = soft_loc

        return output

    def get_lanes(self, frame):
        scores= self.get_lane_scores(frame)
        return self.get_lane_coords(scores)

    def get_lane_coords(self,scores):
        lanes = []
        for i in range(scores.shape[1]):
            if np.sum(scores[:, i] != 0) > 2:
                for k in range(scores.shape[0]):
                    if scores[k, i] > 0:
                        ppp = (int(scores[k, i] * self.col_sample_w * self.frame_w / self.model_input_width) - 1,
                               int(self.frame_h * (self.row_anchors[self.cls_num_per_lane - 1 - k] / self.model_input_height)) - 1)
                        lanes.append(ppp)
        return lanes
