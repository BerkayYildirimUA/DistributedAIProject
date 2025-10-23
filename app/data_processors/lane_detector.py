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
        # Scaling
        self.scaling_x = self.col_sample_w * self.frame_w / self.model_input_width


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

    # TODO: optimise, cant we filter earlier?
    def get_lanes(self, frame):
        scores= self.get_lane_scores(frame)
        lanes = self.get_lane_coords(scores)
        # return torch.stack((x,y), dim=0).cpu().numpy()
        return self.filter_car_lanes(lanes)

    # def get_lane_coords(self,scores):
    #     lanes = []
    #
    #     # For each lane
    #     for i in range(scores.shape[1]):
    #         # Get row
    #         if torch.count_nonzero(scores[:, i]) > 2:
    #             # Convert scores to pixels
    #             for k in range(scores.shape[0]):
    #                 if scores[k, i] > 0:
    #                     ppp = (int(scores[k, i] * self.scaling_x) - 1,
    #                            int(self.frame_h * (self.row_anchors[self.cls_num_per_lane - 1 - k] / self.model_input_height)) - 1)
    #                     lanes.append(ppp)
    #     return lanes

    def get_lane_coords(self, scores):
        lanes = []

        # For each lane
        for i in range(scores.shape[1]):
            lane_pts = []  # points for this lane

            # Only consider lanes with enough points
            if torch.count_nonzero(scores[:, i]) > 2:
                for k in range(scores.shape[0]):  # for each row
                    if scores[k, i] > 0:
                        x = int(scores[k, i] * self.scaling_x) - 1
                        y = int(self.frame_h * (
                                    self.row_anchors[self.cls_num_per_lane - 1 - k] / self.model_input_height)) - 1
                        lane_pts.append((x, y))
            if lane_pts:
                lanes.append(lane_pts)
        return lanes

    def filter_car_lane(self,lanes):
        frame_center = self.frame_w / 2
        best_lane = min(lanes, key=lambda lane: abs(
            torch.tensor([x for x, y in lane], dtype=torch.float).mean() - frame_center))
        return best_lane

    def filter_car_lanes(self, lanes):
        """
        Select the two lanes closest to the frame center (left and right of the car)
        using the same logic as before.
        lanes: list of lanes, each lane is a list of (x, y) points
        Returns:
            best_lanes: list of two lanes
        """
        frame_center = self.frame_w / 2

        # Compute average x of each lane
        lane_avgs = [torch.tensor([x for x, y in lane], dtype=torch.float).mean() for lane in lanes]

        # Split lanes into left and right relative to frame center
        left_lanes = [lane for lane, avg in zip(lanes, lane_avgs) if avg < frame_center]
        right_lanes = [lane for lane, avg in zip(lanes, lane_avgs) if avg >= frame_center]

        # Pick the left lane closest to the center
        best_left = min(left_lanes, key=lambda lane: abs(
            torch.tensor([x for x, y in lane], dtype=torch.float).mean() - frame_center)) \
            if left_lanes else None

        # Pick the right lane closest to the center
        best_right = min(right_lanes, key=lambda lane: abs(
            torch.tensor([x for x, y in lane], dtype=torch.float).mean() - frame_center)) \
            if right_lanes else None

        # Return both lanes (filter out None)
        best_lanes = [lane for lane in [best_left, best_right] if lane is not None]

        return best_lanes

    # import torch
    #
    # def get_lane_coords(self, scores):
    #     """
    #     Convert soft lane scores to pixel coordinates for all lanes using PyTorch tensors.
    #     scores: (num_rows, num_lanes) tensor
    #     Returns:
    #         x: (num_rows, num_lanes)
    #         y: (num_rows, num_lanes)
    #         mask: boolean mask where lane exists
    #     """
    #     num_rows, num_lanes = scores.shape
    #
    #     # Mask of valid lane points
    #     mask = scores > 0
    #
    #     # X coordinates in pixels
    #     x = scores * self.frame_w / self.model_input_width
    #
    #     # Y coordinates in pixels
    #     row_anchors_tensor = torch.tensor(self.row_anchors, device=scores.device, dtype=torch.float)
    #     y_indices = torch.arange(self.cls_num_per_lane - 1, self.cls_num_per_lane - 1 - num_rows, -1,
    #                              device=scores.device)
    #     y = self.frame_h * row_anchors_tensor[y_indices][:, None] / self.model_input_height
    #     y = y.expand(-1, num_lanes)
    #
    #     # Apply mask
    #     x = x * mask.float()
    #     y = y * mask.float()
    #
    #     return x, y, mask
    #
    # def filter_car_lane(self, x, y, mask):
    #     """
    #     Select the lane closest to the center of the frame (assume car is in middle lane).
    #     x, y: (num_rows, num_lanes)
    #     mask: same shape, bool
    #     Returns:
    #         car_lane_points: (num_valid_points, 2) tensor
    #     """
    #     frame_center = self.frame_w / 2
    #
    #     # Compute mean x per lane (avoid division by zero)
    #     lane_sum = (x * mask.float()).sum(0)  # sum over rows
    #     lane_count = mask.sum(0).clamp(min=1)  # number of valid points per lane
    #     lane_avg_x = lane_sum / lane_count
    #
    #     # Choose lane closest to frame center
    #     car_lane_idx = torch.argmin(torch.abs(lane_avg_x - frame_center))
    #
    #     # Extract valid points for this lane
    #     valid_mask = mask[:, car_lane_idx]
    #     car_lane_points = torch.stack([x[:, car_lane_idx], y[:, car_lane_idx]], dim=1)[valid_mask]
    #
    #     return car_lane_points.cpu().numpy().astype(int)
