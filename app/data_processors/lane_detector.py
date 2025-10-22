import torch
import numpy as np
import scipy.special
import torchvision.transforms as transforms
from PIL import Image
from .model.model import parsingNet

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
class LaneDetector:
    def __init__(self, model_path='./data_processors/model/tusimple_18.pth'):
        self.cls_num_per_lane = 56
        self.griding_num = 100
        self.row_anchor = tusimple_row_anchor


        # --- Load model ---
        self.net = parsingNet(
            pretrained=False,
            backbone='18',
            cls_dim=(self.griding_num + 1, self.cls_num_per_lane, 4),
            use_aux=False
        ).cuda()

        state_dict = torch.load(model_path, map_location='cuda')
        # Fix potential "module." prefix mismatch
        state_dict = state_dict.get('model', state_dict)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.net.load_state_dict(new_state_dict, strict=False)
        self.net.eval()

        # --- Define preprocessing ---
        self.transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        # Precompute horizontal sample points
        self.col_sample = np.linspace(0, 800 - 1, self.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]

    def preprocess(self, img):
        """Convert PIL image to model input tensor"""
        self.frame_w, self.frame_h = img.shape[1], img.shape[0]

        img_pil = Image.fromarray(img)
        return self.transform(img_pil).unsqueeze(0).cuda()

    def get_lanes(self, img_tensor):
        img_tensor=self.preprocess(img_tensor)
        """Run inference and return lane polylines [(x1,y1), (x2,y2), ...] per lane"""
        with torch.no_grad():
            out = self.net(img_tensor)

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]  # flip horizontally
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.griding_num) + 1
        loc = np.sum(prob * idx.reshape(-1, 1, 1), axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.griding_num] = 0
        out_j = loc

        lanes = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * self.col_sample_w * self.frame_w / 800) - 1,
                               int(self.frame_h * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1)
                        lanes.append(ppp)
        # for i in range(out_j.shape[1]):  # per lane
        #     if np.sum(out_j[:, i] != 0) > 2:  # at least a few valid points
        #         pts = []
        #         for k in range(out_j.shape[0]):  # per row anchor
        #             if out_j[k, i] > 0:
        #                 x = int(out_j[k, i] * self.col_sample_w * self.img_w / 800) - 1
        #                 y = int(self.img_h * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1
        #                 pts.append((x, y))
        #         if len(pts) > 1:
        #             lanes.append(pts)
        return lanes
