import cv2
import numpy as np
import torch
from TrafficLights.TL_color_detector import TL_color_detector
from data_processors.object_detector import ObjectDetector
from data_processors.object_distance_calculator import ObjectDistanceCalculator
from memory.shared_memory import RGBCameraMemory, DepthCameraMemory, VehicleDistanceMemory
from engine.pov_visualiser import POVVisualiser

# Attach to shared memory
rgb_camera_memory = RGBCameraMemory().get_read_access()
depth_camera_memory = DepthCameraMemory().get_read_access()
vehicle_distance_memory = VehicleDistanceMemory().get_write_access()

object_detector = ObjectDetector()
object_distance_calculator=ObjectDistanceCalculator()

tl_color_detector = TL_color_detector()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tl_model = TL_color_detector.load_tl_model("traffic_light_classifier.pth", device)


# # ---- Toggle: set True to use HSV rule-based color classifier instead of the CNN
# USE_HSV = True
#
# # ---- HSV helpers (standalone, no torch model needed)
# def hsv_color_from_bgr_crop(bgr_crop):
#     import numpy as np, cv2
#     if bgr_crop is None or bgr_crop.size == 0:
#         return "unknown", 0.0
#     hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
#     H, S, V = cv2.split(hsv)
#
#     # Adaptive thresholds based on brightness
#     v_mean = float(np.mean(V))
#     s_thr = 60 if v_mean < 100 else 80
#     v_thr = 80 if v_mean < 100 else 120
#
#     red_mask  = ((H < 10) | (H > 170)) & (S > s_thr) & (V > v_thr)
#     yellow_m  = (H >= 15) & (H <= 35)  & (S > s_thr) & (V > v_thr)
#     green_m   = (H >= 40) & (H <= 85)  & (S > s_thr) & (V > v_thr)
#
#     def score(mask):
#         if mask.sum() == 0:
#             return 0
#         m = (mask.astype(np.uint8) * 255)
#         m = cv2.medianBlur(m, 3)  # small denoise
#         return int((m > 0).sum())
#
#     scores = {"red": score(red_mask), "yellow": score(yellow_m), "green": score(green_m)}
#     label = max(scores, key=scores.get)
#     bright = (V > v_thr)
#     denom = int(bright.sum()) or 1
#     conf = float(scores[label]) / denom
#     if scores[label] == 0:
#         return "unknown", 0.0
#     return label, conf
#
# def predict_colors_batch_hsv(frame_bgr, boxes_xyxy, class_order, conf_threshold=None, pad_ratio=0.02):
#     """
#     HSV-based batch classifier with same output signature as TL_color_detector.predict_colors_batch:
#       returns (boxes_xyxy, class_ids, scores) as tensors.
#     class_order: list like ["red","yellow","green"] used to map ids.
#     """
#     import torch, numpy as np
#     if boxes_xyxy is None or boxes_xyxy.numel() == 0:
#         return (torch.empty((0, 4), dtype=torch.float32),
#                 torch.empty((0,), dtype=torch.long),
#                 torch.empty((0,), dtype=torch.float32))
#
#     H, W = frame_bgr.shape[:2]
#     out_boxes, class_ids, scores = [], [], []
#
#     boxes_cpu = boxes_xyxy.detach().cpu()
#     for (x1, y1, x2, y2) in boxes_cpu.tolist():
#         x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#         pw, ph = int(pad_ratio * (x2 - x1)), int(pad_ratio * (y2 - y1))
#         x1p, y1p = max(0, x1 - pw), max(0, y1 - ph)
#         x2p, y2p = min(W, x2 + pw), min(H, y2 + ph)
#         if x2p <= x1p or y2p <= y1p:
#             continue
#
#         crop = frame_bgr[y1p:y2p, x1p:x2p]
#         label, conf = hsv_color_from_bgr_crop(crop)
#         if label == "unknown":
#             continue
#         if conf_threshold is not None and conf < float(conf_threshold):
#             continue
#
#         out_boxes.append([x1, y1, x2, y2])
#         class_ids.append(int(class_order.index(label)))
#         scores.append(float(conf))
#
#     if not out_boxes:
#         return (torch.empty((0, 4), dtype=torch.float32),
#                 torch.empty((0,), dtype=torch.long),
#                 torch.empty((0,), dtype=torch.float32))
#
#     return (torch.tensor(out_boxes, dtype=torch.float32),
#             torch.tensor(class_ids, dtype=torch.long),
#             torch.tensor(scores, dtype=torch.float32))



try:
    import time
    while True:
        # Convert to Torch tensor and normalize
        frame=rgb_camera_memory.read()
        depth_map = depth_camera_memory.read()
        if np.count_nonzero(frame) == 0:
            # No data yet, skip this iteration
            continue
        # Detect objects
        boxes, class_ids, scores =object_detector.get_objects(frame)

        # --- Stage 2: select only traffic lights ---
        if len(class_ids) > 0:
            cls_names = [object_detector.classes[int(c)] for c in class_ids.tolist()]
            is_tl = torch.tensor([n == "traffic light" for n in cls_names],
                                 dtype=torch.bool, device=boxes.device)
            tl_boxes = boxes[is_tl]
        else:
            tl_boxes = torch.empty((0, 4))

        # --- Stage 3: classify traffic light colors ---
        if len(tl_boxes) > 0:
            tl_boxes_colored, tl_color_ids, tl_scores = tl_color_detector.predict_colors_batch(frame, tl_boxes)
            tl_colors = [tl_color_detector.classes[int(i)] for i in tl_color_ids]
            for (box, color, conf) in zip(tl_boxes_colored.tolist(), tl_colors, tl_scores.tolist()):
                x1, y1, x2, y2 = map(int, box)
                color_map = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[color], 2)

                # place TL color ABOVE the class label (which is at y1-5)
                label = f"{color} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y_top = max(0, y1 - 5 - th - 4)  # a bit higher than the class text

                cv2.rectangle(frame, (x1, y_top - th - 4), (x1 + tw + 4, y_top), (0, 0, 0), -1)
                cv2.putText(frame, f"{color} {conf:.2f}", (x1, max(0, y_top)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            tl_boxes_colored = torch.empty((0, 4))
            tl_color_ids = torch.empty((0,))
            tl_scores = torch.empty((0,))





        # Get distance for each object
        distances=object_distance_calculator.get_distances(boxes,depth_map)

        # Visualise
        visualiser= POVVisualiser(
            object_detector.classes,
            frame,boxes,
            class_ids,
            scores,
            distances)
        visualiser.show()

finally:
    cv2.destroyAllWindows()


