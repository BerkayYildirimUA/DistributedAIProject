import cv2
import numpy as np


class POVVisualiser:
    def __init__(self,class_names,frame, boxes,class_ids,scores,distances,lanes):
        self.boxes = boxes
        self.class_ids = class_ids
        self.scores = scores
        self.distances = distances
        self.lanes = lanes
        self.class_names = class_names
        self.frame = frame

    def add_object_and_distance_overlay(self, frame):
        for distance,(x1, y1, x2, y2), score, cls_id in zip(self.distances,self.boxes, self.scores, self.class_ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = self.class_names[int(cls_id)]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{distance:.1f} m", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_with_boxes_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_with_boxes_bgr

    def add_trajectory_overlay(self, frame):
        colors = [(0, 255, 0), (0, 200, 255), (255, 0, 0), (0, 128, 255)]
        for i, lane in enumerate(self.lanes):
            cv2.circle(frame, lane, 5, (0, 255, 0), -1)

            # color = colors[i % len(colors)]
            # cv2.polylines(frame, [np.array(lane, dtype=np.int32)], False, color, 4)
        return frame

    def show(self):
        frame_with_boxes_bgr=self.add_object_and_distance_overlay(self.frame)
        frame_with_trajectory_bgr=self.add_trajectory_overlay(frame_with_boxes_bgr)
        cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()
