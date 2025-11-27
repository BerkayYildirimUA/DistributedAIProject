import cv2
import numpy as np

class POVVisualiser:
    def __init__(
        self,
        class_names,
        frame,
        boxes,
        class_ids,
        scores,
        distances,
        is_intersected,
        lanes=[],
        # tube_projector=None,
        speed_ms: float = 0.0,
        steer_rad: float = 0.0,
    ):
        self.boxes = boxes
        self.class_ids = class_ids
        self.scores = scores
        self.distances = distances
        if len(lanes) > 1:
            self.left_lane=lanes[0]
            self.right_lane=lanes[1]

        self.is_intersected=is_intersected
        self.class_names = class_names
        self.frame = frame
        self.speed_ms = float(speed_ms)
        self.steer_rad = float(steer_rad)

    def add_object_and_distance_overlay(self, frame_rgb):
        # teken boxes + labels (groen) op RGB, converteer daarna naar BGR voor imshow
        for distance, (x1, y1, x2, y2), score, cls_id in zip(
            self.distances, self.boxes, self.scores, self.class_ids
        ):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = self.class_names[int(cls_id)]
            color = (0, 255, 0)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_rgb,
                f"{cls_name} {int(round(float(score)*100))}%",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            cv2.putText(
                frame_rgb,
                f"{distance:.1f} m",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # cv2.imshow verwacht BGR
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # def add_trajectory_overlay(self, frame):
    #     for i, lane in enumerate(self.lanes):
    #         cv2.circle(frame, lane, 3, (0, 255, 0), -1)

    #         # color = colors[i % len(colors)]
    #         # cv2.polylines(frame, [np.array(lane, dtype=np.int32)], False, color, 4)
    #     return frame
    
    def add_trajectory_overlay(self, frame_bgr,color=(255,255,0),thickness=4):
        cv2.polylines(frame_bgr, [self.left_lane], False, color, thickness, cv2.LINE_AA)
        cv2.polylines(frame_bgr, [self.right_lane], False, color, thickness, cv2.LINE_AA)
        return frame_bgr

    def show(self):
        frame_with_boxes_bgr = self.add_object_and_distance_overlay(self.frame)
        frame_with_trajectory_bgr = self.add_trajectory_overlay(frame_with_boxes_bgr)

        cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()