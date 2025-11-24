import cv2
import numpy as np

class POVVisualiser:
    def __init__(self,class_names,frame, boxes,class_ids,scores,distances):
        self.boxes = boxes
        self.class_ids = class_ids
        self.scores = scores
        self.distances = distances
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
        return frame

    def overlay_radar_points(self, projection, velocities):
        point_radius = 2
        u, v, z, Pc, kept = projection
        if u.size == 0 or len(self.boxes) == 0:
            return

        # Keep poinst that lie inside of any box, these are the ones we want to visualize
        boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        U = u[None, :]
        V = v[None, :]
        x1 = boxes[:, 0:1]
        y1 = boxes[:, 1:2]
        x2 = boxes[:, 2:3]
        y2 = boxes[:, 3:4]
        inside_any = (U >= x1) & (U <= x2) & (V >= y1) & (V <= y2)
        keep_mask = inside_any.any(axis=0)

        if not np.any(keep_mask):
            return

        # Filter the arrays to only those points that fall inside any box
        u_in = u[keep_mask]
        v_in = v[keep_mask]

        # Align velocities to the filtered points if provided and length matches
        vel_in = None
        if velocities is not None and velocities.size == u.size:
            vel_in = velocities[keep_mask]

        if velocities.size == u.size:
            vel_range = 7.5  # m/s window, same as CARLA example
            for ui, vi, vel in zip(u_in, v_in, vel_in):
                if vel <= -vel_range:  # strong negative -> RED
                    bgr = (0, 0, 255)
                elif vel >= vel_range:  # strong positive -> BLUE
                    bgr = (255, 0, 0)
                else:
                    # Linear blend in [-vr, +vr]: RGB = (255-a, a, 0)
                    a = int(((vel + vel_range) / (2.0 * vel_range)) * 255.0)
                    a = max(0, min(255, a))
                    # Convert RGB -> BGR for OpenCV
                    bgr = (0, a, 255 - a)
                cv2.circle(self.frame, (int(round(ui)), int(round(vi))), point_radius, bgr, -1)

    def show(self):
        frame_with_boxes_bgr=self.add_object_and_distance_overlay(self.frame)
        frame_with_trajectory_bgr=self.add_trajectory_overlay(frame_with_boxes_bgr)
        cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()
