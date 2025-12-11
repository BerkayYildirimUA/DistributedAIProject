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
        traffic_line_info=[]
    ):
        self.boxes = boxes
        self.class_ids = class_ids
        self.scores = scores
        self.distances = distances
        if len(lanes) > 1:
            self.left_lane=lanes[0]
            self.center_lane=lanes[1]
            self.right_lane=lanes[2]

        self.is_intersected=is_intersected
        self.class_names = class_names
        self.frame = frame
        self.traffic_line_info = traffic_line_info
    def add_object_and_distance_overlay(self, frame_rgb):
        # teken boxes + labels (groen) op RGB, converteer daarna naar BGR voor imshow
        for distance, (x1, y1, x2, y2), score, cls_id,is_intersecting in zip(
            self.distances, self.boxes, self.scores, self.class_ids,self.is_intersected
        ):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = self.class_names[int(cls_id)]
            color = (255, 0, 0) if is_intersecting else (0, 255, 0)
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
    #     for i, lane in enumerate([*self.left_lane, *self.right_lane]):
    #         cv2.circle(frame, lane, 3, (0, 255, 0), -1)
    #
    #         # color = colors[i % len(colors)]
    #         # cv2.polylines(frame, [np.array(lane, dtype=np.int32)], False, color, 4)
    #     return frame
    def add_trajectory_overlay(self, frame_bgr,color=(255,255,0),thickness=4):
        cv2.polylines(frame_bgr, [self.left_lane], False, color, thickness, cv2.LINE_AA)
        cv2.polylines(frame_bgr, [self.center_lane], False, color, thickness, 4)
        cv2.polylines(frame_bgr, [self.right_lane], False, color, thickness, cv2.LINE_AA)
        return frame_bgr

    def add_traffic_light_overlay(self,frame):
        tl_boxes_colored, tl_colors, tl_scores = self.traffic_line_info
        for (box, color, conf) in zip(tl_boxes_colored.tolist(), tl_colors, tl_scores.tolist()):
            x1, y1, x2, y2 = map(int, box)
            color_map = {"green": (0, 255, 0) ,"yellow": (0, 255, 255), "red": (0, 0, 255)}
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[color], 2)

            # place TL color above the class label (which is at y1-5)
            label = f"{color} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_top = max(0, y1 - 5 - th - 4)  # a bit higher than the class text

            cv2.rectangle(frame, (x1, y_top - th - 4), (x1 + tw + 4, y_top), (0, 0, 0), -1)
            cv2.putText(frame, f"{color} {conf:.2f}", (x1, max(0, y_top)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
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
        frame_with_traffic_light_info = self.add_traffic_light_overlay(self.frame)
        frame_with_boxes_bgr = self.add_object_and_distance_overlay(frame_with_traffic_light_info)
        frame_with_trajectory_bgr = self.add_trajectory_overlay(frame_with_boxes_bgr)
        cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()