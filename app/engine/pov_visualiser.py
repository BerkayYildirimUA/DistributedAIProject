import cv2
from app.data_processors.radar_points_projector import RadarPointsProjector

class POVVisualiser:
    def __init__(self,class_names,frame, boxes,class_ids,scores,distances):
        self.boxes = boxes
        self.class_ids = class_ids
        self.scores = scores
        self.distances = distances
        self.class_names = class_names
        self.frame = frame
        self.RadarPointsProjector = RadarPointsProjector()

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

    def overlay_radar_points(self, radar_points_world, K, P, img_w, img_h,
                             point_radius=2, color_mode='depth'):

        u, v, z, Pc, kept = self.RadarPointsProjector.project_radar_points_world_to_image(
            radar_points_world, K, P, img_w, img_h
        )
        if u.size == 0:
            return

        # Optional color ramp by depth (near=green, far=red)
        if color_mode == 'depth':
            z_min, z_max = float(np.min(z)), float(np.max(z))
            span = max(1e-6, z_max - z_min)

        for ui, vi, zi in zip(u, v, z):
            if color_mode == 'depth':
                t = (zi - z_min) / span
                bgr = (0, int(255 * (1.0 - t)), int(255 * t))
            else:
                bgr = (0, 0, 255)
            cv2.circle(self.frame, (int(round(ui)), int(round(vi))), point_radius, bgr, -1)

    def show(self):
        frame_with_boxes_bgr=self.add_object_and_distance_overlay(self.frame)
        frame_with_trajectory_bgr=self.add_trajectory_overlay(frame_with_boxes_bgr)
        cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()
