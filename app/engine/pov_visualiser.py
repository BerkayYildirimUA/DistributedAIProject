import cv2
# #
# # class POVVisualiser:
# #     def __init__(self,class_names,frame, boxes,class_ids,scores,distances):
# #         self.boxes = boxes
# #         self.class_ids = class_ids
# #         self.scores = scores
# #         self.distances = distances
# #         self.class_names = class_names
# #         self.frame = frame
# #
# #     def add_object_and_distance_overlay(self, frame):
# #         for distance,(x1, y1, x2, y2), score, cls_id in zip(self.distances,self.boxes, self.scores, self.class_ids):
# #             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
# #             cls_name = self.class_names[int(cls_id)]
# #             color = (0, 255, 0)
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
# #             cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
# #             cv2.putText(frame, f"{distance:.1f} m", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
# #
# #         frame_with_boxes_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# #         return frame_with_boxes_bgr
# #
# #     def add_trajectory_overlay(self, frame):
# #         return frame
# #
# #     def show(self):
# #         frame_with_boxes_bgr=self.add_object_and_distance_overlay(self.frame)
# #         frame_with_trajectory_bgr=self.add_trajectory_overlay(frame_with_boxes_bgr)
# #         cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
# #         cv2.waitKey(1)
# #
# #     def cleanup(self):
# #         cv2.destroyAllWindows()
#
# import cv2
#
# from app.run_vehicle_pov import distances
#
#
# class POVVisualiser:
#     def __init__(self,class_names,frame, boxes,class_ids,scores,distancestube_projector=None, speed_ms=0.0, steer_rad=0.0):
#         self.boxes = boxes
#         self.class_ids = class_ids
#         self.scores = scores
#         self.distances = distances
#         self.class_names = class_names
#         self.frame = frame
#         self.tube_projector = self.tube_projector
#         self.speed_ms = speed_ms
#         self.steer_rad = steer_rad
#
#     def add_object_and_distance_overlay(self, frame):
#         for distance,(x1, y1, x2, y2), score, cls_id in zip(self.distances,self.boxes, self.scores, self.class_ids):
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#             cls_name = self.class_names[int(cls_id)]
#             color = (0, 255, 0)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             # score in percent, afgerond op hele procenten
#             cv2.putText(frame, f"{cls_name} {int(round(float(score)*100))}%",
#                         (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#             cv2.putText(frame, f"{distance:.1f} m", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#         frame_with_boxes_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         return frame_with_boxes_bgr
#
#     def add_trajectory_overlay(self, frame_bgr):
#         if self.tube_projector is None:
#             return frame_bgr
#         return self.tube_projector.project_and_draw(
#             frame_bgr, speed_ms=self.speed_ms, steer_rad=self.steer_rad,
#             color=(0, 255, 255), thickness=4
#         )
#
#     def show(self):
#         frame_with_boxes_bgr = self.add_object_and_distance_overlay(self.frame)
#         frame_with_trajectory_bgr = self.add_trajectory_overlay(frame_with_boxes_bgr)
#         cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
#         cv2.waitKey(1)
#
#     def cleanup(self):
#         cv2.destroyAllWindows()



class POVVisualiser:
    def __init__(
        self,
        class_names,
        frame,
        boxes,
        class_ids,
        scores,
        distances,
        tube_projector=None,
        speed_ms: float = 0.0,
        steer_rad: float = 0.0,
    ):
        self.boxes = boxes
        self.class_ids = class_ids
        self.scores = scores
        self.distances = distances
        self.class_names = class_names
        self.frame = frame                      # verwacht RGB
        self.tube_projector = tube_projector    # <-- juist
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

    def add_trajectory_overlay(self, frame_bgr):
        if self.tube_projector is None:
            return frame_bgr
        return self.tube_projector.project_and_draw(
            frame_bgr,
            speed_ms=self.speed_ms,
            steer_rad=self.steer_rad,
            color=(0, 255, 255),
            thickness=4,
        )

    def show(self):
        frame_with_boxes_bgr = self.add_object_and_distance_overlay(self.frame)
        frame_with_trajectory_bgr = self.add_trajectory_overlay(frame_with_boxes_bgr)
        cv2.imshow("Ego Vehicle POV", frame_with_trajectory_bgr)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()