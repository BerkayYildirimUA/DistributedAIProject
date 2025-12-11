import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import constants


class detected_classes_count:

    def make_mask(self, target_names, boxes, cls_names):
        return torch.tensor(
            [n in target_names for n in cls_names],
            dtype=torch.bool,
            device=boxes.device,
        )

    def count_objects(self, boxes, cls_names):
        is_traffic_sign = self.make_mask({constants.OBJECT_CLASS_NAMES[4]}, boxes, cls_names)
        is_pedestrian = self.make_mask({constants.OBJECT_CLASS_NAMES[5]}, boxes, cls_names)
        is_traffic_light = self.make_mask({constants.OBJECT_CLASS_NAMES[3]}, boxes, cls_names)
        is_vehicle_group = self.make_mask({constants.OBJECT_CLASS_NAMES[0], constants.OBJECT_CLASS_NAMES[1], constants.OBJECT_CLASS_NAMES[2]}, boxes, cls_names)

        traffic_sign_count = len(boxes[is_traffic_sign])
        pedestrian_count = len(boxes[is_pedestrian])
        traffic_light_count = len(boxes[is_traffic_light])
        vehicle_group_count = len(boxes[is_vehicle_group])

        total = traffic_sign_count + pedestrian_count + traffic_light_count + vehicle_group_count

        return {
            "vehicles": vehicle_group_count,
            "pedestrians": pedestrian_count,
            "traffic_lights": traffic_light_count,
            "traffic_signs": traffic_sign_count,
            "total": total,
        }


