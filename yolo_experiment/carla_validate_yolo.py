import queue
import random

import carla
import numpy as np
import cv2
import onnxruntime as ort

# https://medium.com/@zain.18j2000/how-to-use-your-yolov11-model-with-onnx-runtime-69f4ea243c01
# TODO: remove overlapping boxes
class ObjectDetection:
    def __init__(self):
        # CUDAExecutionProvider
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession("updated_model.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.classes = ["Vehicle", "Motor", "Bike","traffic light","traffic sign","pedestrian"]
        self.input_size = 800

    def preprocess_frame(self,frame):
        frame_w, frame_h = frame.shape[1], frame.shape[0]

        frame = cv2.resize(frame, (self.input_size, self.input_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1)
        frame = frame.reshape(1, 3, self.input_size, self.input_size)
        frame = frame/255.0
        frame=frame.astype(np.float32)
        return frame, frame_w, frame_h


    def get_objects(self, frame, conf_threshold=0.25):
        outputs = self.session.run(None, {self.input_name: frame})
        predictions = outputs[0].transpose()


        boxes, scores, class_ids = [], [], []

        # Filter predictions
        for pred in predictions:
            pred = pred.squeeze()
            if len(pred[4:])==0:
                continue
            class_id = pred[4:].argmax()
            conf = pred[4:].max()
            if conf < conf_threshold:
                continue
            x_center, y_center, w, h = pred[:4].astype(float)
            boxes.append((x_center, y_center, w, h))
            scores.append(conf)
            class_ids.append(class_id)

        return boxes, scores, class_ids

    def rescale_boxes(self, boxes,frame_w,frame_h):
        rescaled_boxes = []
        for box in boxes:
            x_center, y_center, w, h = box

            x_center = x_center / self.input_size * frame_w
            y_center = y_center / self.input_size * frame_h
            w = w / self.input_size * frame_w
            h = h / self.input_size * frame_h
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            rescaled_boxes.append((x1, y1, x2, y2))
        return rescaled_boxes

    def add_object_boxes(self, frame, boxes, scores, class_ids):
        for (x1,y1,x2,y2), cls_id, score in zip(boxes, class_ids, scores):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = self.classes[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def detect_and_add_overlay(self,frame):
        pro_frame, frame_w, frame_h = self.preprocess_frame(frame)
        boxes, scores, class_ids = self.get_objects(pro_frame)
        rescaled_boxes = self.rescale_boxes(boxes,frame_w,frame_h)
        frame = self.add_object_boxes(frame, rescaled_boxes, scores, class_ids)
        frame_with_boxes_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_with_boxes_bgr

# ---------------------------
# CARLA setup
# ---------------------------
object_detection = ObjectDetection()

client = carla.Client('localhost', 2000)
client.set_timeout(50.0)
world = client.get_world()

settings = world.get_settings()
# settings.synchronous_mode = True
# settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)


client.load_world('Town05')
blueprint_library = world.get_blueprint_library()
vehicle_blueprints = blueprint_library.filter('*vehicle*')
ego_bp = blueprint_library.find('vehicle.tesla.model3')

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()
# Spawn 50 vehicles randomly distributed throughout the map
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

spawned=False
while not spawned:
    try:
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        spawned = True
    except:
        print("Trying other spawn location")

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5),carla.Rotation(pitch=0, yaw=0, roll=0))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("sensor_tick", "0.033")
# camera_bp.set_attribute("sensor_tick", "0.05")  # match fixed_delta_seconds


# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    new_frame = array[:, :, :3]
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    frame_with_boxes=object_detection.detect_and_add_overlay(new_frame)
    cv2.imshow("Camera", frame_with_boxes)
    cv2.waitKey(1)
image_queue = queue.Queue()
camera.listen(image_queue.put)
EgoLocation = ego_vehicle.get_location()

traffic_manager = client.get_trafficmanager()
for vehicle in world.get_actors().filter('*vehicle*'):
    if vehicle.id != ego_vehicle.id:
        vehicle.set_autopilot(True,traffic_manager.get_port())
ego_vehicle.set_autopilot(True, traffic_manager.get_port())

# ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

spectator = world.get_spectator()
while True:
    world.wait_for_tick()
    transform = ego_vehicle.get_transform()
    # Compute position 10m behind and 5m above ego car
    forward_vector = transform.get_forward_vector()
    spectator_location = transform.location - 10 * forward_vector + carla.Location(z=5)

    spectator_transform = carla.Transform(spectator_location, transform.rotation)
    spectator.set_transform(spectator_transform)
    try:
        image_carla = image_queue.get_nowait()
        camera_callback(image_carla)
    except queue.Empty:
        continue




