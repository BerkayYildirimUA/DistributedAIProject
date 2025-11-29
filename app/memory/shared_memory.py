import os

import numpy as np
import constants

class _SharedMemory(np.memmap):
    def write(self,data):
        np.copyto(self, data)
        self.flush()
    def read(self):
        return self.copy()

class SharedMemory:
    def __init__(self, filename,shape,dtype):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        memory_dir = os.path.join(base_dir, "memory_files")

        os.makedirs(memory_dir, exist_ok=True)
        self.filename = os.path.join(memory_dir, filename)
        self.shape = shape
        self.dtype = dtype

        # Create file if it doesn't exist
        nbytes = np.prod(shape) * np.dtype(dtype).itemsize
        if not os.path.exists(self.filename):
            with open(self.filename, "wb") as f:
                f.truncate(nbytes)
            print(f"Memory {self.__class__.__name__} created!")

    def get_write_access(self):
        return _SharedMemory(self.filename, dtype=self.dtype, mode='r+', shape=self.shape)
    def get_read_access(self):
        return _SharedMemory(self.filename, dtype=self.dtype, mode='r', shape=self.shape)

class RGBCameraMemory(SharedMemory):
    def __init__(self):
        filename = "RGB_CAMERA_MEMORY.dat"
        shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)
        dtype = np.uint8
        super().__init__(filename,shape,dtype)

class DepthCameraMemory(SharedMemory):
    def __init__(self):
        filename = "DEPTH_CAMERA_MEMORY.dat"
        shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)
        dtype = np.float32
        super().__init__(filename,shape,dtype)

class VehicleDistanceMemory(SharedMemory):
    def __init__(self):
        filename = "VEHICLE_DISTANCE_MEMORY.dat"
        shape = (1, 1)
        dtype = np.float32
        super().__init__(filename,shape,dtype)

class RadarMemory(SharedMemory):
    def __init__(self):
        filename = "RADAR_MEMORY.dat"
        shape = (constants.RADAR_MAX_DETECTIONS, 5)
        dtype = np.float32
        super().__init__(filename, shape, dtype)

class CameraCalibrationMemory(SharedMemory):
    def __init__(self):
        filename = "CAMERA_CALIBRATION_MEMORY.dat"
        shape = (2, 4, 4)  # store two 4x4 matrices; the intrinsic will just use the top-left 3x3
        dtype = np.float32
        super().__init__(filename, shape, dtype)

class VehicleStateMemory(SharedMemory):
    def __init__(self):
        filename = "VEHICLE_STATE_MEMORY.dat"
        shape = (2,)           # [speed_ms, steer_rad]
        dtype = np.float32
        super().__init__(filename, shape, dtype)

class LaneTubeMemory(SharedMemory):
    def __init__(self, max_pts: int = 256):
        """
        shape = (2, max_pts, 2)
        [0] = left polyline, [1] = right polyline
        punten genormaliseerd: x/=img_w, y/=img_h; lege posities = (-1, -1)
        """
        filename = "LANE_TUBE_MEMORY.dat"
        shape = (2, max_pts, 2)
        dtype = np.float32
        super().__init__(filename, shape, dtype)

class FrameIdMemory(SharedMemory):
    def __init__(self):
        filename = "FRAME_ID_MEMORY.dat"
        shape = (1,)
        dtype= np.int32
        super().__init__(filename, shape, dtype)