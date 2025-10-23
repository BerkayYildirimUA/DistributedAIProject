import os

import numpy as np

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
        shape = (480, 640, 3)
        dtype = np.uint8
        super().__init__(filename,shape,dtype)

class DepthCameraMemory(SharedMemory):
    def __init__(self):
        filename = "DEPTH_CAMERA_MEMORY.dat"
        shape = (480, 640)
        dtype = np.float32
        super().__init__(filename,shape,dtype)

class VehicleDistanceMemory(SharedMemory):
    def __init__(self):
        filename = "VEHICLE_DISTANCE_MEMORY.dat"
        shape = (1, 1)
        dtype = np.float32
        super().__init__(filename,shape,dtype)


