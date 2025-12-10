import torch

class SignClassifier:
    def __init__(self, use_tracking = True):
        # Initialize model
        print("CUDA:", torch.cuda.is_available())
        # TODO: init in your model
        self.model =
        # TODO load checkpoint into model

        # TODO place model in eval mode

        # TODO: define classess
        self.classes = []
        # TODO define transformations


    # TODO: this function should take image from carla as input and return a tensor
    # you will need to apply your defined transformations here as well
    def preprocess_frame(self,frame) -> torch.Tensor:


    def read_sign(self, frame):
        # TODO: call the preprocess fucntion here

        # TODO: Pass the return tensor through the model

        # TODO: extract label based on index from classes
