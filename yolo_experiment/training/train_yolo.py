from ultralytics import YOLO
import torch, torchvision
from torchvision.ops import nms

if __name__=='__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torchvision.__version__)

    print(nms)

    # Load a model
    model = YOLO("yolo11n.pt")

    # Train the model
    model.train(data="data.yaml",
                imgsz=800,
                batch=32,
                epochs=500,
                device=0,
                workers=4,
                patience=50,
                plots=True,
                augment=False,
                val=True,
                classes=[0,1,2])