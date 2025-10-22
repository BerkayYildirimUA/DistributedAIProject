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
    model = YOLO("runs/detect/train8/weights/best.pt")

    # Evaluate on test set
    model.val(
        data="data.yaml",
        task="test",  # run on test set instead of val
        imgsz=800,
        save=True , # saves predictions to runs/val,
        classes=[0,1,2]
    )


