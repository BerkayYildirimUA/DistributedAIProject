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
    model = YOLO("yolo11m.pt")

    # Train the model
    model.train(data="data.yaml",
                imgsz=640,
                batch=8,
                epochs=350,
                device='cuda:0',
                workers=2,
                patience=50,
                plots=True,
                augment=False,
                val=True,
                classes=[0,1,2,3,4,5],
                amp=False,
                project="runs/detect",
                name="train_all_640_s",  # vaste naam (makkelijker voor eval/export)
                exist_ok=True
                )