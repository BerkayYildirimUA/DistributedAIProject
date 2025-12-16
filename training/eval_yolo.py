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
    model = YOLO("runs/detect/train_all_640_m/weights/best.pt")

    # Evaluate on test set
    metrics = model.val(
        data="data.yaml",
        task="test",  # run on test set instead of val
        imgsz=832,
        save=True , # saves predictions
    )
    # logging summary
    print({
        "map50-95": metrics.box.map,  # mAP@[.5:.95]
        "map50": metrics.box.map50,  # mAP@0.50
        "map75": metrics.box.map75,  # mAP@0.75
        "precision": metrics.box.p,
        "recall": metrics.box.r,
    })


