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
    model = YOLO("yolo11l.pt")

    # Freeze alles behalve de Detect-head
    try:
        # In v8/v11: backbone+neck+head als ModuleList
        num_top_modules = len(model.model.model)           # <-- dit is de juiste 'len'
        freeze_backbone = max(num_top_modules - 1, 0)      # laatste = Detect
    except Exception:
        # Fallback als attribuut anders heet; voor YOLO11* is Detect meestal index 23
        freeze_backbone = 23

    print(f"Freezing first {freeze_backbone} layers (all but Detect).")

    # Train the model
    model.train(
        data="data.yaml",
        imgsz=640,
        batch=-1,
        epochs=70,
        patience=30,
        amp=True,
        optimizer="AdamW",
        lr0=0.002,
        cache='disk',
        workers=6,          # als je nog MemoryError krijgt op Windows: zet dit op 0
        plots=True,         # zorg voor curves (results.png & results.csv)
        project="runs/detect",
        name="trainm_all_640_s",
        exist_ok=True,
        freeze=freeze_backbone
    )