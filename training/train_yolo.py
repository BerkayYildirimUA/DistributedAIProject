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



# from ultralytics import YOLO
#
# model = YOLO("runs/detect/train_all_640_s/weights/best.pt")
#
# model.train(
#     data="data.yaml",
#     imgsz=704,           # iets kleiner dan 736 → nog minder mem, grotere batch
#     batch=-1,
#     epochs=15,
#     patience=7,
#     amp=True,
#     optimizer="AdamW",
#     lr0=0.001,
#     # >>> RAM sparen:
#     cache='disk',        # i.p.v. 'ram'
#     workers=0,           # Windows-stabiel (geen spawn-kopieën)
#     pin_memory=False,    # werkt in recente versies; zo niet gewoon weglaten
#     # >>> Geen zware aug voor finetune:
#     mosaic=0.0,
#     copy_paste=0.0,
#     mixup=0.0,
#     close_mosaic=0,      # irrelevant als mosaic=0, maar voorkomt reinit
#     rect=True,           # sneller, minder padding (shuffle=False is normaal)
#     plots=False,
#     val=False,
#     freeze=10,
#     project="runs/detect",
#     name="finetune_704_noram",
#     exist_ok=True,
# )
#
# # Valideer één keer achteraf:
# YOLO("runs/detect/finetune_704_noram/weights/best.pt").val(data="data.yaml", imgsz=704)
