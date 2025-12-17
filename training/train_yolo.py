from ultralytics import YOLO
import torch, torchvision
from torchvision.ops import nms

if __name__=='__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torchvision.__version__)
    print(nms)

    # we start by loading in the pretrained model
    model = YOLO("yolo11l.pt")

    # Next we freeze all the layers except the last one (will be used for fine tuning).
    # Why last layer and not another one? Because early layers contain generic visual features (edges,textures,shapes...).
    # The last layer (aka detection head) is the one that maps those features to my specific classes and bounding boxes.
    try:
        num_top_modules = len(model.model.model)
        freeze_backbone = max(num_top_modules - 1, 0)      # last = detect
    except Exception:
        # Fallback as attribute, for YOLO mostly 23
        freeze_backbone = 23

    print(f"Freezing first {freeze_backbone} layers (all but Detect).")

    # Train the model
    model.train(
        data="data.yaml",   # this tells YOLO where dataset spits are defined
        imgsz=832,          # input resolution, higher gives potentially better detection of small objects but uses more VRAM and is slower
        batch=16,           # '-1' tells ultralytics to auto pick a batch size that fits your GPU memory, our case 16 was mostly chosen
        epochs=300,         # this means the training loop can run up to 300 full passes over my training set
        patience=30,        # enables early stopping: if validation doesn’t improve for 30 epochs, training stops.
                            # this is also an “overfitting control” choice.
        amp=True,           # uses mixed precision, which speeds up training on modern GPUs and reduces memory usage
        optimizer="AdamW",  # AdamW is often more stable than plain SGD for fine-tuning, and it works well with modern YOLO training schedules.
        lr0=0.01,
        cache='disk',       # caches images to speed up training
        workers=6,          # dataloader parallelism (if memerror, then put 0)
        plots=True,         # makes Ultralytics generate training curves and a results.csv file.
                            # This allows us to explain training behaviour and overfitting.
        project="runs/detect",
        name="train_large_832imgsize_0.01lr_300epochs_16batch",
        exist_ok=True,      # if the map where you save exists then use that
        freeze=freeze_backbone
    )