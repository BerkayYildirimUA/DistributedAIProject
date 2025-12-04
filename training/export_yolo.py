from ultralytics import YOLO
import torch, torchvision
from torchvision.ops import nms
import onnx

if __name__=='__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torchvision.__version__)

    print(nms)

    # Load a model
    model = YOLO("runs/detect/train_all_640_m/weights/best.pt")

    # Export the model
    model.export(format="onnx",imgsz=640)

    # https: // github.com / ultralytics / ultralytics / issues / 16959  # issuecomment-2416464815
    # model_path = 'runs/detect/train8/weights/best.onnx'
    # model = onnx.load(model_path)
    # print(f"Current IR version: {model.ir_version}")

    # new_ir_version = 8
    # model.ir_version = new_ir_version
    #
    # updated_model_path = 'best.onnx'
    # onnx.save(model, updated_model_path)

    # print(f"IR version updated to: {new_ir_version}")




