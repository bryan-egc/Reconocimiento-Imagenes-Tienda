from ultralytics import YOLO

model = YOLO('Modelos/yolov8l.pt')

model.export(format='onnx')