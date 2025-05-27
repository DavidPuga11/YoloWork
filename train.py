from ultralytics import YOLO

# Caminho para o ficheiro YAML do dataset
dataset_path = "data.yaml"

# Carregar modelo base (YOLOv8n é o mais leve)
model = YOLO("yolov8n.pt")

# Treinar o modelo
model.train(
    data=dataset_path,
    epochs=30,
    imgsz=640,
    batch=8,
    project="posture_model",
    name="yolo_position",
    device='cpu'  # <- usa CPU para treinar
)
