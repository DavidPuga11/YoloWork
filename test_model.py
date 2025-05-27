import sys
import cv2
from ultralytics import YOLO

# Caminho para o modelo
MODEL_PATH = "posture_model/yolo_position2/weights/best.pt"

def main():
    if len(sys.argv) < 2:
        print("❌ Por favor indica o caminho da imagem.")
        return

    image_path = sys.argv[1]

    # Carregar o modelo treinado
    model = YOLO(MODEL_PATH)

    # Fazer a predição
    print(f"🔍 A analisar: {image_path}")
    results = model(image_path, conf=0.5)

    # Mostrar classes detetadas
    for r in results:
        boxes = r.boxes
        if boxes:
            print(f"✅ Detetado: {[model.names[int(cls)] for cls in boxes.cls]}")
        else:
            print("❌ Nada detetado.")

        # Mostrar imagem com deteções
        img = r.plot()  # Desenha boxes na imagem
        cv2.imshow("Resultado", img)
        cv2.waitKey(0)  # Espera até tecla ser premida
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
