import cv2
from ultralytics import YOLO
import math
import requests
import threading

class ShopIA:
    # Init
    def __init__(self):
        # VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # MODELS:
        try:
            # Cargar el modelo YOLO y moverlo a la GPU si está disponible
            self.ObjectModel = YOLO('Modelos/yolov8l.onnx')  # Cambiado a yolov8n para optimizar la velocidad
        except Exception as e:
            print(f"Error al cargar el modelo YOLO: {e}")
            return

        # CLASES:
        self.clsObject = ['donut', 'cell phone', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'banana', 'apple', 'orange', 
                          'backpack', 'dog', 'mouse', 'keyboard', 'book', 'clock', 'scissors', 'toothbrush']

        # Total balance
        self.total_balance = 0
        self.pay = ''
        self.cached_products = {}  # Caché para respuestas de la API

    # DRAW FUNCTIONS
    def draw_area(self, img, color, xi, yi, xf, yf, thickness=2, style='solid'):
        overlay = img.copy()
        # Fondo semitransparente para mejorar visibilidad
        cv2.rectangle(overlay, (xi, yi), (xf, yf), color, -1)
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)  # Añadir transparencia
        return img

    def draw_text(self, img, color, text, xi, yi, size, thickness, back=True):
        img = cv2.putText(img, text, (xi, yi), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness)
        return img

    def area(self, frame, xi, yi, xf, yf):
        al, an, c = frame.shape
        xi, yi = int(xi * an), int(yi * al)
        xf, yf = int(xf * an), int(yf * al)
        return xi, yi, xf, yf

    def draw_colorful_bbox(self, img, bbox, label, color=(50, 150, 255)):
        x1, y1, x2, y2 = bbox
        overlay = img.copy()
        # Dibujar cuadros semitransparentes
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        # Añadir borde al cuadro
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Texto en el cuadro
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    # Optimización con caché y llamadas asincrónicas
    def marketplace_list(self, frame, object):
        # Utilizar caché para evitar múltiples solicitudes API
        if object in self.cached_products:
            price = self.cached_products[object]
        else:
            try:
                response = requests.get(f'https://api-proyecto-seminario.onrender.com/products/{object}')
                if response.status_code == 200:
                    product = response.json()
                    price = float(product['price'])  # Convertir el precio a float
                    self.cached_products[object] = price  # Guardar en caché
                else:
                    print(f"Producto no encontrado: {object}")
                    return frame
            except requests.exceptions.RequestException as e:
                print(f"Error al conectar con la API: {e}")
                return frame

        # Mostrar el producto y el precio en la lista de compras
        list_area_xi, list_area_yi, list_area_xf, list_area_yf = self.area(frame, 0.7739, 0.0486, 0.9649, 0.9444)
        size_obj, thickness_obj = 0.60, 1
        text = f'{object} --> Q{price}'
        frame = self.draw_text(frame, (200, 200, 255), text, list_area_xi + 10, list_area_yi + (40 + (self.posicion_products * 20)), size_obj, thickness_obj, back=False)
        self.posicion_products += 1
        self.accumulative_price += price

        return frame

    # Optimizar procesamiento de predicciones en la GPU
    def prediction_model(self, clean_frame, frame, model, clase):
        results = model(clean_frame, stream=True, verbose=False)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0])

                class_name = self.ObjectModel.names[cls]

                if class_name in self.clsObject:
                    if clase == 0:
                        label = f'{class_name} {int(conf * 100)}%'
                        frame = self.marketplace_list(frame, class_name)
                        frame = self.draw_colorful_bbox(frame, [x1, y1, x2, y2], label, (50, 150, 255))  # Azul claro
        return frame

    # Main
    def tiendaIA(self, cap):
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            frame_counter += 1

            # Procesar solo cada 2do frame para reducir la carga de procesamiento
            if frame_counter % 2 != 0:
                continue

            clean_frame = frame.copy()
            self.posicion_products = 1
            self.accumulative_price = 0

            # Áreas rediseñadas con transparencia
            shop_area_xi, shop_area_yi, shop_area_xf, shop_area_yf = self.area(frame, 0.0351, 0.0486, 0.7539, 0.9444)
            frame = self.draw_area(frame, (100, 100, 255), shop_area_xi, shop_area_yi, shop_area_xf, shop_area_yf)
            frame = self.draw_text(frame, (255, 255, 255), "Shopping area", shop_area_xi, shop_area_yf + 30, 0.75, 1)

            # Área de la lista redimensionada con transparencia
            list_area_xi, list_area_yi, list_area_xf, list_area_yf = self.area(frame, 0.7739, 0.0486, 0.9649, 0.9444)
            frame = self.draw_area(frame, (180, 200, 255), list_area_xi, list_area_yi, list_area_xf, list_area_yf)
            frame = self.draw_text(frame, (255, 255, 255), "Shopping List", list_area_xi+55, list_area_yi+30, 0.65, 1)

            # Procesar frames alternados para mejorar rendimiento
            frame = self.prediction_model(clean_frame, frame, self.ObjectModel, clase=0)

            text_price = f'Compra total: {self.accumulative_price} Q'
            frame = self.draw_text(frame, (255, 255, 255), text_price, list_area_xi + 10, list_area_yf, 0.60, 1)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            cv2.imshow("Tienda IA", frame)

        self.cap.release()
        cv2.destroyAllWindows()


# Ejecutar la aplicación
shop_app = ShopIA()
shop_app.tiendaIA(shop_app.cap)
