import cv2
from ultralytics import YOLO
import math
import requests

class ShopIA:
    # Init
    def __int__(self):
        # VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # MODELS:
        # Object model
        ObjectModel = YOLO('Modelos/yolov8l.onnx')
        self.ObjectModel = ObjectModel

        billModel = YOLO('Modelos/billBank2.onnx')
        self.billModel = billModel

        # CLASES:
        # Objects
        #clsObject = ObjectModel.names
        clsObject = ['donut', 'cell phone', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'banana', 'apple', 'orange', 'backpack', 'dog', 'mouse', 'keyboard', 'book', 'clock', 'scissors', 'toothbrush']
        self.clsObject = clsObject

        # Bills Bank
        clsBillBank = ['Billete10', 'Billete20', 'Billete50']
        self.clsBillBank = clsBillBank

        # Total balance
        total_balance = 0
        self.total_balance = total_balance
        self.pay = ''

        return self.cap

    # DRAW FUNCTIONS
    # Area
    def draw_area(self, img, color, xi, yi, xf, yf):
        img = cv2.rectangle(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    # Text
    def draw_text(self, img, color, text, xi, yi, size, thickness, back = False):
        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, thickness)
        dim = sizetext[0]
        baseline = sizetext[1]
        if back == True:
            img = cv2.rectangle(img, (xi, yi - dim[1] - baseline), (xi + dim[0], yi + baseline - 7),(0, 0, 0), cv2.FILLED)
        img = cv2.putText(img, text, (xi, yi - 5), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness)
        return img

    # Line
    def draw_line(self, img, color, xi, yi, xf, yf):
        img = cv2.line(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def area(self, frame, xi, yi, xf, yf):
        # Info
        al, an, c = frame.shape
        # Coordenates
        xi, yi = int(xi * an), int(yi * al)
        xf, yf = int(xf * an), int(yf * al)

        return xi, yi, xf, yf

    # MarketPlace list
    def marketplace_list(self, frame, object):
        # Realiza una solicitud GET a la API para obtener la información del producto
        try:
            response = requests.get(f'https://api-proyecto-seminario.onrender.com/products/{object}')
            if response.status_code == 200:
                product = response.json()
                price = product['price']

                # Asegúrate de convertir el precio a tipo numérico (int o float)
                price = float(price)  # O float(price) si tu precio tiene decimales
                
                if object not in [item[0] for item in self.shopping_list]:
                    self.shopping_list.append([object, price])
                    
                    # Configuración de texto
                    list_area_xi, list_area_yi, list_area_xf, list_area_yf = self.area(frame, 0.7739, 0.6250, 0.9649, 0.9444)
                    size_obj, thickness_obj = 0.60, 1
                    
                    # Mostrar el producto y el precio en la lista de compras
                    text = f'{object} --> Q{price}'
                    frame = self.draw_text(frame, (0, 255, 0), text, list_area_xi + 10,
                                        list_area_yi + (40 + (self.posicion_products * 20)),
                                        size_obj, thickness_obj, back=False)
                    self.posicion_products += 1
                    self.accumulative_price += price
            else:
                # En caso de que no se encuentre el producto o haya un error
                print(f"Producto no encontrado: {object}")
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con la API: {e}")
        
        return frame

    # Balance process
    def balance_process(self, bill_type):
        if bill_type == 'Billete10':
            self.balance = 10000
        elif bill_type == 'Billete20':
            self.balance = 20000
        elif bill_type == 'Billete50':
            self.balance = 50000

    # Payment process
    def payment_process(self, accumulative_price, accumulative_balance):
        payment = accumulative_balance - accumulative_price
        print(payment)
        if payment < 0:
            text = f'Falta cancelar {abs(payment)}Q'

        elif payment > 0:
            text = f'Su cambio es de: {abs(payment)}Q'
            self.accumulative_price = 0
            self.total_balance = 0

        elif payment == 0:
            text = f'Gracias por su compra!'
            self.accumulative_price = 0
            self.total_balance = 0

        return text

    # INFERENCE
    def prediction_model(self, clean_frame, frame, model, clase):
        bbox = []
        cls = 0
        conf = 0
        # Yolo | AntiSpoof
        results = model(clean_frame, stream=True, verbose=False)
        for res in results:
            # Box
            boxes = res.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Error < 0
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 < 0: x2 = 0
                if y2 < 0: y2 = 0

                bbox = [x1,y1,x2,y2]

                # Class
                cls = int(box.cls[0])

                # Confidence
                conf = math.ceil(box.conf[0])

                # Obtener el nombre de la clase detectada
                class_name = self.ObjectModel.names[cls]  # Esto obtiene el nombre de la clase del modelo YOLO

                # Verifica si el nombre de la clase está en tu lista de interés
                if class_name in self.clsObject:
                    if clase == 0:
                        # Draw
                        text_obj = f'{class_name} {int(conf * 100)}%'
                        # Marketplace list
                        frame = self.marketplace_list(frame, class_name)

                        # Draw
                        size_obj, thickness_obj = 0.75, 1
                        frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, size_obj, thickness_obj, back=True)
                        frame = self.draw_area(frame, (0, 255, 0), x1, y1, x2, y2)

                if clase == 1:
                    # Draw
                    bill_type = self.clsBillBank[cls]
                    text_obj = f'{self.clsBillBank[cls]} {int(conf * 100)}%'
                    self.balance_process(bill_type)

                    # Draw
                    size_obj, thickness_obj = 0.75, 1
                    frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, size_obj, thickness_obj, back=True)
                    frame = self.draw_area(frame, (0, 255, 0), x1, y1, x2, y2)
        return frame

    # Main
    def tiendaIA(self, cap):
        while True:
            # Frames
            ret, frame = cap.read()
            # Read keyboard
            t = cv2.waitKey(50)

            # Frame Object Detect
            clean_frame = frame.copy()

            # Info Marketplace list
            shopping_list = []
            self.shopping_list = shopping_list
            posicion_products = 1
            self.posicion_products = posicion_products
            accumulative_price = 0
            self.accumulative_price = accumulative_price
            # Info Payment process
            balance = 0
            self.balance = balance

            # Areas
            # Shopping area
            shop_area_xi, shop_area_yi, shop_area_xf, shop_area_yf = self.area(frame, 0.0351, 0.0486, 0.7539, 0.9444)
            # Draw
            color = (0,255,0)
            text_shop = f'Shopping area'
            size_shop, thickness_shop = 0.75, 1
            frame = self.draw_area(frame, color, shop_area_xi, shop_area_yi, shop_area_xf, shop_area_yf)
            frame = self.draw_text(frame, color, text_shop, shop_area_xi, shop_area_yf + 30, size_shop, thickness_shop)

            # Payment area
            pay_area_xi, pay_area_yi, pay_area_xf, pay_area_yf = self.area(frame, 0.7739, 0.0486, 0.9649, 0.6050)
            # Draw
            text_pay = f'Payment area'
            size_pay, thickness_pay = 0.50, 1
            frame = self.draw_line(frame, color, pay_area_xi, pay_area_yi, pay_area_xi, int((pay_area_yi+pay_area_yf)/2))
            frame = self.draw_line(frame, color, pay_area_xi, pay_area_yi, int((pay_area_xi+pay_area_xf)/2), pay_area_yi)
            frame = self.draw_line(frame, color, pay_area_xf, int((pay_area_yi+pay_area_yf)/2), pay_area_xf, pay_area_yf)
            frame = self.draw_line(frame, color, int((pay_area_xi+pay_area_xf)/2), pay_area_yf, pay_area_xf, pay_area_yf)
            frame = self.draw_text(frame, color, text_pay, pay_area_xf-100, shop_area_yi+10, size_pay, thickness_pay)

            # List area
            list_area_xi, list_area_yi, list_area_xf, list_area_yf = self.area(frame, 0.7739, 0.6250, 0.9649, 0.9444)
            # Draw
            text_list = f'Shopping List'
            size_list, thickness_list = 0.65, 1
            frame = self.draw_line(frame, color, list_area_xi, list_area_yi, list_area_xi, list_area_yf)
            frame = self.draw_line(frame, color, list_area_xi, list_area_yi, list_area_xf, list_area_yi)
            frame = self.draw_line(frame, color, list_area_xi+30, list_area_yi+30, list_area_xf-30, list_area_yi+30)
            frame = self.draw_text(frame, color, text_list, list_area_xi+55, list_area_yi+30, size_list, thickness_list)

            # Predict Object
            frame = self.prediction_model(clean_frame, frame, self.ObjectModel, clase=0)
            # Predict Bills Bank
            frame = self.prediction_model(clean_frame, frame, self.billModel, clase=1)

            # Accumulative Price Show
            text_price = f'Compra total: {self.accumulative_price} Q'
            frame = self.draw_text(frame, (0, 255, 0), text_price, list_area_xi + 10, list_area_yf, 0.60, 1, back=False)
            # Total Balance Show
            text_balance = f'Saldo total: {self.total_balance} Q'
            frame = self.draw_text(frame, (0, 255, 0), text_balance, list_area_xi + 10, list_area_yf + 30, 0.60, 1, back=False)
            # Payment
            frame = self.draw_text(frame, (0, 255, 0), self.pay, list_area_xi + - 300, list_area_yf + 30, 0.60, 1, back=False)


            # Show
            cv2.imshow("Tienda IA", frame)

            # Balance
            if t == 83 or t == 115:
                self.total_balance = self.total_balance + self.balance
                self.balance = 0
            # Payment
            if t == 80 or t == 112:
                self.pay = self.payment_process(self.accumulative_price, self.total_balance)
            # Exit
            if t == 27:
                break

        # Release
        self.cap.release()
        cv2.destroyAllWindows()