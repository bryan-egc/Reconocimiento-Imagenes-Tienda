
# Proyecto Final - Inteligencia Artificial

Proyecto final - curso Inteligencia Articial---

## 🚀 Integrantes del grupo  

🙍‍♂️: Bryan Ernesto Gámez Cipriano

🪪: 9490-19-3938

🙍‍♂️: Helder Iván Ajcalón Jacobo

🪪: 9490-20-10326

🙍‍♂️: Rudy Alexander Amado Soto Rosil

🪪: 9490-20-387

🙍‍♂️: Alfredo Geovanni Ramirez Tzunun

🪪: 9490-20-427

🙍‍♂️: Daniel Antonio Hernández Santos

🪪: 9490-20-1190

## Descripción del proyecto

Este proyecto utiliza técnicas de inteligencia artificial para reconocer y clasificar productos y billetes en imágenes capturadas dentro de una tienda. Emplea modelos de detección de objetos, específicamente YOLO, para identificar artículos como bolsas, botellas y más, así como diferentes denominaciones de billetes. Además, el sistema puede calcular el precio total de los artículos detectados y gestionar el proceso de pago, proporcionando retroalimentación en tiempo real sobre el saldo y el cambio. El flujo de video y la visualización de resultados se manejan usando OpenCV.

## Tecnologías utilizadas

| Tecnología        | Descripción                                                                                         |
|-------------------|-----------------------------------------------------------------------------------------------------|
| **OpenCV**       | Biblioteca de visión por computadora que permite la captura, procesamiento y análisis de imágenes en tiempo real.                |
| **YOLO (You Only Look Once**    | Algoritmo de detección de objetos en tiempo real que puede identificar múltiples objetos en una sola pasada de la imagen.   |
| **Python** | Lenguaje de programación utilizado para implementar la lógica del reconocimiento de imágenes y la interacción con los modelos de IA.                                           |
| **Ultralytics**      | Implementación de YOLO en Python que facilita el uso y la integración de modelos de detección de objetos.                    |
| **Modelos YOLOv8**       | Modelos preentrenados utilizados para la detección de objetos y billetes en las imágenes. Descarga los modelos del siguiente enlace: https://drive.google.com/drive/folders/1RlE-Hz7JqQZ5wcb5P_Ooc8ypKeC1nCDZ|

## Instrucciones de instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/bryan-egc/Reconocimiento-Imagenes-Tienda.git. 

2. **Navegar al directorio del proyecto:**
    ```bash
    cd Reconocimiento-Imagenes-Tienda 
3. **Instalar las dependencias:**
    - Asegúrate de tener Python 3.x instalado.
    - Instala las dependencias requeridas (puedes utilizar un entorno virtual):
    ```bash
        pip install opencv-python
        pip install ultralytics
4. **Descargar los modelos:**
    - Descarga los modelos YOLOv8 desde este enlace: https://drive.google.com/drive/folders/1RlE-Hz7JqQZ5wcb5P_Ooc8ypKeC1nCDZ
    - Coloca los archivos descargados en la carpeta Modelos dentro del directorio del proyecto.
5. **Ejecutar el script principal:**
    ```bash
        python Tienda.py

## Ejemplos de uso

- **Ejemplo 1:** Detección de bananas y botella con sus correspondientes porcentajes de confianza.
![image](https://github.com/bryan-egc/Reconocimiento-Imagenes-Tienda/assets/41811332/cfe704d4-a74a-4ce5-a99c-c230ae807429)

- **Ejemplo 2:** Detección de mouse y teclado con sus correspondientes porcentajes de confianza.
![image](https://github.com/bryan-egc/Reconocimiento-Imagenes-Tienda/assets/41811332/078e3a87-ffcf-48fb-b289-76a27bdb915e)

- **Ejemplo 3:** Detección de cuchillo, cuchara y tijeras con sus correspondientes porcentajes de confianza.
![image](https://github.com/bryan-egc/Reconocimiento-Imagenes-Tienda/assets/41811332/3952f857-ef7f-4070-b25a-2ef37d73b6ac)


## Observaciones

- **Compatibilidad de Cámara:** Asegúrate de que la cámara esté correctamente conectada y funcionando antes de ejecutar el script. El sistema utiliza la cámara para capturar las imágenes en tiempo real.

- **Ubicación de los Modelos:** Los modelos preentrenados deben estar en la carpeta `Modelos` dentro del directorio del proyecto. Si la carpeta no existe, créala y coloca los archivos allí.

- **Requisitos de Hardware:** Para obtener un rendimiento óptimo, se recomienda utilizar una máquina con una GPU compatible, ya que el procesamiento de imágenes y la detección en tiempo real pueden ser intensivos en recursos.

- **Precisión del Modelo:** La precisión del reconocimiento de objetos y billetes puede variar dependiendo de la calidad del modelo y de las imágenes capturadas. Asegúrate de utilizar modelos bien entrenados y de alta calidad.

- **Entorno de Desarrollo:** Se recomienda utilizar un entorno virtual para manejar las dependencias del proyecto y evitar conflictos con otras instalaciones de Python en tu sistema.

- **Actualización de Dependencias:** Asegúrate de que las dependencias estén actualizadas para evitar problemas de compatibilidad.

- **Errores Comunes:** Si encuentras problemas al ejecutar el script, verifica que todas las dependencias estén instaladas correctamente y que los modelos estén en la ubicación correcta. Consulta los mensajes de error para obtener pistas sobre cómo resolver los problemas.


## Licencia

MIT License
