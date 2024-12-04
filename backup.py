import cv2
import numpy as np
import pyttsx3
import time
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolo11s.pt")  # Asegúrate de tener el modelo descargado

# Parámetros de la cámara (esto depende de la calibración de tu cámara)
focal_length = 800  # Longitud focal (en píxeles)
known_width = 0.5  # Ancho real del objeto (en metros, por ejemplo, el ancho promedio de una persona)
real_height = 1.7  # Altura real del objeto (en metros, por ejemplo, la altura promedio de una persona)

# Iniciar la cámara
cap = cv2.VideoCapture("http://172.20.10.9:8080/video")  # '0' para la cámara web predeterminada

# Inicializar pyttsx3 para síntesis de voz
engine = pyttsx3.init()

# Diccionario para almacenar objetos detectados y el tiempo de la primera mención
detected_objects = {}

while True:
    ret, frame = cap.read()  # Capturar un cuadro del video
    if not ret:
        break  # Si no se puede capturar, salir del loop
    
    # Realizar la detección de objetos con YOLOv8
    results = model(frame)  # La imagen se pasa al modelo

    # Obtener las predicciones desde 'results' de la forma correcta
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas de la caja delimitadora (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.cpu().numpy()  # Confianza de la detección
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # ID de clase detectada
    
    # Dibujar las cajas y las etiquetas de las clases
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        if confidence > 0.5:  # Umbral de confianza
            label = f'{model.names[class_id]} {confidence:.2f}'  # Etiqueta con clase y confianza
            
            # Estimación de la distancia basada en el tamaño del objeto (en píxeles) y el tamaño real
            object_width = x2 - x1  # Ancho del objeto en píxeles
            # Estimación de la distancia (con la fórmula de la distancia)
            distance = (known_width * focal_length) / object_width  # Distancia en metros

            # Agregar la distancia a la etiqueta
            label += f' - Dist: {distance:.2f}m'

            # Dibujar la caja y la etiqueta
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Dibuja la caja
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Dibuja la etiqueta
            
            object_name = model.names[class_id]

            # Si el objeto no ha sido detectado previamente o han pasado 20 segundos desde la última mención
            current_time = time.time()
            if object_name not in detected_objects or (current_time - detected_objects[object_name]['time']) >= 15:
                # Sintetizar la voz para decir el objeto y su distancia
                text_to_say = f'{object_name} a una distancia de {distance:.2f} metros'
                engine.say(text_to_say)  # Convertir texto en voz
                engine.runAndWait()  # Reproducir la voz

                # Almacenar el objeto con el tiempo de la última mención
                detected_objects[object_name] = {
                    'label': object_name,
                    'distance': distance,
                    'time': current_time  # Guardamos el tiempo en el que fue mencionado
                }

    # Mostrar el cuadro con la predicción en tiempo real
    cv2.imshow("YOLOv8 Real-time Object Detection with Distance Estimation", frame)

    # Si presionas 'q', el loop se detendrá
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
