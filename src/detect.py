import cv2
import torch
from PIL import Image
import os
from ultralytics import YOLO

# Cargar el modelo
model = YOLO("src/model/best.pt")

# Realizar predicciones
preds = model('src/assets')

# Guardar cada imagen con las predicciones
for i, result in enumerate(preds):
    # Obtener la imagen con las anotaciones (en formato RGB)
    plotted_img = result.plot()  # Esto devuelve un array de numpy
    
    # Convertir a formato PIL y guardar
    img_name = os.path.basename(result.path)  # Nombre original del archivo
    output_path = os.path.join('runs', img_name)
    Image.fromarray(plotted_img).save(output_path)  # Guardar con PIL
