import gradio as gr
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

# Пути и модель
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "best_model.onnx")
ort_sess = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_sess.get_inputs()[0].name

# Классы
classes = ["Apple", "Banana", "Orange"]

# Предобработка
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype(np.float32) / 255.0
    # Normalize
    x = (x - IMAGENET_MEAN) / IMAGENET_STD # (H,W,C) -> (C,H,W)
    x = np.transpose(x, (2, 0, 1)) # Add batch
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x

def predict(img: Image.Image):
    x = preprocess(img)
    logits = ort_sess.run(None, {input_name: x})[0][0]
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    return {cls: float(p) for cls, p in zip(classes, probs)} # Возвращаем float для Label в  Gradio

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3)
)

iface.launch()
