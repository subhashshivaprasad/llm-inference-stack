from fastapi import FastAPI
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np

app = FastAPI()

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
session = InferenceSession("model.onnx")

@app.get("/")
def root():
    return {"message": "LLM Inference API is running."}

@app.post("/predict/")
def predict(input_text: str):
    inputs = tokenizer(input_text, return_tensors="np")
    ort_inputs = {k: v for k, v in inputs.items()}
    outputs = session.run(None, ort_inputs)
    return {"logits": outputs[0].tolist()}
