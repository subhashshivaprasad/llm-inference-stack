from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
torch.onnx.export(
    model, (inputs["input_ids"],),
    "model.onnx",  # This saves the file in the root directory
    input_names=["input_ids"],
    output_names=["output"],
    opset_version=14
)
