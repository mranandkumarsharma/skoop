from fastapi import FastAPI, Query
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Classification API"}

# Load the saved tokenizer and model
model_path = "fake-news-bert-base-uncased"  # Path to the directory containing the saved model
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Define a function to perform inference using the loaded model
def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(1)
    predicted_label = torch.argmax(probs, dim=1).item()
    return predicted_label

# Define API endpoint for text classification using GET method
@app.get("/classify_text/")
def classify_text(text: str = Query(..., title="Text to classify")):
    predicted_label = get_prediction(text)
    return {"predicted_label": predicted_label}
