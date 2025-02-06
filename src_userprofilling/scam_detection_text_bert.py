import torch # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
import pandas as pd # type: ignore
from transformers import BertForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)


# Load Pretrained BERT Model & Tokenizer
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 classes: Scam/Legit

# Function to Predict Scam or Legit
def predict_scam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs).item()
    return "Scam" if prediction == 1 else "Legit"

# Example Predictions
texts = [
    "Win a free iPhone! Click here: scam.com",
    "We are hiring. Apply now!",
    "Your PayPal account is restricted! Update details now."]

for txt in texts:
    print(f"'{txt}' → Prediction: {predict_scam(txt)}")

