from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = "news_classifier"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Load full pretrained BERT model with classification head
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# Save locally
tokenizer.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)

print(" CPU-friendly BERT model ready in 'news_classifier'")
