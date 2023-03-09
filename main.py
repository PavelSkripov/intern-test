import uvicorn
from fastapi import FastAPI
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = ['neutral', 'happiness', 'sadness', 'enthusiasm', 'fear', 'anger', 'disgust']
tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')

app = FastAPI()

@torch.no_grad()    
def predict_emotions(text: str) -> list:
    """
        It takes a string of text, tokenizes it, feeds it to the model, and returns a dictionary of emotions and their
        probabilities
        :param text: The text you want to classify
        :type text: str
        :return: A dictionary of emotions and their probabilities.
    """
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    emotions_list = {}
    for i in range(len(predicted.numpy()[0].tolist())):
        emotions_list[LABELS[i]] = predicted.numpy()[0].tolist()[i]
    return emotions_list

# Получение словаря с данными о тональности текста и преобразование его в список по формату
def list_format(text: dict):
    emotions_list = []
    e_dict = {}
    emotions_dict = predict_emotions(text["text"])
    for item in emotions_dict.items():
        e_dict["label"] = item[0]
        e_dict["score"] = item[1]
        emotions_list.append(e_dict)
    return emotions_list

# Реализация endpoint
@app.post('/detect_emotion')
def get_emotions_list(text: dict):
    detect = list_format(text)
    return detect
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
