from fastapi import FastAPI, HTTPException
from transformers import pipeline

app = FastAPI()

qa_pipeline = pipeline("Question-answering", model="deepset/roberta-base-squad2")

@app.post("/answer")