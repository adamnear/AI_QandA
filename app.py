from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

@app.route("/", methods=["GET", "POST"])
