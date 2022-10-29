# -*- coding: utf-8 -*-

import os

from flask import Flask, request, render_template, send_from_directory
from datetime import datetime
from main import infer_by_web

import json
import requests

API_TOKEN = "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"

__author__ = 'Hardik'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # project abs path

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload_page", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)

    option = request.form.get('optionsPrediction')
    context = request.form.get('context')
    question = request.form.get('question')

    answer = NLP_predict(context, question)

    for upload in request.files.getlist("file"):
        filename = upload.filename

        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")

        filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')#upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)
        result, probability = predict_image(destination, option)

        f1_score = compute_f1(result, answer)

    print("Send File Name to html: ", filename)
    return render_template("complete.html", image_name=filename, result=[result,probability, context, question, answer, f1_score]) 


def predict_image(path, type):
    print(path)
    return infer_by_web(path, type)

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

# Compute Exact Match Score (em score)
def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

#Compute F1 score
def compute_f1(prediction, truth):
  # Normalizing predicted and true answer
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()
  
  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)
  
  # Intersection of Predicted and true answer
  common_tokens = set(pred_tokens) & set(truth_tokens)
  
  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0
  
  # Ratio of length of common tokens to complete tokens length
  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)
  
  # Return Harmonic Mean
  return 2 * (prec * rec) / (prec + rec)

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/hd94/roberta-hindi"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    resp = json.loads(response.content.decode("utf-8"))

    if response.status_code != 200:
        return "Error", response.status_code

    return resp["answer"], response.status_code


def NLP_predict(context, question):
  resp = query({"inputs": {"question": question,"context": context}})
  
  while resp[0] == 'Error':
    resp = query({"inputs": {"question": question,"context": context}})

  return resp[0]

if __name__ == "__main__":
    app.run(port=5555, debug=True)

