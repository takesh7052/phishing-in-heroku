import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import creme
from creme import compose
from creme import linear_model
from creme import metrics
from creme import preprocessing
from creme import datasets

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        f = request.form['csvfile']
   
    metric = metrics.Accuracy()
    data = datasets.Phishing()

    model = compose.Pipeline(
     preprocessing.StandardScaler(),
     linear_model.LogisticRegression()
    )
   # output=[]

    for A, b in data:
        pred = model.predict_one(A) 
        metric = metric.update(b, pred)
        render_template('output.html', accuracy=metric)
        model = model.fit_one(A, b)


   # output = round(prediction[0], 2)

    return render_template('output.html', accuracy=metric)


if __name__ == "__main__":
    app.run(debug=True)
