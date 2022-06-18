import numpy as np
from flask import Flask, request, jsonify, render_template
import spacy


nlp = spacy.load('en_core_web_md')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sentences = [x for x in request.form.values()]
    doc1 = nlp(sentences[0]) 
    doc2 = nlp(sentences[1])
    output = doc1.similarity(doc2)
    
    
    return render_template('index.html',prediction_text='The similarity score for the given sentences is : {} '.format(output))


if __name__ == "__main__":
    app.run(debug=True)