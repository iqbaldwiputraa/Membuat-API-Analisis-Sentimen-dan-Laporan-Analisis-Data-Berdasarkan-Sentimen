import re
import pandas as pd
import sqlite3
from flask import Flask, jsonify, request, render_template, redirect, url_for
import pickle
import numpy as np
import re
from string import punctuation
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import json

app = Flask(__name__, template_folder='templates')

#CNN
# Memuat tokenizer dari file
tokenizer_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\CNN\pickle\tokenizer.pkl'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
# Memuat onehot encoder dari file
onehot_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\CNN\pickle\onehot.pkl'
onehot = pickle.load(open(onehot_path, 'rb'))
# Memuat model CNN dari file
cnn_model_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\CNN\h5\model_cnn.h5'
load_cnn = keras.models.load_model(cnn_model_path)

#load akurasi
json_file_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\CNN\json\report_cnn.json'
with open(json_file_path, 'r') as json_file:
    cnn_report = json.load(json_file)

#LSTM
# Memuat tokenizer dari file
tokenizer_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\LSTM\pickle\tokenizer.pkl'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
# Memuat onehot encoder dari file
onehot_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\LSTM\pickle\onehot.pkl'
onehot = pickle.load(open(onehot_path, 'rb'))
# Memuat model CNN dari file
lstm_model_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\LSTM\h5\model_lstm.h5'
load_lstm = keras.models.load_model(lstm_model_path)

#load akurasi
json_file_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\LSTM\json\report_lstm.json'
with open(json_file_path, 'r') as json_file:
    lstm_report = json.load(json_file)

#NN
# Memuat tokenizer dari file
tokenizer_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\NN\pickle\tokenizer.pickle'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

x_pad_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\NN\pickle\x_pad_sequences.pickle'
x_pad = pickle.load(open(onehot_path, 'rb'))

y_labels_path = r'C:\Users\iqbal\OneDrive\Documents\BOOTCAMP\CHALLANGE PLATINUM\NN\pickle\y_labels.pickle'
y_labels = pickle.load(open(onehot_path, 'rb'))

#### Mempersiapkan Dataset

dataset = pd.read_csv("C:/Users/iqbal/OneDrive/Documents/BOOTCAMP/CHALLANGE PLATINUM/train_preprocess.tsv.txt",encoding="latin1",sep='\t',header=None,names=["text","label"])
dataset

#### Feature Extraction

def lowercasing(paragraph):

	return paragraph.lower()
	
def menghilangkan_tandabaca(paragraph):
	new_paragraph = re.sub(fr'[{punctuation}]', r'', paragraph)
	return new_paragraph

def text_normalization(paragraph):
	paragraph = lowercasing(paragraph)
	paragraph = re.sub(r"[ ]+",r' ',paragraph)
	return paragraph


      # <option value="Logistic Regression">Logistic Regression</option>
      # <option value="CNN">CNN</option>
      # <option value="NLP">NLP</option>
      # <option value="LSTM">LSTM</option>


#route untul logistic
@app.route('/', methods=['GET', "POST"])
def page_utama():
    if request.method == 'POST':
        opsi=request.form["pilihan"]
        if opsi=="Logistic Regression":
            return redirect(url_for("logistic"))
        elif opsi=="CNN":
            return redirect(url_for("CNN"))
        elif opsi=="LSTM":
            return redirect(url_for("LSTM"))
        elif opsi=="NLP":
            return redirect(url_for("NLP"))
    else:
        return render_template("page_utama.html")

#route untul logistic
@app.route('/logistic',methods=["GET","POST"])
def logistic():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("logistic_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/logistic_prediksi',methods=["GET","POST"])
def logistic_prediksi():
    if request.method=="POST":
        text=request.form["inputText"]
        text=text_normalization(text)
        text=pd.DataFrame({"text":[text]})
        text_tf=tokenizer.texts_to_sequences(text["text"])
        text_pad=pad_sequences(text_tf,padding="post",maxlen=91)
        prediksi=load_cnn(text_pad)
        hasil=onehot.inverse_transform(prediksi).tolist()[0][0]
        probabilitas=np.max(prediksi,axis=1).tolist()[0]

        json_response=jsonify({"prediksi":hasil,"probabilitas":probabilitas})
        return json_response
    else:
        return render_template("prediksi.html")

#route untuk CNN
@app.route('/CNN',methods=["GET","POST"])
def CNN():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return cnn_report
        elif opsi =="prediksi":
            return redirect(url_for("CNN_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/CNN_prediksi',methods=["GET","POST"])
def CNN_prediksi():
    if request.method=="POST":
        text=request.form["inputText"]
        text=text_normalization(text)
        text=pd.DataFrame({"text":[text]})
        text_tf=tokenizer.texts_to_sequences(text["text"])
        text_pad=pad_sequences(text_tf,padding="post",maxlen=91)
        prediksi=load_cnn(text_pad)
        hasil=onehot.inverse_transform(prediksi).tolist()[0][0]
        probabilitas=np.max(prediksi,axis=1).tolist()[0]

        json_response=jsonify({"prediksi":hasil,"probabilitas":probabilitas})
        return json_response
    else:
        return render_template("prediksi.html")

#route untuk LSTM
@app.route('/LSTM',methods=["GET","POST"])
def LSTM():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("LSTM_prediksi"))
    else:
        return render_template("page_kedua.html")


@app.route('/LSTM_prediksi',methods=["GET","POST"])
def LSTM_prediksi():
    if request.method=="POST":
        text=request.form["inputText"]
        text=text_normalization(text)
        text=pd.DataFrame({"text":[text]})
        text_tf=tokenizer.texts_to_sequences(text["text"])
        text_pad=pad_sequences(text_tf,padding="post",maxlen=91)
        prediksi=load_cnn(text_pad)
        hasil=onehot.inverse_transform(prediksi).tolist()[0][0]
        probabilitas=np.max(prediksi,axis=1).tolist()[0]

        json_response=jsonify({"prediksi":hasil,"probabilitas":probabilitas})
        return json_response
    else:
        return render_template("prediksi.html")


#route untuk NLP
@app.route('/NLP',methods=["GET","POST"])
def NLP():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("NLP_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/NLP_prediksi',methods=["GET","POST"])
def NLP_prediksi():
    if request.method=="POST":
        text=request.form["inputText"]
        text=text_normalization(text)
        text=pd.DataFrame({"text":[text]})
        text_tf=tokenizer.texts_to_sequences(text["text"])
        text_pad=pad_sequences(text_tf,padding="post",maxlen=91)
        prediksi=load_cnn(text_pad)
        hasil=onehot.inverse_transform(prediksi).tolist()[0][0]
        probabilitas=np.max(prediksi,axis=1).tolist()[0]

        json_response=jsonify({"prediksi":hasil,"probabilitas":probabilitas})
        return json_response
    else:
        return render_template("prediksi.html")

if __name__ == '__main__':
    app.run(debug=True)

 
