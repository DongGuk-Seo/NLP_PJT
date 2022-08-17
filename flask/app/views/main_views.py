import numpy as np
from flask import Blueprint, redirect, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Mecab
import torch
from torch.utils.data import Dataset, DataLoader


mecab = Mecab()
xtrain = list(np.load('../xtrain_add.txt.npy',allow_pickle=True))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xtrain)
threshold = 3
total_cnt = len(tokenizer.word_index) 
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
tokenizer = Tokenizer(total_cnt - rare_cnt + 1)
tokenizer.fit_on_texts(xtrain)
X_train = tokenizer.texts_to_sequences(xtrain)

loaded_model = load_model('../model/mecab_epoch-15_bat-64_th-3_acc-8556_add.h5')

def pr(new_sentence):
    new_sentence = mecab.morphs(new_sentence) # 토큰화
    new_sentence = [word for word in new_sentence] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 30) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    return score

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def main():
    return render_template('index.html')

@bp.route('/',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        word = request.form['query']
        result = pr(word)
        print(result)
        return render_template('index.html',word=word, result=result)
    else:
        return render_template('index.html')
# @bp.route('/',methods = ['POST', 'GET'])
# def result():
#     if request.method == 'POST':
#         word = request.form['query']
#         result = pr(word)
#         print(result)
#         return redirect(f'{url_for('result',word=word, result=result)}#query')
#     else:
#         return render_template('index.html')
