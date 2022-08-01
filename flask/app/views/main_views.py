import numpy as np
from flask import Blueprint, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Mecab

mecab = Mecab()
xtrain = list(np.load('../xtrain.txt.npy',allow_pickle=True))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xtrain)
loaded_model = load_model('../model/mecab_epoch-10_bat-64_maskoff_stoff_leoff_loss-4005_acc-8237.h5')
tokenizer = Tokenizer(23134)
tokenizer.fit_on_texts(xtrain)
X_train = tokenizer.texts_to_sequences(xtrain)


def pr(new_sentence):
    # 입력된 데이터 전처리
    new_sentence = mecab.morphs(new_sentence)# # 토큰화
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
        return render_template('index.html',word=word, result=result)
    else:
        return render_template('index.html')
