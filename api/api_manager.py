import shutil
import time
import zlib
from flask import Flask, jsonify, request
from os.path import abspath, dirname
import os
import requests
import gzip
from langdetect import detect
from api.fastText.fasttext_manager import *
from api.constants import PATHS_2_UNSUPERVISED_MODELS, PATH_2_SUPERVISED_MODEL, LANGS
import re


"""
Для старта нужно выполнить setup.py в ~/libs/setup.py
Если запустиь API без обработки текста, поставьте PROCESS_TEXT=False
и закомментируйте строки с 22 по 36
(не успел сделать нормальное взаимодействие между клиентом и сервером)
"""
PROCESS_TEXT=True

from libs.text_normalizer.header_all_text_normalizers import *
from libs.tokenizers.header_all_tokenizers import *
from preprocessors.text_preprocessor.header_text_preprocessor import *

text_preprocessor = TextPreprocessor(
    word_tokenizer=RegexWordTokenizer(),
    word_normalizer=WordNormalizer(
        # func_word_normalizer=pymorphy_,
        stop_words=stop_words(),
        min_len_word=2,
        max_len_word=25,
    ),
    # text_normalizer=TextNormalizer(),
)


models_manager = FastTextManager()
models_manager.load_models(PATH_2_SUPERVISED_MODEL, PATHS_2_UNSUPERVISED_MODELS)


app = Flask(__name__)


def parse_train_data(data):
    parsed_data = []
    data = data.split()
    new_offer = ''
    for token in data:
        if re.search('__label__', token):
            if new_offer:
                parsed_data.append(new_offer)
            new_offer = token
        else:
            if PROCESS_TEXT:
                token = text_preprocessor.preprocess_text(token)
                if token:
                    new_offer += ' ' + token[0]
            else:
                new_offer += token
    return parsed_data


def retrain_model(atrs):
    global models_manager
    name_train = 'train.txt'

    if atrs['input']:
        data = atrs['input']
        with open(name_train, 'w') as f:
            for text in parse_train_data(data):
                f.write(text+'\n')

    retrained_model = fasttext.train_supervised(
        input=name_train,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=50,
        loss='ova',
    )

    retrained_model.quantize()

    models_manager.set_model(SUPERVISED, SUPERVISED, retrained_model)
    models_manager.get_model(SUPERVISED, SUPERVISED).save_model(PATH_2_SUPERVISED_MODEL)
    os.remove(name_train)


def download_model(url):
    global models_manager
    try:
        name = url.split('/')[-1].lower()
        bin_file = dirname(abspath(__file__)) + "/models/unsupervised/" + name[:-3]
        if len(name) >= 100 or re.findall('[a-z]+', name)[1] not in LANGS:
            return f"Incorrect code {re.findall('[a-z]+', name)[1]}"


        if bin_file in PATHS_2_UNSUPERVISED_MODELS and bin_file in models_manager.get_facebook_models:
            return f"Model {name} downloaded"


        elif bin_file in PATHS_2_UNSUPERVISED_MODELS:
            return f"Model {name} is downloading"


        PATHS_2_UNSUPERVISED_MODELS.append(bin_file)
        with requests.get(url, stream=True) as r:
            d = zlib.decompressobj(zlib.MAX_WBITS | 32)
            with open(bin_file, 'wb') as fname:
                for chunk in r.iter_content(1024**2):
                    buffer = d.decompress(chunk)
                    fname.write(buffer)
                buffer = d.flush()
                fname.write(buffer)

        models_manager.load_models(paths_unsupervised_models=bin_file)

        return f"Start download model {name}"
    except:
        return f"Error download file"


"""
Example (download_fasttext_model):
http://127.0.0.1:5000/download/en
"""


@app.route('/download/<lang>')
def download_fasttext_model(lang):

    result = download_model(f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz")

    return result


"""
Example (downloaded_fasttext_models):
http://127.0.0.1:5000/downloaded
"""


@app.route('/downloaded')
def downloaded_fasttext_models():
    return jsonify(models_manager.get_facebook_models)


"""
Example (retrain_classifier):
http://127.0.0.1:5000/retrain_classifier?input=__label__sport привет, завтра будет турнир по футболу __label__sport давай играть в догонялки __label__sport будем бегать на перегонки __label__esports пошли играть в компьютер? __label__esports завтра будет турнир по доте!!! __label__esports хочу скачать вот эту игру на компьютер __label__esports играем в морской бой онлайн __label__sport займуська я плаванием __label__sport буду прыгать через скакалку __label__sport вчера был чемпионат мира по волейболу!!! __label__esports игра, в которую можно играть с друзьями по сети __label__esports буду тренироваться играть за пятёрку в доте __label__esports в какие игры любишь играть на пк? __label__sport я сегодня поборол тренера! __label__sport завтра буду качать пресс __label__sport скоро зима, будем кататься на лыжах! __label__sport и так, делаем наклоны туловища вперед и назад __label__esports надо обновить пк, чтобы играть в такие игры __label__esports настало время поиграть в эту видеоигру __label__esports мам, мне надо играть, чтобы стать киберспорстменом!
"""


@app.route('/retrain_classifier', methods=["GET"])
def retrain_classifier():

    try:
        data = dict(input=request.args.get('input'))
        retrain_model(data)

        return "Model retrained"
    except:
        return "Failed"


"""
Example (classify):
http://127.0.0.1:5000/classify?input=хотелось бы поиграть в онлайн игры((
http://127.0.0.1:5000/classify?input=слушай, займись спортом, побегай по утрам
"""


@app.route('/classify', methods=["GET"])
def classify():

    try:
        data = request.args.get('input')
        data = ' '.join(text_preprocessor.preprocess_text(data))
        result = models_manager.get_model(SUPERVISED, SUPERVISED).predict(data)

        return str(result)
    except:
        return "Error classify text"


"""
for this function to work correctly, download cc.ru.300.bin
http://127.0.0.1:5000/download/ru


Example (get_word_vector):
http://127.0.0.1:5000/get_word_vector?input=мама папа дедушка и сын
"""


@app.route('/word_vectors', methods=["GET"])
def get_word_vector():

    try:
        data = request.args.get('input')
    except:
        return "incorrect input"

    try:
        lang = LANGS[detect(data)]
        model = models_manager.get_model(UNSUPERVISED, f"cc.{lang}.300.bin")
        result = []
        for word in data.split():
            result.append([str(vector) for vector in model.get_word_vector(word)])
        return jsonify(result)
    except:
        return "Error vectorize text"


@app.route('/')
def index():
    return "API FastText"

if __name__ == '__main__':
    app.run(
        debug=True,
        use_reloader=False
    )