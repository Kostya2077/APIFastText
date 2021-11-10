import nltk
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from tqdm import tqdm


_PYMORPHY2 = MorphAnalyzer().parse


def pymorphy_(word):
    return _PYMORPHY2(word)[0].normal_form

def stop_words(lang="russian"):
    try:
        return {word:0 for word in stopwords.words(lang)}
    except:
        nltk.download('stopwords')
        return {word:0 for word in stopwords.words(lang)}


class TextPreprocessor:


    def __init__(self,
                 word_tokenizer=None,
                 word_normalizer=None,
                 text_normalizer=None,
                 ):
        self.word_normalizer = word_normalizer
        self.word_tokenizer = word_tokenizer
        self.text_normalizer = text_normalizer



    def __word_normalize(self, text):
        result = []
        for index in range(len(text)):
            norm_word = self.word_normalizer.normalize(text[index])
            if norm_word != '':
                result.append(self.word_normalizer.normalize(text[index]))
        return result


    def __text_normalize(self, text):
        return self.text_normalizer.normalize(text)


    def preprocess_text(self, text):


        if self.word_tokenizer != None:
            text = self.word_tokenizer.tokenize(text)
        else:
            text = text.split()


        if self.word_normalizer:
            text = self.__word_normalize(text)


        if self.text_normalizer != None:
            self.text_normalizer._add_one_corpus(text)


        return text


    def preprocess_data(self, data):

        for index in tqdm(range(len(data))):
            data[index] = self.preprocess_text(data[index])

        if self.text_normalizer:
            for index in tqdm(range(len(data))):
                data[index] = self.__text_normalize(data[index])


        return data
