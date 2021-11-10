from libs.text_normalizer.common.constants import TOKEN_CONST
from libs.tf_idf.header_tf_idf import *
import re


cdef class WordNormalizer:

    cdef object normalizer, stop_words
    cdef dict vocabulary, true_symbols
    cdef bint active_vocabulary
    cdef int max_len_word, min_len_word

    def __init__(self,
                 true_symbols=TOKEN_CONST.TRUE_SYMBOLS,
                 func_word_normalizer=None,
                 stop_words=[],
                 active_vocabulary=True,
                 min_len_word=2,
                 max_len_word=25):
        self.true_symbols = {i:0 for i in true_symbols}
        self.normalizer = func_word_normalizer
        self.stop_words = stop_words
        self.vocabulary = {}
        self.active_vocabulary = active_vocabulary
        self.max_len_word=max_len_word
        self.min_len_word=min_len_word


    cpdef str normalize(self, str word):
        cdef str symbol, result=word.lower()

        if self.active_vocabulary and word in self.vocabulary:
            return self.vocabulary[word]
        elif not (self.min_len_word <= len(result) <= self.max_len_word):
            return ''

        if self.true_symbols:
            for symbol in word:
                if symbol not in self.true_symbols:
                    result = result.replace(symbol, '')
        if self.normalizer:
            result = self.normalizer(result)
        if self.stop_words:
            if result in self.stop_words:
                result = ''
        if self.active_vocabulary:
            self.vocabulary[word] = result
        return result



cdef class TextNormalizer:

    cdef int min_count
    cdef list frange
    cdef object idf

    def __init__(self, min_count=1, frange=[10e-10, 10e10]):
        self.min_count = min_count
        self.frange = frange
        self.idf = Idf(min_count)

    cpdef void _add_one_corpus(self, list text):
        self.idf.add_one_corpus(text)

    cpdef void _add_full_curpus(self, list data):
        self.idf.add_all_corpus(data)



    cpdef list normalize(self, list text):
        cdef list result = []
        cdef float res
        cdef object token, tfidf = Tfidf(self.idf)
        for token in text:
            res = tfidf.calc(token, text)
            if self.frange[0] <= res <= self.frange[1]:
                result.append(token)
        return result