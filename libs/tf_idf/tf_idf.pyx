from libc.math cimport log10


cpdef float Tf(object term, list text):
    return text.count(term) / len(text)


cdef class Idf:


    cdef dict idf
    cdef int min_count, corpus_len


    def __init__(self, int min_count=20):
        self.idf = dict()
        self.min_count = min_count
        self.corpus_len = 0


    cpdef float __calc(self, int count):
        if count >= self.min_count:
            return log10(self.corpus_len / count)
        return 0


    cpdef void add_one_corpus(self, list corpus):
        cdef object token
        cdef dict tmp = {i:0 for i in corpus}
        self.corpus_len+=1
        for token in tmp:
            try:
                self.idf[token] += 1
            except:
                self.idf[token] = 1


    cpdef void add_all_corpus(self, list data):
        cdef object corpus
        for corpus in data:
            self.add_one_corpus(corpus)


    def get_vocab(self):
        return self.idf


    def get_idf(self):
        return {item:self.__calc(self.idf[item]) for item in self.idf}


    def __getitem__(self, item):
        if item in self.idf:
            return self.__calc(self.idf[item])
        return 0.0





cdef class Tfidf:


    cdef object idf


    def __init__(self, idf):
        self.idf = idf


    cpdef float calc(self, object term, list data):
        cdef float tf_ = Tf(term, data)
        cdef float idf_ = self.idf[term]
        return  tf_*idf_
