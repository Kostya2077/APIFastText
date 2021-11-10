import fasttext
from fasttext import *


SUPERVISED = "SUPERVISED"
UNSUPERVISED = "UNSUPERVISED"
NAME = "NAME"


class FastTextManager(object):


    def __init__(self):
        self.__models={
            SUPERVISED: {SUPERVISED: None},
            UNSUPERVISED: {"TEST":'test'}
        }


    def get_model(self, type_model, name_model):
        return self.__models[type_model][name_model]


    def set_model(self, type_model, name_model, model):
        self.__models[type_model][name_model] = model



    @property
    def get_facebook_models(self):
        models = [name for name in self.__models[UNSUPERVISED]]
        return models


    @get_facebook_models.setter
    def get_facebook_models(self):
        models = [name for name in self.__models[UNSUPERVISED] if self.__models[UNSUPERVISED][name]]
        return models


    def __kill_model(self, type_model, name):
        if name in self.__models[type_model]:
            self.__models[type_model][name] = None


    def load_models(self, path_supervised_model=None, paths_unsupervised_models=None):

        try:
            self.__models[SUPERVISED][SUPERVISED] = load_model(path_supervised_model)
        except:
            pass


        if paths_unsupervised_models:
            if isinstance(paths_unsupervised_models, list):
                for path in paths_unsupervised_models:
                    try:
                        name = path.split('/')[-1]
                        self.__kill_model(UNSUPERVISED, name)
                        self.__models[UNSUPERVISED][name] = load_model(path)
                    except:
                        pass
            elif isinstance(paths_unsupervised_models, str):
                try:
                    name = paths_unsupervised_models.split('/')[-1]
                    self.__kill_model(UNSUPERVISED, name)
                    self.__models[UNSUPERVISED][name] = load_model(paths_unsupervised_models)
                except:
                    pass













