from libs.tokenizers.common.constants import TOKENIZER_PARAMS
import re

class RegexWordTokenizer:


    def __init__(self, split_symbols=TOKENIZER_PARAMS.SPLIT_SYMBOLS):
        self.tokenizer = re.compile(f"[^ {''.join([i for i in split_symbols])}]+")


    def tokenize(self, text):
        return self.tokenizer.findall(text)

