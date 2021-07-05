import html
import logging
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from gensim.summarization import keywords
from sklearn.linear_model import LogisticRegression as lr

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore", module="sklearn")


class Classifier:
    punc = '''!()[]|{};":\<>/@#$%^&*_~'''
    relp = " " * len(punc)

    def __init__(self, model_type="10"):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(this_dir, "data", "extracted_keyword_freq.csv"))
        df["rel_freq"] = (df["pol_freq"] * df["total_freq"].sum()) / (df["total_freq"] * df["pol_freq"].sum()) - 1
        self.word_freq_dict: dict = dict(zip(df["word"], df["total_freq"]))
        self.rel_freq_dict: dict = dict(zip(df["word"], df["rel_freq"]))
        self.model: lr = joblib.load(os.path.join(this_dir, "models", f"lr_train_{model_type}_pos.sav"))
        logging.info(f"Model Params: {self.model.coef_}, {self.model.intercept_}")

    @staticmethod
    def preprocess(s):
        """Custom function to preprocess text, punctuations etc."""
        result = re.sub(r"’", "'", str(s))
        result = re.sub(r"\n", ' ', str(result))
        result = re.sub(r"r/", ' ', str(result))
        result = result.encode("ascii", "ignore").decode()
        result = re.sub(r"http\S+", ' ', str(result), flags=re.MULTILINE)
        result = html.unescape(result)
        result = result.translate(str.maketrans(Classifier.punc, Classifier.relp))
        result = re.sub(' +', ' ', str(result))
        result = result.strip()
        return result

    def resolve_compound_word(self, compound_word):
        """return  most frequent word from a compound word."""
        if len(compound_word) < 2 or " " not in compound_word:
            return compound_word

        word_freq = 0
        word_ret = ""
        for word in compound_word.split():
            if word in ["null", "nan"]:
                continue
            elif self.word_freq_dict.get(word, 0) > word_freq:
                word_ret = word
                word_freq = self.word_freq_dict[word]
        return word_ret

    def get_rel_freq(self, word):
        return self.rel_freq_dict.get(word, 0)

    def make_final_word_list(self, word_list):
        return [self.resolve_compound_word(word) for word in word_list]

    def get_x_from_word_list(self, word_list):
        ret_val = [self.get_rel_freq(word) for word in word_list]
        ret_val = np.sort(np.array(ret_val))[::-1]
        ret_val = np.pad(ret_val, (0, max(0, 10 - len(ret_val))), 'constant')  # Padding to 10 items
        return ret_val

    # Extract keywords using textrank implementation in gensim
    @staticmethod
    def get_keywords_textrank(text):
        text_keys = keywords(text, ratio=1, lemmatize=True, scores=True, split=True, pos_filter=())
        text_keys = [tup[0] for tup in text_keys[:10]]
        return text_keys

    def predict(self, text):
        text_keys = self.get_keywords_textrank(self.preprocess(text))
        text_keys = self.make_final_word_list(text_keys)[:10]

        if len(text_keys) < 2:
            raise Exception("Text not long enough", "SHORT_TEXT")
        else:
            logging.info(f"Keywords: {text_keys}")
        data = self.get_x_from_word_list(text_keys)
        logging.info(f"keywords rel freq: {data}")
        res = self.model.predict([data])[0]
        return res


if __name__ == "__main__":
    test_str = """
    Crossing fingers was commonly used as a sort of protective charm for warding off evil. This 
    explains why people wishing you luck would cross their fingers, as well as why those breaking a promise would do 
    it 
    """
    test_str_2 = """
    July 4, 2021, will mark the 245th anniversary of the founding of the United States. Independence Day -- known 
    colloquially as the Fourth of July -- is celebrated in the United States on July 4 annually. 
    """

    model = Classifier()
    print(model.predict(test_str))
    print(model.predict(test_str_2))
