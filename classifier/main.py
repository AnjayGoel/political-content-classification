import html
import os
import re

import joblib
import numpy as np
import pandas as pd
from gensim.summarization import keywords
from sklearn.linear_model import LogisticRegression as lr


class Classifier:
    punc = '''!()[]|{};":\<>/@#$%^&*_~'''
    relp = " " * len(punc)

    def __init__(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(this_dir, "data", "extracted_keyword_freq.csv"))
        df["rel_freq"] = (df["pol_freq"] * df["total_freq"].sum()) / (df["total_freq"] * df["pol_freq"].sum()) - 1
        self.word_freq_dict: dict = dict(zip(df["word"], df["total_freq"]))
        self.rel_freq_dict: dict = dict(zip(df["word"], df["rel_freq"]))
        self.model: lr = joblib.load(os.path.join(this_dir, "models", "lr_train_50_pos.sav"))

        # print(self.model.coef_, self.model.intercept_)

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

    def get_most_common(self, compound_word):  # get most frequent word from a phrase.
        if len(compound_word) < 2:
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

    def split_words(self, word_list):
        ret_list = []
        for i in word_list:
            ret_list.append(self.get_most_common(i))
        return ret_list

    def get_x_from_word_list(self, word_list):
        ret_val = []

        for word in word_list:
            ret_val.append(self.get_rel_freq(word))

        ret_val = np.sort(np.array(ret_val))[::-1]
        ret_val = np.pad(ret_val, (0, max(0, 10 - len(ret_val))), 'constant')  # Padding to 10 items
        return ret_val

    # Extract keywords using textrank implementation in gensim
    @staticmethod
    def keywords_tr(text):
        text_keys = keywords(text, ratio=1, lemmatize=True, scores=True)
        text_keys = [tup[0] for tup in text_keys[:10]]
        return text_keys

    def predict(self, text):
        text_keys = self.keywords_tr(text)
        text_keys = self.split_words(text_keys)[:10]
        if len(text_keys) < 2:
            raise ("Text not long enough", "SHORT_TEXT")
        else:
            pass
            # print(text_keys)
        data = self.get_x_from_word_list(text_keys)

        res = self.model.predict([data])[0]
        # print(res)
        return res


if __name__ == "__main__":
    test_str = """Smile and Wave boys, just smile and wave"""
    test_str_2 = """Brazil vaccine scandal imperils Jair Bolsonaro as protests spread"""
    model = Classifier()
    print(model.predict(test_str))
    print(model.predict(test_str_2))
