import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report


class Logistic:


    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1,1))
        # self.vectorizer = CountVectorizer(binary=False, stop_words='english', ngram_range=(1,4))

        vect_trans = make_column_transformer((self.vectorizer, "input_text"), (self.vectorizer, "aspect"))
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
        self.model = make_pipeline(vect_trans, classifier)
        self.aspect_numb = {"negative": 0, "positive": 1, "neutral": 2, "conflict": 3}


    def train(self, dataset):
        # Train logistic regression for the current dataset
        df = pd.DataFrame(dataset)
        X = df[['input_text', 'aspect']]
        Y = np.array([self.aspect_numb[i] for i in list(df.polarity)])
        self.model.fit(X, Y)



    def test(self, dataset):
        test_data = pd.DataFrame(dataset)
        # test_data['aspect_predicted'] = test_data['aspect']
        X_test = test_data[['input_text', 'aspect']]
        y_true = np.array([self.aspect_numb[i] for i in list(test_data.polarity)])
        y_pred =self.model.predict(X_test)
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral', 3: 'conflict'}

        # get classification report
        out_data = test_data.copy(deep=True)
        out_data['y_pred'] = [label_map[label] for label in y_pred]
        return out_data, classification_report(y_true, y_pred)