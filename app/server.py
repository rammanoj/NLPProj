import http.server
import socketserver
import json
import cosine_similarity
from ngram_nb import Ngram_NB
# from bert_uncased import BERTUncased
from lstm_atae import AspectBasedSentimentAnalysis
import pandas as pd
import torch

PORT = 8081

from transformers import BertForSequenceClassification, BertTokenizer

# Initialize the BERT model architecture
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Load the saved model state
# model.load_state_dict(torch.load("bert_uncase.pth"))

# model.eval()

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#other models
ngram_model = Ngram_NB(1)
# bert_model = BERTUncased(5)
lstm_model = AspectBasedSentimentAnalysis()
df = pd.read_csv("semeval_cleaned.csv")
df = df[df['aspect'] != 'anecdotesmiscellaneous']
print("Training models...")
ngram_model.train(df, text_col="input_text", category_col="aspect")
# bert_model.train(df)
lstm_model.train(df)
print("Models trained!")

label_map = {0: 'negative', 1: 'positive', 2: 'neutral', 3: 'conflict'}

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8000')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = post_data.decode('utf-8')
        predicted_aspect = cosine_similarity.predict_aspect(post_data)
        ngram_sentiment = ngram_model.score(post_data, predicted_aspect)
        input_df = pd.DataFrame({'input_text':post_data, 'aspect': predicted_aspect, 'polarity': "positive", 'index': [0]})
        lstm_sentiment, _, _ = lstm_model.test(input_df)
        lstm_sentiment = list(lstm_sentiment["predicted_polarity"])[0]
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8000')
        self.end_headers()
        response_data = {'message': 'Received POST request', 'data': {"aspect": predicted_aspect, 
                                                                      "ngram_sentiment": ngram_sentiment, 
                                                                      "lstm_sentiment": lstm_sentiment}}
                                                                #       "bert_sentiment": label_map[predicted_label]}} 
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

Handler = MyHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Server started at port", PORT)
    httpd.serve_forever()
