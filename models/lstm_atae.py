import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim
from gensim.models import Word2Vec

from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.layers import Input, Dense, LSTM, Bidirectional, Concatenate, Attention, Reshape
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import RMSprop

from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Dot, Activation
from keras.layers import TimeDistributed, RepeatVector, Permute, Multiply, Flatten, Reshape
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from keras.models import load_model

from sklearn.metrics import accuracy_score

class AspectBasedSentimentAnalysis:
    def __init__(self):
        self.model = None
        self.word2vec_model = None
        self.polarity_mapping = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}

    def train_word2vec(self, df):
        # List of terms
        sentence = df['input_text'].tolist()
        aspect = df['aspect'].tolist()

        # Tokenize sentences
        nltk.download('punkt')

        # Combine sentences and aspect terms
        filtered_sentence_tokens = [word_tokenize(sentence.lower()) for sentence in sentence]
        aspect_tokens = [word_tokenize(item.lower()) for item in aspect]
        all_tokens = filtered_sentence_tokens + aspect_tokens

        # Initialize the Word2Vec model
        model = Word2Vec(
            vector_size=100,  # Adjust dimensionality as needed
            window=5,
            min_count=1,
            workers=4,
            sg=0,  # Use CBOW architecture (skip-gram: sg=1)
        )

        # Build the vocabulary
        model.build_vocab(all_tokens)

        # Train the model
        model.train(all_tokens, total_examples=model.corpus_count, epochs=10)

        # Save the model
        model.save("word2vec.model")
        print("Word2Vec model trained and saved.")

    def train(self, df):
        # unique categories
        unique_categories = df['polarity'].unique()

        # List of terms
        sentence = df['input_text'].tolist()
        aspect = df['aspect'].tolist()

        # Pair sentences with their corresponding aspect terms
        sentence_aspect_pairs = list(zip(sentence, aspect))

        # Tokenize sentences
        nltk.download('punkt')

        # Combine sentences and aspect terms
        filtered_sentence_tokens = [word_tokenize(sentence.lower()) for sentence in sentence]
        aspect_tokens = [word_tokenize(item.lower()) for item in aspect]
        
        # Load Word2Vec model
        model_file = 'word2vec.model'  # Replace with your model file path
        self.word2vec_model = Word2Vec.load(model_file)

        # Convert tokenized sentences and aspect terms to embeddings
        sentence_embeddings = np.array([np.mean([self.word2vec_model.wv[token] for token in sentence], axis=0) for sentence in filtered_sentence_tokens])
        category_embeddings = np.array([np.mean([self.word2vec_model.wv[token] for token in aspect], axis=0) for aspect in aspect_tokens])

        # Assuming you have a DataFrame 'df' with a column 'polarity'
        df['polarity_numeric'] = df['polarity'].map(self.polarity_mapping)

        # Prepare labels
        labels = df['polarity_numeric'].values

        # Define model architecture
        sentence_input_layer = Input(shape=(None, sentence_embeddings.shape[1],))  # Input shape based on sentence embeddings
        category_input_layer = Input(shape=(category_embeddings.shape[1],))  # Input shape based on category embeddings

        # Add a time dimension to the sentence and category embeddings
        sentence_input_layer_reshaped = Reshape((1, sentence_embeddings.shape[1]))(sentence_input_layer)
        category_input_layer_reshaped = Reshape((1, category_embeddings.shape[1]))(category_input_layer)

        # LSTM layer
        lstm_layer = LSTM(128, return_sequences=True)(sentence_input_layer_reshaped)

        # Transform category_embeddings to have the same dimension as lstm_layer
        transform_layer = Dense(128, activation='relu')  # 128 to match the dimension of lstm_layer
        transformed_category_embeddings = transform_layer(category_input_layer_reshaped)

        # Attention mechanism
        attention = Dot(axes=-1)([lstm_layer, transformed_category_embeddings])
        attention = Activation('softmax')(attention)
        attention = Permute((2, 1))(attention)
        representation = Multiply()([lstm_layer, attention])
        representation = Lambda(lambda x: K.sum(x, axis=-2))(representation)

        # Output layer
        output_layer = Dense(4, activation='softmax')(representation)  # 4 classes (positive, neutral, negative, conflict)

        # Split data into train, validation, and test sets
        X_train_sentence, X_temp_sentence, y_train, y_temp = train_test_split(sentence_embeddings, labels, test_size=0.2, random_state=42)
        X_val_sentence, X_test_sentence, y_val, y_test = train_test_split(X_temp_sentence, y_temp, test_size=0.5, random_state=42)

        X_train_category, X_temp_category = train_test_split(category_embeddings, test_size=0.2, random_state=42)
        X_val_category, X_test_category = train_test_split(X_temp_category, test_size=0.5, random_state=42)

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes=4)
        y_val = to_categorical(y_val, num_classes=4)

        self.model = Model(inputs=[sentence_input_layer, category_input_layer], outputs=output_layer)

        # Compile and train the model
        optimizer = RMSprop(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit([X_train_sentence, X_train_category], y_train, epochs=20, validation_data=([X_val_sentence, X_val_category], y_val))

        # Save the model to a file
        self.model.save('my_model.h5')

    def test(self, test_dataset):

        # Load the model from the file
        self.model = load_model('my_model.h5')

        # Convert labels to one-hot encoding
        test_dataset['polarity_numeric'] = test_dataset['polarity'].map(self.polarity_mapping)
        y_test = to_categorical(test_dataset['polarity_numeric'].values, num_classes=4)

        # Convert tokenized sentences and aspect terms to embeddings
        sentence_embeddings = np.array([np.mean([self.word2vec_model.wv[token] for token in word_tokenize(sentence.lower())], axis=0) for sentence in test_dataset['input_text']])
        category_embeddings = np.array([np.mean([self.word2vec_model.wv[token] for token in word_tokenize(item.lower())], axis=0) for item in test_dataset['aspect']])

        # Evaluate the model on the test data
        loss, accuracy = self.model.evaluate([sentence_embeddings, category_embeddings], y_test)
        
        # Predict the labels
        y_pred = self.model.predict([sentence_embeddings, category_embeddings])

        # Convert the predicted labels back to their original form
        y_pred = np.argmax(y_pred, axis=1)
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral', 3: 'conflict'}
        y_pred = [label_map[label] for label in y_pred]
        
        # Add the predicted labels to the DataFrame
        test_dataset['predicted_polarity'] = y_pred
        
        return test_dataset, loss, accuracy