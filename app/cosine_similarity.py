import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Load spaCy model with GloVe vectors
#!python -m spacy download en_core_web_lg #run this for every new session
nlp = spacy.load("en_core_web_lg")
df = pd.read_csv('./semeval_cleaned.csv')
# Define aspects and their corresponding words
aspects = {
    'food': 'food delicious tasty delicious flavor meal cuisine chef portion seasoned authentic',
    'service': 'service staff friendly helpful attentive prompt rude',
    'ambience': 'ambience ambiance atmosphere decor decoration cozy elegant modern chic view clean dirty dusty',
    'price': 'price value affordable expensive budget cost cheap',
    # 'anecdotesmiscellaneous': 'location parking cleanliness presentation variety'

}

import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# aspects = {
#     'food': 'food drinks',
#     'service': 'service staff',
#     'ambience': 'ambience music location',
#     'price': 'price money'
# }

# Define a function to compute aspect embeddings
def compute_aspect_embeddings(aspects, nlp):
    aspect_embeddings = {}
    for aspect, words in aspects.items():
        # Tokenize and average the word vectors for each aspect
        tokens = nlp(words)
        aspect_vectors = [token.vector for token in tokens if token.has_vector]
        aspect_embeddings[aspect] = np.mean(aspect_vectors, axis=0)
    return aspect_embeddings

# Define a function to compute review embeddings
def compute_review_embedding(review, nlp):
    # Process the review using spaCy pipeline
    doc = nlp(review)
    sentence_embeddings = []
    for sent in doc.sents:
        # Compute sentence embedding by averaging word vectors
        sent_vector = np.mean([word.vector for word in sent if word.has_vector], axis=0)
        sentence_embeddings.append(sent_vector)
    # Replace new words with zeros and compute review embedding
    review_embedding = np.mean(sentence_embeddings, axis=0)
    return np.nan_to_num(review_embedding)


def compute_sentence_embedding(sentence, nlp):
    # Process the sentence using spaCy pipeline
    doc = nlp(sentence)
    # Compute sentence embedding by averaging word vectors
    sent_vector = np.mean([word.vector for word in doc if word.has_vector], axis=0)
    # Replace NaN values with zeros
    return np.nan_to_num(sent_vector)


# Compute semantic similarity between review sentences and aspect embeddings
def compute_semantic_similarity(review_sentences, aspect_embeddings):
    aspect_names = list(aspect_embeddings.keys())
    similarity_scores = []
    for sentence in review_sentences:
        sentence_vector = compute_review_embedding(sentence, nlp)
        sentence_vector = sentence_vector.reshape(1, -1)  # Reshape to match cosine_similarity function input
        sentence_similarity = []
        for aspect_name, aspect_vector in aspect_embeddings.items():
            aspect_vector = aspect_vector.reshape(1, -1)  # Reshape to match cosine_similarity function input
            similarity = cosine_similarity(sentence_vector, aspect_vector)[0][0]
            sentence_similarity.append(similarity)
        similarity_scores.append({aspect_name: score for aspect_name, score in zip(aspect_names, sentence_similarity)})
    return similarity_scores


review_sentences = df['input_text']
aspect_embeddings = compute_aspect_embeddings(aspects, nlp)
def predict_aspect(sentence):
    p = r'[^\w\s]'
    sentence = re.sub(p, '', sentence)
    for token in sentence.lower().split():
        print(token)
    tokens = [token.lower() for token in sentence.split() if token.lower() not in stop_words]
    new_sentence = ' '.join(tokens)
    scores = compute_semantic_similarity([new_sentence], aspect_embeddings)[0]
    return max(scores, key = lambda x: scores[x])