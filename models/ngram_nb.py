import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from collections import Counter, defaultdict

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>" 

def tokenize_line(line: str, ngram: int,
                  by_char: bool = False,
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
    for p in string.punctuation:
        line = line.replace(p, '')
        
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        stop_words = set(stopwords.words('english'))
        # otherwise split on white space and ignore stopwords
        inner_pieces = [w.lower() for w in line.split() if w.lower() not in stop_words]

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens

def tokenize(data: pd.DataFrame, col: str, ngram: int,
             by_char: bool = False,
             sentence_begin: str = SENTENCE_BEGIN,
             sentence_end: str = SENTENCE_END):
    """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
    total = []
    # also glue on sentence begin and end items
    for line in data[col]:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total

def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
    ngrams = []
    for i in range(len(tokens) + 1 - n):
        ngrams.append(tuple(tokens[i:i + n]))
    return ngrams

# determines if token is special
def is_special(token):
    return token == SENTENCE_BEGIN or token == SENTENCE_END

class Ngram_NB:
    def __init__(self, n_gram):
        """Initializes an untrained n-gram Naive Bayes Model
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        self.n = n_gram
        self.category_probabilities = None
        self.ngrams = None
        self.raw_vocab = None
        
    def train(self, df, text_col="text", category_col="category", label_col = "polarity", verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          df (pd.DataFrame): dataframe containing the text to train the model on
          text_col (str): the column of the dataframe containing the input text
          category_col (str): the column of the dataframe containing the aspect category
          label_col (str): the column of the dataframe containing the sentiment
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        # initialize class variables
        self.ngrams = dict()
        self.category_probabilities = defaultdict(lambda: defaultdict(int))
        
        # get the list of all tokens to determine rare words
        tokens = tokenize(df, text_col, self.n)
        self.raw_counts = Counter(tokens)
        
        # get the list of categories
        categories = df[category_col].unique()
        
        # for each category, filter the rows
        for category in categories:
            # defaultdict used to easily add elements, initialize to 1 for smoothing purposes
            self.ngrams[category] = defaultdict(lambda: defaultdict(lambda: 1)) 
            category_df = df[df[category_col] == category]
            # for each row, tokenize the sentence and create ngrams
            for index, row in category_df.iterrows():
                sent_tokens = tokenize_line(row[text_col], self.n)
                filtered_sent_tokens = self.smooth_tokens(sent_tokens)
                ngrams = create_ngrams(filtered_sent_tokens, self.n)
                polarity = row[label_col]
                
                # update class counts for this category
                self.category_probabilities[category][polarity] += 1
                
                # for each ngram in the set of created ngrams, update the ngram counts for this category
                for ngram in ngrams:
                    # defaultdict should create these entries if they don't exist
                    self.ngrams[category][ngram][polarity] += 1
        if verbose:
            print(self.category_probabilities)
   
    def smooth_tokens(self, tokens: list) -> list:
        """Smooths a list of tokens by replacing rare words with UNK
         Args:
          tokens (list): the list of tokens to be smoothed
         Returns:
          list: list of smoothed tokens
        """
        if not self.raw_counts:
            raise ValueError("Model not yet trained")
        return [token if is_special(token) or self.raw_counts[token] > 1 else UNK for token in tokens]
    
    def score(self, input_string: str, category: str = 'food', verbose = False) -> tuple[str, list]:
        """Calculates the probability scores for each polarity for a given string representing a single sequence of tokens.
        Args:
          input_string (str): a tokenized sequence to be scored by this model
          category (str): the category to determine the sentiment for
          verbose (bool): toggle debug output
        Returns:
          str: the most likely class for this string
          list: list of tuples, with each tuple containing the polarity and the corresponding
        """
        if not self.ngrams:
            raise ValueError("Model not yet trained")

        # default category is food        
        tokens = tokenize_line(input_string, self.n)

        # replace rare words with UNK
        smooth_tokens = self.smooth_tokens(tokens)
        
        ngrams = create_ngrams(smooth_tokens, self.n)
        polarities = self.category_probabilities[category].keys()
        category_total = sum(self.category_probabilities[category].values())
        scores = dict()
        
        for polarity in polarities:
            # probability string is <polarity> = 
            # P(positive) * PRODUCT(# occurrences of ngram in positive docs / total words in positive docs) for each ngram
            score = self.category_probabilities[category][polarity] / category_total # P(<polarity>)
            
            if verbose:
                print(f'probability of {polarity} = {score}')
            
            # get the total number of ngrams in this polarity
            # could calculate this value once during training
            polarity_count = 0
            for ngram, polarity_counts in self.ngrams[category].items():
                polarity_count += polarity_counts[polarity]
            
            if verbose:
                print(f'number of terms with polarity {polarity} = {polarity_count}')
            for ngram in ngrams:
                ngram_polarity_occurrence = self.ngrams[category][ngram][polarity]
                if verbose:
                    print(f'number of times {ngram} appeared in polarity = {ngram_polarity_occurrence} / {polarity_count}')
                score *= ngram_polarity_occurrence / polarity_count
            scores[polarity] = score
        label = max(scores, key=scores.get)
        return label, scores
    