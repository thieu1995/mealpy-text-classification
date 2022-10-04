#!/usr/bin/env python
# Created by "Thieu" at 08:08, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def generate_data(path_to_data="data/corpus.txt", test_size=0.3):
    # load the dataset
    with open(path_to_data, encoding='utf8') as f:
        data = f.read()

    labels, texts = [], []
    for idx, line in enumerate(data.split("\n")):
        content = line.split()
        labels.append(content[0])
        texts.append(" ".join(content[1:]))

    # create a dataframe using texts and labels
    trainDF = pd.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = train_test_split(trainDF['text'], trainDF['label'], test_size=test_size)

    # label encode the target variable
    encoder = LabelEncoder()
    encoder.fit(train_y)
    train_y = encoder.transform(train_y)
    valid_y = encoder.transform(valid_y)

    return {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}


## Feature engineering section

def features_as_count_vectors(trainDF, train_x, valid_x, network=False):
    """
    Count Vector is a matrix notation of the dataset in which every row represents a document
    from the corpus, every column represents a term from the corpus, and every cell represents
    the frequency count of a particular term in a particular document.
    """
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    X_train = count_vect.transform(train_x)
    X_valid = count_vect.transform(valid_x)

    if network:
        return X_train.toarray(), X_valid.toarray()
    return X_train, X_valid


def features_as_TF_IDF_vectors(trainDF, train_x, valid_x, kind="word", max_features=5000):
    """
    kind = "word", "N-gram", or "char"

    word: Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents
    ngram: N-gram Level TF-IDF : N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams
    char: Character Level TF-IDF : Matrix representing tf-idf scores of character level n-grams in the corpus
    """

    if kind == "word":
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
        tfidf_vect.fit(trainDF['text'])
        X_train = tfidf_vect.transform(train_x)
        X_valid = tfidf_vect.transform(valid_x)

    elif kind == "ngram":
        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=max_features)
        tfidf_vect_ngram.fit(trainDF['text'])
        X_train = tfidf_vect_ngram.transform(train_x)
        X_valid = tfidf_vect_ngram.transform(valid_x)

    elif kind == "char":
        # characters level tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=max_features)
        tfidf_vect_ngram_chars.fit(trainDF['text'])
        X_train = tfidf_vect_ngram_chars.transform(train_x)
        X_valid = tfidf_vect_ngram_chars.transform(valid_x)
    else:
        raise TypeError("kin parameter is only supported 'word', 'ngram' or 'char'!")

    return X_train, X_valid


def features_as_word_embeddings():
    """
    1. Loading the pretrained word embeddings
    2. Creating a tokenizer object
    3. Transforming text documents to sequence of tokens and pad them
    4. Create a mapping of token and their respective embeddings
    """
    # load the pre-trained word-embedding vectors
    # embeddings_index = {}
    # for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
    #     values = line.split()
    #     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
    #
    # # create a tokenizer
    # token = text.Tokenizer()
    # token.fit_on_texts(trainDF['text'])
    # word_index = token.word_index
    #
    # # convert text to sequence of tokens and pad them to ensure equal length vectors
    # train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    # valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
    #
    # # create token-embedding mapping
    # embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector


def features_as_topic_models():
    """
    Topic Modelling is a technique to identify the groups of words (called a topic) from a collection of documents
    that contains best information in the collection. I have used Latent Dirichlet Allocation for generating
    Topic Modelling Features. LDA is an iterative model which starts from a fixed number of topics.
    Each topic is represented as a distribution over words, and each document is then represented as a distribution over topics.
    Although the tokens themselves are meaningless, the probability distributions over words provided by the topics
    provide a sense of the different ideas contained in the documents.
    """
    # train a LDA Model
    # lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
    # X_topics = lda_model.fit_transform(xtrain_count)
    # topic_word = lda_model.components_
    # vocab = count_vect.get_feature_names()
    #
    # # view the topic src
    # n_top_words = 10
    # topic_summaries = []
    # for i, topic_dist in enumerate(topic_word):
    #     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    #     topic_summaries.append(' '.join(topic_words))

