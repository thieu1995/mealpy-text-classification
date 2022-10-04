#!/usr/bin/env python
# Created by "Thieu" at 20:48, 01/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.naive_bayes import MultinomialNB
from src.utils.model_util import training_process
from src.utils.data_util import generate_data
from src.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors


## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
df = generate_data()


## Features as Count Vectors
# X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
# accuracy = training_process(MultinomialNB(alpha=0.9), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, Count Vectors: ", accuracy)



## Features as Word Level TF IDF Vectors
# X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="word")
# accuracy = training_process(MultinomialNB(alpha=0.9), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, WordLevel TF-IDF: ", accuracy)



## Features as Ngram Level TF IDF Vectors
# X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="ngram")
# accuracy = training_process(MultinomialNB(alpha=0.9), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, N-Gram Vectors: ", accuracy)



## Features as Character Level TF IDF Vectors
X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="char")
accuracy = training_process(MultinomialNB(alpha=0.9), X_train, df["y_train"], X_valid, df["y_valid"])
print("NB, CharLevel Vectors: ", accuracy)

