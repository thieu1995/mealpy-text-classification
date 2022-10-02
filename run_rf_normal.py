#!/usr/bin/env python
# Created by "Thieu" at 00:18, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.ensemble import RandomForestClassifier
from models.utils.model_util import training_process
from models.utils.data_util import generate_data
from models.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors


## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
df = generate_data()


## Naive Bayes on Count Vectors
# X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
# accuracy = training_process(RandomForestClassifier(), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, Count Vectors: ", accuracy)



## Naive Bayes on Word Level TF IDF Vectors
# X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="word")
# accuracy = training_process(RandomForestClassifier(), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, WordLevel TF-IDF: ", accuracy)



## Naive Bayes on Ngram Level TF IDF Vectors
# X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="ngram")
# accuracy = training_process(RandomForestClassifier(), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, N-Gram Vectors: ", accuracy)



## Naive Bayes on Character Level TF IDF Vectors
X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="char")
accuracy = training_process(RandomForestClassifier(), X_train, df["y_train"], X_valid, df["y_valid"])
print("NB, CharLevel Vectors: ", accuracy)

