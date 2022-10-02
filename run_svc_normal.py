#!/usr/bin/env python
# Created by "Thieu" at 21:38, 01/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.svm import SVC
from models.utils.model_util import training_process
from models.utils.data_util import generate_data
from models.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors


## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
df = generate_data()


## Features as Count Vectors
# X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
# accuracy = training_process(SVC(), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, Count Vectors: ", accuracy)



## Features as Word Level TF IDF Vectors
# X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="word")
# accuracy = training_process(SVC(), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, WordLevel TF-IDF: ", accuracy)



## Features as Ngram Level TF IDF Vectors
# X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="ngram")
# accuracy = training_process(SVC(), X_train, df["y_train"], X_valid, df["y_valid"])
# print("NB, N-Gram Vectors: ", accuracy)



## Features as Character Level TF IDF Vectors
X_train, X_valid = features_as_TF_IDF_vectors(df["trainDF"], df["train_x"], df["valid_x"], kind="char")
accuracy = training_process(SVC(), X_train, df["y_train"], X_valid, df["y_valid"])
print("NB, CharLevel Vectors: ", accuracy)

