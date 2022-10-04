#!/usr/bin/env python
# Created by "Thieu" at 09:19, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from keras import layers, models
from src.utils.model_util import training_process
from src.utils.data_util import generate_data, features_as_count_vectors


def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size,), sparse=True)
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer="adam", loss='binary_crossentropy')
    return classifier


## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
df = generate_data()

## Features as Count Vectors
X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"], network=True)
classifier = create_model_architecture(X_train.shape[1])
accuracy = training_process(classifier, X_train, df["y_train"], X_valid, df["y_valid"], is_neural_net=True)
print("NB, Count Vectors: ", accuracy)
