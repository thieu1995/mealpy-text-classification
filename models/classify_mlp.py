#!/usr/bin/env python
# Created by "Thieu" at 09:11, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
Multi-layer perceptron (MLP)

A neural network is a mathematical model that is designed to behave similar to biological neurons and nervous system.
These models are used to recognize complex patterns and relationships that exists within a labelled data.
A shallow neural network (MLP) contains mainly three types of layers – input layer, hidden layer, and output layer.

"""
from keras import layers, models
from permetrics.classification import ClassificationMetric
from mealpy.utils.problem import Problem


class ClassifyMLP(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Multi-layer Perceptron", **kwargs):
        super().__init__(lb, ub, minmax, data=data, **kwargs)  ## data is needed because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.name = name

    def decode_solution(self, solution):
        act_int = int(solution[1])
        opt_int = int(solution[2])
        act = self.data["ACT_ENCODER"].inverse_transform([act_int])[0]
        opt = self.data["OPT_ENCODER"].inverse_transform([opt_int])[0]
        return {
            "n_unit": int(solution[0]),
            "activation": act,
            "optimizer": opt
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')

        # create input layer
        input_layer = layers.Input((self.data["input_size"],), sparse=True)

        # create hidden layer
        hidden_layer = layers.Dense(structure["n_unit"], activation=structure["activation"])(input_layer)

        # create output layer
        output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

        classifier = models.Model(inputs=input_layer, outputs=output_layer)
        classifier.compile(optimizer=structure["optimizer"], loss='binary_crossentropy')

        classifier.fit(self.data["X_train"], self.data["y_train"], verbose=0)
        return classifier

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        predictions = model.predict(self.data["X_valid"], verbose=0)
        y_pred = predictions.argmax(axis=-1)
        evaluator = ClassificationMetric(self.data["y_valid"], y_pred, decimal=6)
        acc = evaluator.accuracy_score(average="macro")
        return 1 - acc

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        return fitness
