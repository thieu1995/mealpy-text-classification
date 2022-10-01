#!/usr/bin/env python
# Created by "Thieu" at 20:37, 01/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
Naive Bayes

Naive Bayes is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors.
A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature
"""
from permetrics.classification import ClassificationMetric
from sklearn.naive_bayes import MultinomialNB
from mealpy.utils.problem import Problem


class ClassifyNB(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Naive Bayes", **kwargs):
        super().__init__(lb, ub, minmax, data=data, **kwargs)  ## data is needed because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.name = name

    def decode_solution(self, solution):
        return {
            "alpha": solution[0],
            "fit_prior": bool(round(solution[1]))
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        model = MultinomialNB(alpha=structure["alpha"], fit_prior=["fit_prior"])
        model.fit(self.data["X_train"], self.data["y_train"])
        return model

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        y_pred = model.predict(self.data["X_valid"])
        evaluator = ClassificationMetric(self.data["y_valid"], y_pred, decimal=6)
        acc = evaluator.accuracy_score(average="macro")
        return 1 - acc

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        return fitness
