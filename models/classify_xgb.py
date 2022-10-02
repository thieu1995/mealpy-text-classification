#!/usr/bin/env python
# Created by "Thieu" at 07:47, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from xgboost import XGBClassifier
from permetrics.classification import ClassificationMetric
from mealpy.utils.problem import Problem


class ClassifyXGB(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Extreme Gradient Boosting", **kwargs):
        super().__init__(lb, ub, minmax, data=data, **kwargs)  ## data is needed because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.name = name

    def decode_solution(self, solution):
        return {
            "learning_rate": solution[0],
            "min_split_loss": solution[1],
            "max_depth": int(solution[2]),
            "reg_lambda": solution[3],
            "reg_alpha": solution[4]
        }

    def generate_trained_model(self, structure):
        model = XGBClassifier(booster="gbtree", learning_rate=structure["learning_rate"], min_split_loss=structure["min_split_loss"],
                              max_depth=structure["max_depth"], reg_lambda=structure["reg_lambda"], reg_alpha=structure["reg_alpha"])
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
