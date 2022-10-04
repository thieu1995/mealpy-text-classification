#!/usr/bin/env python
# Created by "Thieu" at 23:49, 01/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
Logistic Regression (aka logit, MaxEnt) classifier

Logistic regression measures the relationship between the categorical dependent variable and one or
more independent variables by estimating probabilities using a logistic/sigmoid function.
"""
from permetrics.classification import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from mealpy.utils.problem import Problem


class ClassifyLC(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Logistic Regression", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def decode_solution(self, solution):
        # solver_integer = int(solution[1])
        # solver = SOLVER_ENCODER.inverse_transform([solver_integer])[0]
        # 0 - 0.99 ==> 0 index ==> should be 'newton-cg' (for example)
        # 1 - 1.99 ==> 1 index ==> should be 'lbfgs'

        solver_integer = int(solution[1])
        solver = self.data["SOLVER_ENCODER"].inverse_transform([solver_integer])[0]
        return {
            "C": solution[0],
            "fit_intercept": bool(round(solution[1])),
            "solver": solver
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        model = LogisticRegression(C=structure["C"], fit_intercept=structure["fit_intercept"], solver=structure["solver"], max_iter=1000)
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
