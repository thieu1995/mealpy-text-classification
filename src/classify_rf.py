#!/usr/bin/env python
# Created by "Thieu" at 00:19, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.ensemble import RandomForestClassifier
from permetrics.classification import ClassificationMetric
from mealpy.utils.problem import Problem


class ClassifyRF(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Random Forest", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def decode_solution(self, solution):
        # C = solution[0]
        #
        # kernel_integer = int(solution[1])
        # kernel = KERNEL_ENCODER.inverse_transform([kernel_integer])[0]
        # 0 - 0.99 ==> 0 index ==> should be linear (for example)
        # 1 - 1.99 ==> 1 index ==> should be poly

        estimator_integer = int(solution[0])
        estimator = 10 * self.data["ESTIMATOR_ENCODER"].inverse_transform([estimator_integer])[0]

        depth_integer = int(solution[1])
        depth = self.data["DEPTH_ENCODER"].inverse_transform([depth_integer])[0]
        return {
            "n_estimators": estimator,
            "max_depth": depth,
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        model = RandomForestClassifier(n_estimators=structure["n_estimators"], max_depth=structure["max_depth"])
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
