#!/usr/bin/env python
# Created by "Thieu" at 00:05, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.preprocessing import LabelEncoder
from models.classify_lc import ClassifyLC
from models.utils.data_util import generate_data
from models.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors
from permetrics.classification import ClassificationMetric
from mealpy.swarm_based import WOA

"""
Tuning hyper-parameter of Logistic Regression model with 3 parameters

C : float, default=1.0
    Inverse of regularization strength; must be a positive float.
    Like in support vector machines, smaller values specify stronger regularization.

fit_intercept : bool, default=True
    Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
"""


if __name__ == "__main__":
    list_solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    solver_encoder = LabelEncoder()
    solver_encoder.fit(list_solvers)

    ## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
    df = generate_data()
    df["SOLVER_ENCODER"] = solver_encoder

    ## Count Vectors feature engineering
    X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
    df["X_train"] = X_train
    df["X_valid"] = X_valid

    # x1. C: float [0.1 to 10000.0]
    # x2. fit_intercept: [0, 1]
    # x3. solver: [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]
    LB = [0.1, 0., 0.]
    UB = [10000.0, 1.0, 3.99]

    problem = ClassifyLC(lb=LB, ub=UB, minmax="min", data=df, save_population=False, log_to="console")

    algorithm = WOA.OriginalWOA(epoch=10, pop_size=20)
    best_position, best_fitness = algorithm.solve(problem)

    best_solution = problem.decode_solution(best_position)

    print(f"Best fitness (accuracy score) value: {1 - best_fitness}")
    print(f"Best parameters: {best_solution}")

    ###### Get the best tuned model to predict test set
    best_model = problem.generate_trained_model(best_solution)
    y_valid = best_model.predict(df["X_valid"])

    evaluator = ClassificationMetric(df["y_valid"], y_valid, decimal=6)
    print(evaluator.get_metrics_by_list_names(["AS", "RS", "PS", "F1S", "F2S"]))
