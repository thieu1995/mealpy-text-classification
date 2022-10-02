#!/usr/bin/env python
# Created by "Thieu" at 21:14, 01/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from models.classify_nb import ClassifyNB
from models.utils.data_util import generate_data
from models.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors
from permetrics.classification import ClassificationMetric
from mealpy.swarm_based import WOA

"""
Tuning hyper-parameter of Naive Bayes model with 2 parameters

alpha : float [0, 1.0]
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

fit_prior : bool, default=True
    Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
"""

if __name__ == "__main__":
    ## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
    df = generate_data()

    ## Count Vectors feature engineering
    X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
    df["X_train"] = X_train
    df["X_valid"] = X_valid

    LB = [0.01, 0.0]
    UB = [1.0, 1.0]

    problem = ClassifyNB(lb=LB, ub=UB, minmax="min", data=df, save_population=False, log_to="console")

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
