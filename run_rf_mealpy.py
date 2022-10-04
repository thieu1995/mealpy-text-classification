#!/usr/bin/env python
# Created by "Thieu" at 00:22, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.preprocessing import LabelEncoder
from src.classify_rf import ClassifyRF
from src.utils.data_util import generate_data
from src.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors
from permetrics.classification import ClassificationMetric
from mealpy.swarm_based import WOA

"""
Tuning hyper-parameter of Random Forest (Bagging) model with 2 parameters


n_estimators : int, default=100
        The number of trees in the forest.
       
max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.    
"""


if __name__ == "__main__":
    list_estimators = list(range(1, 50))                    # 49 values
    estimator_encoder = LabelEncoder()
    estimator_encoder.fit(list_estimators)

    list_depths = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]        # 10 values
    depth_encoder = LabelEncoder()
    depth_encoder.fit(list_depths)

    ## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
    df = generate_data()
    df["ESTIMATOR_ENCODER"] = estimator_encoder
    df["DEPTH_ENCODER"] = depth_encoder

    ## Count Vectors feature engineering
    X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
    df["X_train"] = X_train
    df["X_valid"] = X_valid

    ## x[0]: n-estimator has 49 values (0 to 48: 49 values)
    ## x[1]: max-depth has 10 values (0 to 9: 10 values)
    LB = [0.0, 0.0]
    UB = [48.99, 9.99]

    problem = ClassifyRF(lb=LB, ub=UB, minmax="min", data=df, save_population=False, log_to="console")

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
