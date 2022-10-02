#!/usr/bin/env python
# Created by "Thieu" at 08:48, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from models.classify_xgb import ClassifyXGB
from models.utils.data_util import generate_data
from models.utils.data_util import features_as_count_vectors, features_as_TF_IDF_vectors
from permetrics.classification import ClassificationMetric
from mealpy.swarm_based import WOA

"""
https://xgboost.readthedocs.io/en/stable/parameter.html

Tuning hyper-parameter of Extreme Gradient Boosting (XGB) model with booster="gbtree" and:

eta (learning_rate) [default=0.3, alias: learning_rate], range: [0,1]
    Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the 
    weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.

gamma (min_split_loss) [default=0, alias: min_split_loss], range: [0,100]
    Minimum loss reduction required to make a further partition on a leaf node of the tree. 
    The larger gamma is, the more conservative the algorithm will be.

max_depth [default=6], range: [2,10]
    Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 
    0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. 
    exact tree method requires non-zero value.

lambda (reg_lambda) [default=1, alias: reg_lambda], range: [0., 10.]
    L2 regularization term on weights. Increasing this value will make model more conservative.

alpha (reg_alpha) [default=0, alias: reg_alpha], range: [0., 10.]
    L1 regularization term on weights. Increasing this value will make model more conservative.

"""

if __name__ == "__main__":
    ## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
    df = generate_data()

    ## Tuning hyper-parameter of SVM model with Count Vectors feature engineering
    X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"])
    df["X_train"] = X_train
    df["X_valid"] = X_valid

    LB = [0.0, 0.0, 2, 0.0, 0.0]
    UB = [1.0, 100, 10, 10.0, 10.0]

    problem = ClassifyXGB(lb=LB, ub=UB, minmax="min", data=df, save_population=False, log_to="console")

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
