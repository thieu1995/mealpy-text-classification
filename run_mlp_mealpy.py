#!/usr/bin/env python
# Created by "Thieu" at 09:32, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.preprocessing import LabelEncoder
from src.classify_mlp import ClassifyMLP
from src.utils.data_util import generate_data, features_as_count_vectors
from permetrics.classification import ClassificationMetric
from mealpy.swarm_based import WOA

"""
Tuning hyper-parameter of Multi-layer Perceptron model with 3 parameters

n_unit : int, [2, 100]
        Number of hidden unit

activation: str, ["elu", "relu", "selu", "gelu", "tanh", "sigmoid", "exponential", "linear"]
        Activation of hidden layer.
        Since we are using binary-crossentropy loss function, so the activation of output layer is set to sigmoid.
        https://keras.io/api/layers/activations/

optimizer: str, ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]
        The optimizer of neural network
        https://keras.io/api/optimizers/
"""

if __name__ == "__main__":
    list_acts = ["elu", "relu", "selu", "gelu", "tanh", "sigmoid", "exponential", "linear"]
    list_opts = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]

    act_encoder = LabelEncoder()
    act_encoder.fit(list_acts)

    opt_encoder = LabelEncoder()
    opt_encoder.fit(list_opts)

    ## {"train_x": train_x, "y_train": train_y, "valid_x": valid_x, "y_valid": valid_y, "encoder": encoder, "trainDF": trainDF}
    df = generate_data()

    ## Count Vectors feature engineering
    X_train, X_valid = features_as_count_vectors(df["trainDF"], df["train_x"], df["valid_x"], network=True)
    df["X_train"] = X_train
    df["X_valid"] = X_valid

    df["ACT_ENCODER"] = act_encoder
    df["OPT_ENCODER"] = opt_encoder
    df["input_size"] = X_train.shape[1]

    LB = [2, 0.0, 0.0]
    UB = [100, 7.99, 7.99]

    problem = ClassifyMLP(lb=LB, ub=UB, minmax="min", data=df, save_population=False, log_to="console")

    algorithm = WOA.OriginalWOA(epoch=10, pop_size=20)
    best_position, best_fitness = algorithm.solve(problem)

    best_solution = problem.decode_solution(best_position)

    print(f"Best fitness (accuracy score) value: {1 - best_fitness}")
    print(f"Best parameters: {best_solution}")

    ###### Get the best tuned model to predict test set
    best_model = problem.generate_trained_model(best_solution)
    predictions = best_model.predict(df["X_valid"])
    y_valid = predictions.argmax(axis=-1)

    evaluator = ClassificationMetric(df["y_valid"], y_valid, decimal=6)
    print(evaluator.get_metrics_by_list_names(["AS", "RS", "PS", "F1S", "F2S"]))

