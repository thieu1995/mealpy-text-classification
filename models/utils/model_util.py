#!/usr/bin/env python
# Created by "Thieu" at 20:25, 01/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from permetrics.classification import ClassificationMetric


def training_process(classifier, features_train, labels_train, features_valid, labels_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(features_train, labels_train)

    # predict the labels on validation dataset
    predictions = classifier.predict(features_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    evaluator = ClassificationMetric(labels_valid, predictions, decimal=6)
    acc = evaluator.accuracy_score(average="macro")
    return acc

