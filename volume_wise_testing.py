import os
import sys
import keras
import numpy as np
import keras.optimizers as opt
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import (Input, Conv2DTranspose,
                          MaxPooling3D, Concatenate, UpSampling2D,
                          Activation, BatchNormalization, Dense,
                          Flatten, Dropout, Conv3D)
from mri_functions import load_row_from_csv, load_cases_from_csv
from math import floor
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt

# processes test data to evaluate network performance
def data_preparation(pos_file, neg_file):
    pos_cases = load_cases_from_csv(csv_file_name=pos_file)
    neg_cases = load_cases_from_csv(csv_file_name=neg_file)


    # cases were made same size
    pos_test_cases, neg_test_cases = len(pos_cases), len(neg_cases)

    test_data = np.zeros(shape=(pos_test_cases + neg_test_cases, 40))


    for case_idx, case in enumerate(pos_cases):

        row = load_row_from_csv(csv_file_name="pos_test_class_3.csv", row_id=case_idx)

        # print("Size array: {0}".format(len(row)))

        if case_idx < pos_test_cases:
            if len(row) < 40:
                num_entry = 40 - len(row)
                for number_e in range(num_entry):
                    row.append(0)

            elif len(row) > 40:
                num_entry = len(row) - 40
                for number_e in range(num_entry):
                    row.pop(0)

            test_data[case_idx, :] = row

    for neg_case_idx, case in enumerate(neg_cases):
        # print("NegCase: {0}".format(neg_case_idx))

        row = load_row_from_csv(csv_file_name="neg_test_class_3.csv", row_id=neg_case_idx)
        if neg_case_idx < neg_test_cases:
            if len(row) < 40:
                num_entry = 40 - len(row)
                for number_e in range(num_entry):
                    row.append(0)
            elif len(row) > 40:
                num_entry = len(row) - 40
                for number_e in range(num_entry):
                    row.pop(0)

            test_data[neg_case_idx + pos_test_cases, :] = row


    test_labels = np.array([1] * pos_test_cases + [0] * neg_test_cases)

    for test_idx in range(test_data.shape[0]):
        print(test_data[test_idx, :])
    print(test_labels)

    return test_data, test_labels



def main():

    # model = load_model("models/model_agg_run_1.hdf5")
    # pos_file, neg_file = "test_list/positive_test_list_run_1.csv", "test_list/negative_test_list_run_1.csv"

    # loads trained "3D" volume-wise model
    model = load_model("model_agg_3.hdf5")

    # specifies files listing positive and negative cases
    pos_file, neg_file = "positive_test_list_3.csv", "negative_test_list_3.csv"

    # arrays to store network predicted labels and generated scores
    predicted_labels, network_scores = [], []

    # processes test data for performance evaluation
    test_data, test_labels = data_preparation(pos_file=pos_file, neg_file=neg_file)

    # compiles model with classification specific parameters
    model.compile(optimizer=opt.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])


    # iterates through samples of test dataset
    for test_idx in range(test_data.shape[0]):

        # takes sample to perform network inference on it
        buf_sample = np.array(test_data[test_idx, :])

        # reshapes scores' array to network expected shape
        buf_sample = buf_sample.reshape((1,40))

        # performs model prediction on sample of data
        pre_score = model.predict(buf_sample)

        # stores prediction score in score array
        network_scores.append(pre_score)

        # defines volume label from score
        def_label = 1 if pre_score > 0.5 else 0

        # stores label prediction in array
        predicted_labels.append(def_label)


    #target_names = ['Non Active', 'Active']
    #print(classification_report(test_labels, np.array(predicted_labels), target_names=target_names))


    # Performance metrics before assessment with Youden index

    print("Accuracy: {0}".format(accuracy_score(test_labels, predicted_labels)))
    print("Recall: {0}".format(recall_score(test_labels, predicted_labels)))
    print("Precision: {0}".format(precision_score(test_labels, predicted_labels)))

    # obtains confusion matrix for network performance vs ground truth
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_labels).ravel()
    print("TN: {0} FP: {1} FN: {2} TP: {3}".format(tn, fp, fn, tp))

    #
    # # # # #
    # analysis under optimal threshold
    # # # # #
    #

    fpr, tpr, thresholds = roc_curve(test_labels.flatten(), np.array(network_scores).flatten(), pos_label=1)

    # calculation of optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print("Youden Analysis")
    print("---------------")
    print("Optimal: {0}".format(optimal_threshold))


    # calculates new labels using Youden index
    optimal_labels = []
    for test_score in network_scores:
        opt_label = 1 if test_score > optimal_threshold else 0
        optimal_labels.append(opt_label)

    # performance metrics calculations
    print("Accuracy: {0}".format(accuracy_score(test_labels, optimal_labels)))
    print("Recall: {0}".format(recall_score(test_labels, optimal_labels)))
    print("Precision: {0}".format(precision_score(test_labels, optimal_labels)))
    tn, fp, fn, tp = confusion_matrix(test_labels, optimal_labels).ravel()
    print(" TN: {0} FP: {1} FN: {2} TP: {3}".format(tn, fp, fn, tp))





if __name__ == '__main__':
    main()
