import numpy as np
from sklearn.metrics import roc_curve, auc
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


def quick_file_append_list(file_name, data):
    data_file = open(file_name, "a+")
    for idx in range(len(data)):
        data_file.write("{0}".format(data[idx]))
        data_file.write("\n")

    data_file.close()

def quick_file_append(file_name, data):
    data_file = open(file_name, "a+")
    data_file.write("{0}".format(data))
    data_file.write("\n")
    data_file.close()

# evaluation of network performance on testing data
def check_model(model, test_data, test_labels):

    # loads model from file to assess prediction on testing data
    print("Model Name: {0}".format(model))

    model_pred = load_model(model)

    # array which stores network predicted labels ex. [0 1 0 ...]
    stats_arr = np.ndarray((test_data.shape[0]))

    # array which stores network predicted scores ex. [0.4 0.7 0.2 ...]
    score_arr = np.ndarray((test_data.shape[0]))

    # loops through slices in testing set evaluating performance
    for slice_idx in range(test_data.shape[0]):

        # selects one slice from testing set
        image_to_predict = test_data[slice_idx, :, :, :]

        # reshapes slice into Keras specific format
        image_to_predict = image_to_predict.reshape(1, test_data.shape[1], test_data.shape[2], 3)

        # performs network prediction on testing data
        predicted_score = model_pred.predict(image_to_predict)

        # defines predicted label upon network output score
        if predicted_score[0][0] > 0.5:
            out_score = 1
        else:
            out_score = 0

        # saves predicted label and score in defined arrays
        stats_arr[slice_idx] = int(out_score)
        score_arr[slice_idx] = predicted_score[0][0]

    # finds network accuracy upon ground truth labels and predicted labels
    accuracy = (accuracy_score(stats_arr, test_labels))

    print(stats_arr)
    print(test_labels)

    print(accuracy)

    # lists for obtaining network performance metrics
    pos_vals = []
    neg_vals = []

    # obtains arrays for calculating performance metrics
    for val in range(len(stats_arr)):
        if stats_arr[val] == test_labels[val] and test_labels[val] == 1 :
            pos_vals.append(1)
        elif test_labels[val] == 1:
            pos_vals.append(0)

    for val in range(len(stats_arr)):
        if stats_arr[val] == test_labels[val] and test_labels[val] == 0 :
            neg_vals.append(1)
        elif test_labels[val] == 0:
            neg_vals.append(0)


    # finds positive predictive value, negative predictive value, plus additional metrics
    ppv = sum(pos_vals) / len(pos_vals)
    npv = sum(neg_vals) / len(neg_vals)
    sensitivity = sum(pos_vals) / (sum(pos_vals) + (len(neg_vals) - sum(neg_vals)))
    specificity = sum(neg_vals) / (sum(neg_vals) + (len(pos_vals) - sum(pos_vals)))

    # prints obtained performance metrics
    print("Accuracy: {0} Sensitivity: {1} Specificity: {2} PPV: {3} NPV: {4}".format(accuracy, sensitivity, specificity,
                                                                                     ppv, npv))

    # obtains fpr tpr values upon ground truth labels and network predicted scores
    fpr, tpr, thresholds = roc_curve(test_labels.flatten(), score_arr.flatten(), pos_label=1)

    # additional files to evaluate ROC curve of network predicted scores
    quick_file_append_list("met_tpr.csv", tpr.flatten())
    quick_file_append_list("met_fpr.csv", fpr.flatten())


    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # at this step evaluation upon Youden index is performed
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #

    # finds optimal (Youden) threshold to maximize network predictive capabilities
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # arrays to hold network scores and optimal threshold labels
    opt_stats_arr = np.ndarray((test_data.shape[0]))
    opt_score_arr = np.ndarray((test_data.shape[0]))

    # loops through testing set data slices
    for slice_idx_opt in range(test_data.shape[0]):

        # takes test set slice for prediction by network
        image_to_predict = test_data[slice_idx_opt, :, :, :]

        # reshapes sample slice for Keras specific format
        image_to_predict = image_to_predict.reshape(1, test_data.shape[1], test_data.shape[2], 3)

        # performs network prediction on sample slice
        predicted_score = model_pred.predict(image_to_predict)

        # find network predicted labels upon optimal threshold
        if predicted_score[0][0] > optimal_threshold:
            out_score = 1
        else:
            out_score = 0

        # saves obtained labels and scores to previously defined lists
        opt_stats_arr[slice_idx_opt] = int(out_score)
        opt_score_arr[slice_idx_opt] = predicted_score[0][0]


    # finds confusion matrix to obtain performance metrics
    tn, fp, fn, tp = confusion_matrix(test_labels, opt_stats_arr).ravel()

    print("TN: {0} | FP: {1} | FN: {2} | TP: {3}".format(tn, fp, fn, tp))

    # assessment performance metrics
    acc = (tp+tn)/(tp+tn+fn+fp)
    sensi = tp/(tp+fn)
    speci = tn/(tn+fp)
    pospv = tp/(tp+fp)
    negpv = tn/(tn+fn)
    fpr = fn/(fn+tp)
    fnr = fp/(fp+tn)

    # prints network performance metrics
    print("Accuracy: {0} Sensitivity: {1} Specificity: {2} PPV: {3} NPV: {4} FPR: {5} FNR: {6}".format(acc, sensi, speci,
                                                                                     pospv, negpv, fpr, fnr))

    # obtains fpr, tpr based on Youden index
    fpr, tpr, thresholds = roc_curve(test_labels.flatten(), score_arr.flatten(), pos_label=1)

    # saves fpr, tpr values for assesment of ROC curve
    np.save(file="fpr_run1.npy", arr=fpr)
    np.save(file="tpr_run1.npy", arr=tpr)

    # returns performance metrics for non-Youden analysis
    return accuracy, sensitivity, specificity, ppv, npv


def main():
    pass


if __name__ == '__main__':
    main()
