import os
import numpy as np
from keras.models import load_model
from skimage.measure import label, regionprops
from mri_functions import load_cases_from_csv, data_extraction_from_files


# function which processes slices to count number of enhancements and lesions
def count_tissue_in_volume(volume_label, gad_class=2, predict_list=None, score_list=None):

    # variables to count and check gad enhancements
    count, gad_count, gad_detected = 0, 0, False

    # isolates enhancements from ground truth
    gad_locations = np.isin(volume_label, gad_class)

    #
    vol_lesions = label(gad_locations, return_num=True, connectivity=2)

    # lists to store unique slices presenting lesion and enhancements
    gt_slices, t2_lesions_slices = [], []

    # checks slices which contain enhancement
    for lesion in regionprops(vol_lesions[0]):
        for coordinate in lesion.coords:
            gt_slices.append(coordinate[-1])


    # checks for number of enhancing lesions and slices
    for lesion in regionprops(vol_lesions[0]):
        # counts gad lesion in volume
        count += 1
        # goes through lesion coordinates to identify slices presenting enhancement
        for coordinate in lesion.coords:
            z_coord = coordinate[-1]

            if predict_list[z_coord] == 1:
                gad_detected = True
                break

        if gad_detected:
            gad_detected = False
            gad_count += 1

    #
    gt_slices = np.array(gt_slices)
    vol_gad = True if 1 in predict_list else False




    # array to store number of enhancements per slice per volume
    gad_count_val_list = np.zeros(volume_label.shape[-1],)

    # counts number of enhancements per slice per volume asnd stores it in array
    for z_idx in range(volume_label.shape[-1]):

        lesion_count = 0

        slice_label = volume_label[:, :, z_idx]

        slice_locations = np.isin(slice_label, gad_class)

        slice_lesions = label(slice_locations, return_num=True, connectivity=2)

        # counts number of lesions per slice
        for lesion in regionprops(slice_lesions[0]):
            lesion_count += 1

        gad_count_val_list[z_idx] = lesion_count



    # checks for presence of t2-lesions in volume using ground truth
    for z_axis_idx in range(volume_label.shape[-1]):
        t2_lesion_check = 1 if 1 in volume_label[:, :, z_axis_idx] else 0
        t2_lesions_slices.append(t2_lesion_check)

    # counts number of t2-lesions in volume
    count_slices_t2 = 0
    for entry_idx, entry in enumerate(t2_lesions_slices):
        if entry == 1 and predict_list[entry_idx] == 1:
            count_slices_t2 += 1

    # variables for number of enhancing lesions in ground truth and network output
    num_gad_val, num_gad_network = count, gad_count

    # counts slices in ground truth and network
    num_slices_val, num_slices_predicted_network = len(np.unique(gt_slices)), np.sum(predict_list)

    # arrays of ground truth and network output slice-wise labels and scores
    slices_scores, slices_predicts, slices_val = score_list, predict_list, gad_count_val_list

    # array containing all data obtained from ground truth and network predictions
    case_eval = [num_gad_network, num_gad_val, num_slices_predicted_network
        , num_slices_val, slices_scores, slices_predicts, slices_val, vol_gad, t2_lesions_slices, count_slices_t2]

    return case_eval


# predicts and evaluates case processed by network
def evaluate_case(model, case_data, label_data):

    # number of slices in volume
    case_z_axis = case_data.shape[-2]

    # array to store network predictions on slices
    vol_prediction = np.zeros(case_z_axis,)

    # arrays to store network scores on slices
    vol_score = np.zeros(case_z_axis,)

    # iterates through volume slices doing prediction
    for cur_slice in range(case_z_axis):

        # picks slice from volume data and corresponding ground truth slice too
        slice_data, slice_label = case_data[:, :, cur_slice, :], label_data[:, :, cur_slice, :]

        # reshapes slice_data into format which Keras based network can process
        slice_data = slice_data.reshape(1, case_data.shape[0], case_data.shape[1], 3)

        # predicts slice class using network
        slice_score = model.predict(slice_data)

        # creates label according to obtained score
        slice_predict = 1 if slice_score > 0.5 else 0

        # saves predictions and scores to arrays
        vol_prediction[cur_slice] = slice_predict

        vol_score[cur_slice] = ("{:0.2f}".format(slice_score[0][0]))

    # counts t2 lesions and enhancing lesions in volume
    vol_gad = count_tissue_in_volume(label_data[:, :, :, 0], predict_list=vol_prediction, score_list=vol_score)

    return vol_gad



def main():
    # specifies folder containing folders of image volumes
    folder_path = "Image Containing Folder Path "

    # specifies MR images file format
    file_format = ".nii.gz"

    # specifies model file name and file extension
    model_name, model_extension = "top_only", "hdf5"

    # specifies sequences considered in network evaluation
    interest_sequences = ["flair", "t1_pre", "t2"]

    # creates variables to handle data of all MRI volumes
    list_cases = []
    scores_cases = None
    gad_lesions_cases = None

    # sets to generate slice-wise scores from. "training" set contains both training and validation in this context
    sets_to_generate = ["training", "test"]

    # cleans files if previously existed
    for curr_set in sets_to_generate:
        try:
            os.remove("pos_{0}_class_3.csv".format(curr_set))
            os.remove("neg_{0}_class_3.csv".format(curr_set))
        except:
            continue

    # iterates through cases in both training and testing set to build arrays of scores to be used for patient wise prediction
    for current_set in sets_to_generate:

        # loads positive cases from list, previously partitioned
        pos_cases = load_cases_from_csv(csv_file_name="positive_{0}_list_3.csv".format(current_set))

        # loads negative cases from list, previously partitioned
        neg_cases = load_cases_from_csv(csv_file_name="negative_{0}_list_3.csv".format(current_set))

        # generates patient-wise labels for training/testing of network
        cases_classes = np.array([1] * int(len(pos_cases)) + [0] * int(len(neg_cases)))

        # complete list of positive and negative cases
        cases_list = pos_cases + neg_cases

        # loads model to perform prediction on volumes' slices
        model = load_model("{0}.{1}".format(model_name, model_extension))

        # loops through all cases in set
        for case_idx, case in enumerate(cases_list):
            print("case: {0}".format(case))
            # loads case MRI volume data along with ground truth label data
            case_data, label_data = data_extraction_from_files(dataset_path=folder_path, img_folder=case
                                                   , training_seq=interest_sequences, file_format=file_format)

            # removes few top & bottom slices which can be corrupt in some cases
            case_data, label_data = case_data[:, :, 2:42, :], label_data[:, :, 2:42, :]

            # evaluates case and obtains various data on the result
            vol_gad = evaluate_case(model=model, case_data=case_data, label_data=label_data)

            # this is a list of various data
            # Case, Network_Enhancing_lesions, Ground_Truth_Enhancing_lesions, Net_Enhacing_slices, Ground_Truth_Enhacing_slices,
            # Network_Enhancing_lesions_at_T2_lesion_slices
            # Network_slices_Scores, Network_slices_Predictions, Ground_Truth_Labels, T2_Lesion_Present_at_Slice
            vol_prediction = vol_gad[7]

            # makes list of all volumes' slice-wise predictions
            list_cases.append(int(vol_prediction))

            # saves scores and ground truth labels as well
            if case_idx == 0:
                scores_cases = vol_gad[4]
                gad_lesions_cases = vol_gad[6]
            else:
                scores_cases = np.concatenate((scores_cases, vol_gad[4]))
                gad_lesions_cases = np.concatenate((gad_lesions_cases, vol_gad[6]))

            # saves positive and negative cases results for volume-wise prediction
            if cases_classes[case_idx] == 1:
                append_to_csv(csv_file_name="pos_{0}_class_3.csv".format(current_set), list_to_save=vol_gad[4])
            else:
                append_to_csv(csv_file_name="neg_{0}_class_3.csv".format(current_set), list_to_save=vol_gad[4])



if __name__ == '__main__':
    main()
