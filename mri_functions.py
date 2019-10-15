import os
import csv
import random
import pickle
import numpy as np
import nibabel as nib


# formats labels arrays to integrate original ground truth with gadolinium file
def format_final_label(image_resol, objective_slice_num, validated_array, gad_array, values):

    # creates buffer to save modified label array
    final_lbl = np.zeros((image_resol[0], image_resol[1], image_resol[2], 1))

    # loads ground truth array, truncates to image resolution
    validated_array = validated_array[:, :, 0:image_resol[2]]

    # keeps desired labels in ground truth
    labels_mask = np.isin(validated_array, values)
    validated_array *= labels_mask

    # sets T2-lesion tissues to value 1
    validated_array[validated_array != 0] = 1

    # sets enhacing tissues to value 2
    gad_array[gad_array != 0] = 2

    # truncates gad_array to image resolution
    gad_array = gad_array[:, :, 0:image_resol[2]]

    # builds a single ground truth combining both data arrays
    buff_arr = np.where(gad_array == 0, validated_array, gad_array)

    # unifies previous modifications in single array
    final_lbl[:, :, 0:image_resol[2], 0] = buff_arr

    return final_lbl


# loads Nifti files and creates image and labels arrays
def data_extraction_from_files(dataset_path, img_folder, training_seq=['flair', 't1_pre', 't2'],
                               validation_seq=["validated", "gad"], objective_slice_num=44
                               , image_res=(256, 256, 50), values=[4], file_format="nii", all_values=[1, 2, 3, 4]):

    gad_list = []
    final_img = np.zeros((image_res[0], image_res[1], image_res[2], len(training_seq)))

    for sequence_name in sorted(training_seq):

        # each of the subsequent blocks of this type loads data from file, truncates to the desired image resolution, and then adds it to
        # an array either containing MRI images or ground truth labels
        if sequence_name == 'flair':
            # loads file volume data into a numpy array
            current_img = nib.load(dataset_path + "/" + img_folder + "/" + sequence_name + file_format)
            current_img_data = current_img.get_data()
            # limits array to 3 axes (in case data is corrupt)
            current_img_data = current_img_data[:, :, :]

            # truncates axis to general resolution
            if current_img_data.shape[2] > objective_slice_num:
                current_img_data = current_img_data[:, :, 0:objective_slice_num]

            # saves sequence array to first axis of array containing all sequences
            final_img[:, :, 0:current_img_data.shape[2], 0] = current_img_data


        #
        elif sequence_name == 't1_pre':
            current_img = nib.load(dataset_path + "/" + img_folder + "/" + sequence_name + file_format)
            current_img_data = current_img.get_data()
            current_img_data = current_img_data[:, :, :]

            if current_img_data.shape[2] > objective_slice_num:
                current_img_data = current_img_data[:, :, 0:objective_slice_num]

            final_img[:, :, 0:current_img_data.shape[2], 1] = current_img_data


        elif sequence_name == 't2':
            current_img = nib.load(dataset_path + "/" + img_folder + "/" + sequence_name + file_format)
            current_img_data = current_img.get_data()
            current_img_data = current_img_data[:, :, :]

            if current_img_data.shape[2] > objective_slice_num:
                current_img_data = current_img_data[:, :, 0:objective_slice_num]

            final_img[:, :, 0:current_img_data.shape[2], 2] = current_img_data

    final_img = final_img[:, :, 0:current_img_data.shape[2], :]




    final_lbl = None
    current_lbl_data = None
    current_gad_data = None

    # similar as above but this block process ground truth label data from files
    for label_name in sorted(validation_seq):
        if label_name == 'validated':
            current_lbl = nib.load(dataset_path + "/" + img_folder + "/" + label_name + file_format)
            current_lbl_data = current_lbl.get_data()


            current_lbl_data = current_lbl_data[:, :, :]
            val_mask = np.isin(current_lbl_data, all_values)
            current_lbl_data = current_lbl_data * val_mask

            if current_lbl_data.shape[2] > objective_slice_num:
                current_lbl_data = current_lbl_data[:, :, 0:objective_slice_num]


            final_lbl = current_lbl_data

        if label_name == 'gad':
            current_gad = nib.load(dataset_path + "/" + img_folder + "/" + label_name + file_format)
            current_gad_data = current_gad.get_data()
            current_gad_data = current_gad_data[:, :, :]

    gad_list.append(img_folder)


    # depending on the number of label files considered
    if len(validation_seq) != 1:
        final_lbl = format_final_label(current_lbl_data.shape, objective_slice_num, current_lbl_data, current_gad_data,
                                       values)
    else:
        labels_mask = np.isin(final_lbl, values)
        final_lbl *= labels_mask


    final_img /= np.amax(final_img)

    return final_img, final_lbl

# loads rows of case IDS into a list
def load_cases_from_csv(csv_file_name):

    baseline = []

    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            baseline.append(row[0])

    baseline.sort()

    return baseline

# loads a single row of case ID into a list
def load_row_from_csv(csv_file_name, row_id):

    row_back = None

    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row_idx, row in enumerate(csv_reader):
            if row_idx == row_id:
                row_back = row

    return row_back


# returns list with subdirectories in a folder
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



def main():
    pass



if __name__ == '__main__':
    main()

