import numpy as np
from math import floor
import keras.optimizers as opt
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense,
                          Flatten, Dropout, Conv3D)
from mri_functions import load_row_from_csv, load_cases_from_csv



# function which creates dense model for volume-wise predictions
def model_definition():

    model = Sequential()
    model.add(Dense(units=64, input_shape=(40,)))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation(activation='sigmoid'))

    return model


def main():

    epochs = 100

    # loads positive and negative cases in list
    pos_cases = load_cases_from_csv(csv_file_name="positive_training_list_3.csv")
    neg_cases = load_cases_from_csv(csv_file_name="negative_training_list_3.csv")


    # breaks training set into a smaller training set and a validation set for volume-wise training
    pos_train_cases, neg_train_cases = int(floor(len(pos_cases)*0.8)), int(floor(len(neg_cases)*0.8))
    pos_val_cases, neg_val_cases = int(len(pos_cases) - int(floor(len(pos_cases)*0.8)))\
        , int(len(neg_cases) - int(floor(len(neg_cases)*0.8)))

    # data arrays initialization
    train_data = np.zeros(shape=(pos_train_cases + neg_train_cases, 40))
    val_data = np.zeros(shape=(pos_val_cases + neg_val_cases, 40))


    # creates arrays of training and validation data from positive cases
    for case_idx, case in enumerate(pos_cases):

        row = load_row_from_csv(csv_file_name="pos_training_class_3.csv", row_id=case_idx)

        #print("Size array: {0}".format(len(row)))

        if case_idx < pos_train_cases:
            if len(row) < 40:
                num_entry = 40 - len(row)
                for number_e in range(num_entry):
                    row.append(0)

            elif len(row) > 40:
                num_entry = len(row) - 40
                for number_e in range(num_entry):
                    row.pop(0)

            train_data[case_idx, :] = row

        if case_idx >= pos_train_cases:
            if len(row) < 40:
                num_entry = 40 - len(row)
                for number_e in range(num_entry):
                    row.append(0)
            elif len(row) > 40:
                num_entry = len(row) - 40
                for number_e in range(num_entry):
                    row.pop(0)

            val_data[case_idx - pos_train_cases, :] = row

    # creates arrays of training and validation data from negative cases
    for neg_case_idx, case in enumerate(neg_cases):
        print("NegCase: {0}".format(neg_case_idx))

        row = load_row_from_csv(csv_file_name="neg_training_class_3.csv", row_id=neg_case_idx)
        if neg_case_idx < neg_train_cases:
            if len(row) < 40:
                num_entry = 40 - len(row)
                for number_e in range(num_entry):
                    row.append(0)
            elif len(row) > 40:
                num_entry = len(row) - 40
                for number_e in range(num_entry):
                    row.pop(0)

            train_data[neg_case_idx + pos_train_cases, :] = row

        if neg_case_idx >= neg_train_cases:
            if len(row) < 40:
                num_entry = 40 - len(row)
                for number_e in range(num_entry):
                    row.append(0)
            elif len(row) > 40:
                num_entry = len(row) - 40
                for number_e in range(num_entry):
                    row.pop(0)

            val_data[neg_case_idx - neg_train_cases + pos_val_cases, :] = row


    # creates volume-wise labels for volumes available
    train_labels = np.array([1] * pos_train_cases + [0] * neg_train_cases)
    val_labels = np.array([1] * pos_val_cases + [0] * neg_val_cases)


    for train_idx in range(train_data.shape[0]):
        print(train_data[train_idx, :])

    for val_idx in range(val_data.shape[0]):
        print(val_data[val_idx, :])


    # checkpoints model and saves the one that reaches lowest loss on unobserved data
    model_checkpoint = ModelCheckpoint("model_volume_wise.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

    # creates model for training
    model = model_definition()

    # compiles model
    model.compile(optimizer=opt.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # trains volume-wise model
    model.fit(x=train_data, y=train_labels, epochs=epochs
               , validation_data=(val_data, val_labels), callbacks=[model_checkpoint], shuffle=True)



if __name__ == '__main__':
    main()
