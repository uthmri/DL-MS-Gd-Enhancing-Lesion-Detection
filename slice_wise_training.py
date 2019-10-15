import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, SGD
from sklearn.model_selection import KFold
from slice_wise_testing import check_model
from keras.utils.training_utils import multi_gpu_model



# class to train network model on multiple GPUs
# and also save this model using the ModelCheckpoint function
class ModelMGPU(Model):
    # defines new class based on Keras own class for multi-gpu training
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    # allows saving or loading custom Model class
    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def top_model_definition(base_model_output_shape):
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model_output_shape))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    return top_model


# trains top of VGG16 on training data, validation is done at the end of each epoch on unobserved data
def train_top_only(train_data, train_labels, val_data, val_labels, epochs, batch_size, data_input_shape, model_name):

    # creates VGG16 base architecture based on previously imagenet optimized weights, dense layers are not included
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=data_input_shape)

    # Sets network layers to not trainable (freezes layers' weights)
    for layer in model.layers:
        layer.trainable = False

    # creates network top model (dense layers) to train on new data based on VGG16 bottleneck features
    top_model = top_model_definition(base_model_output_shape=model.output_shape[1:])


    # joins VGG16 base model with custom top model to create a single model
    model = Model(inputs=model.input, outputs=top_model(model.output))

    # prints summary of model created
    model.summary()

    # calls checkpint function to save model for which lowest validation loss is obtained
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',
                                       verbose=1, save_best_only=True)


    # logs model training metrics for further analysis
    csv_logger = CSVLogger("log_transfer.csv", append=True, separator=';')

    # creates custom model for conducting training on multiple GPUs
    gpu_model = ModelMGPU(model, gpus=2)

    # compiles model for binary classification optimization
    #
    gpu_model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # sets data augmentation variables to improve network performance on validation data
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=5,
        zoom_range=0.2,
        width_shift_range=5,
        height_shift_range=5
    )

    # creates data generator object for network performance assessment on validation data
    validation_datagen = ImageDataGenerator()

    # defines Keras specific generator for training data
    train_generator = train_datagen.flow(
        train_data, y=train_labels,
        batch_size=batch_size)

    # defines Keras specific generator for validation data
    validation_generator = validation_datagen.flow(
        val_data, y=val_labels,
        batch_size=batch_size)

    # fits models based on the defined data generator
    # steps are set upon batch size
    # number of workers defines the processes feeding GPUs with data from CPUs
    gpu_model.fit_generator(
        train_generator,
        steps_per_epoch=train_data.shape[0] // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=val_data.shape[0] // batch_size,
        callbacks=[model_checkpoint, csv_logger], verbose=2, workers=16)


# unfreezes few last convolutional layers of VGG16 for fine-tuning on new data
def fine_tune_top(top_model_file, train_data, train_labels, val_data, val_labels, batch_size, epochs, net_input_shape, fine_tune_model_file):

    # creates VGG16 model under similar conditions as in training top only
    fine_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=net_input_shape)

    # creates network top model (dense layers) to train on new data based on VGG16 bottleneck features
    top_model = top_model_definition(base_model_output_shape=fine_model.output_shape[1:])


    # joins VGG16 model with top model and loads weights previously optimized
    fine_model = Model(inputs=fine_model.input, outputs=top_model(fine_model.output))
    fine_model.load_weights(top_model_file)

    # unfreezes few of VGG16 convolutional layers for fine-tuning using previously trained weights
    for layer in fine_model.layers[:len(fine_model.layers) - 5]:
        layer.trainable = False

    # prints summary of network layers
    fine_model.summary()

    # compiles model for binary classification optimization
    #
    fine_model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # sets data augmentation variables to improve network performance on validation data
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        zoom_range=0.25,
        width_shift_range=10,
        height_shift_range=10
    )

    # creates data generator object for network performance assessment on validation data
    validation_datagen = ImageDataGenerator()

    # defines Keras specific generator for training data
    train_generator = train_datagen.flow(
        train_data, y=train_labels,
        batch_size=batch_size)

    # defines Keras specific generator for validation data
    validation_generator = validation_datagen.flow(
        val_data, y=val_labels,
        batch_size=batch_size)

    # calls checkpint function to save model for which lowest validation loss is obtained
    model_checkpoint = ModelCheckpoint(fine_tune_model_file, monitor='val_loss',
                                       verbose=1, save_best_only=True)

    # logs model training metrics for further analysis
    csv_logger = CSVLogger("fine_log.csv", append=True, separator=';')

    # fits models based on the defined data generator
    # steps are set upon batch size
    # number of workers defines the processes feeding GPUs with data from CPUs
    fine_model.fit_generator(
        train_generator,
        steps_per_epoch=train_data.shape[0] // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=val_data.shape[0] // batch_size,
        callbacks=[model_checkpoint, csv_logger], verbose=2)


def main():

    # K-fold variable to define the number of splits for data considered
    kf = KFold(n_splits=5)
    # Batch size for network training
    batch_size = 16
    # Numbers of epochs to be used at each stage of training
    epochs_top_train, epochs_fine_tune = 50, 100
    # Input shape to the network, image dimensions and number of channels
    input_shape = (256, 256, 3)

    train_array_pos, train_array_neg = None, None
    test_array_pos, test_array_neg = None, None


    # Input data are arrays of slices containing both positive ("enhancing slices") from patients presenting enhancement
    # and non-enhancing slices from non-enhancing patients.
    train_data_pos = train_array_pos
    train_data_neg = train_array_neg


    # Variable to track different stages of cross-validation
    cnt_chk = 1

    for train_idx, test_idx in kf.split(train_data_pos):

        # model definition
        base_model = 'models/model_top_{0}.hdf5'.format(cnt_chk)
        fine_tune_model = 'models/model_slice_wise_{0}.hdf5'.format(cnt_chk)


        # partitions training data for cross validation
        # pos_train being the positive slices and neg_train the negative slices
        pos_train = train_data_pos[train_idx]
        neg_train = train_data_neg[train_idx]

        # partitions validation data for cross validation
        # pos_train being the positive slices and neg_train the negative slices
        pos_val = train_data_pos[test_idx]
        neg_val = train_data_neg[test_idx]


        # creates labels 0/1 for specifying positive and negative cases
        train_labels = np.array([1] * int(len(pos_train)) + [0] * int(len(neg_train)))
        val_labels = np.array([1] * int(len(pos_val)) + [0] * int(len(neg_val)))



        train_data = np.concatenate((pos_train, neg_train))
        val_data = np.concatenate((pos_val, neg_val))



        # trains top of VGG16 on training data
        train_top_only(train_data=train_data, train_labels=train_labels, val_data=val_data
                       , val_labels=val_labels, epochs=epochs_top_train, data_input_shape=input_shape
                       , model_name=base_model, batch_size=batch_size)

        # unfreezes few last convolutional layers of VGG16 for fine-tuning on new data
        fine_tune_top(top_model_file=base_model, train_data=train_data, train_labels=train_labels
                      , val_data=val_data, val_labels=val_labels, batch_size=batch_size
                      , epochs=epochs_fine_tune, net_input_shape=input_shape, fine_tune_model_file=fine_tune_model)



        test_pos_data, test_neg_data = test_array_pos, test_array_neg

        # creates labels for testing set
        test_labels = np.array([1] * int(len(test_pos_data)) + [0] * int(len(test_neg_data)))

        test_data = np.concatenate((test_pos_data, test_neg_data))

        # evaluates network performance on testing dataset
        accuracy, sensitivity, specificity, ppv, npv = check_model(fine_tune_model, test_data, test_labels)

        # network performance evaluation on testing data
        print("Accuracy: {0} Sensitivity: {1} Specificity: {2} PPV: {3} NPV: {4}".format(accuracy,
                                                                                         sensitivity,
                                                                                         specificity, ppv, npv))

        break




if __name__ == '__main__':
    main()
