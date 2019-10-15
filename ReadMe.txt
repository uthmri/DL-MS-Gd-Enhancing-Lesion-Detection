To perform gadolinium classification, six files are included.


slice_wise_training conducts training and testing using arrays of slices selected as mentioned in the paper.
slice_wise_testing contains a function to evaluate model performance which is called after training has finished.

slice_wise_to_volume_wise runs the 2d network on all available volumes' slices and generates scores to do volume wise training and testing.
volume_wise_training uses 2D network's scores to train a dense layer to perform volume-wise prediction.
volume_wise_testing performs evaluation of volume-wise network performance.

mri_functions contains auxiliar functions used on the other scripts.