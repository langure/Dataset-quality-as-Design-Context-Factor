from utils import *
import os

WORK_DIR = './workdir'
TRAIN_DATA_PATH = os.path.join(WORK_DIR, 'train_dataframe.csv')
VALIDATION_DATA_PATH = os.path.join(WORK_DIR, 'validation_dataframe.csv')

RESAMPLE_DATA = False


# The parameter here is the validation_fraction, which is the proportion of data to set aside for validation
# The function returns two dataframes, one for training and one for validation. The validation set will never change. It is always the same.
# Measures have been taken to ensure that the validation set has a similar label distribution to the main dataset
def resample_data():
    print("Splitting the main data into training and validation sets...")
    # Define the proportion of data to set aside for validation
    validation_fraction = 0.15
    # me sure that the uncompressed version of the dataset is in database/soul_emotions.db:
    train_data, validation_data = split_main_data(validation_fraction=validation_fraction)

    print("Validation Set Label Distribution, with total samples: ", validation_data.shape[0])
    print(validation_data['shared_emotion'].value_counts())

    print("\nMain training Dataset Label Distribution, with total samples: ", train_data.shape[0])
    print(train_data['shared_emotion'].value_counts())

    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    validation_data.to_csv(VALIDATION_DATA_PATH, index=False)

# This is just to sample the data, divide the data in test and validation set. To actually run the models, each model has its own script
# with the corresponding name: ARI.py, CNN.py, etc.
if __name__ == '__main__':
    if RESAMPLE_DATA:
        resample_data()