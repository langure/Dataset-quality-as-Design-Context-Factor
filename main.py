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


if __name__ == '__main__':
    if RESAMPLE_DATA:
        resample_data()

    train_data = pd.read_csv(TRAIN_DATA_PATH)

    original_gini_index = gini_coefficient(train_data)
    print("\nOriginal Gini Index:", original_gini_index)

    #balanced_gini, balanced_index = redistribute_gini_index(train_data, target_index=0.0, change_step=20, tolerance=0.01, max_iterations=1000)
    #print("\nBalanced Gini Index:", balanced_index)
    #print("\nNew Dataframe Label Distribution, with total samples: ", balanced_gini['shared_emotion'].value_counts())
    # save the dataset to cvs
    #balanced_gini.to_csv(os.path.join(WORK_DIR, 'balanced_gini.csv'), index=False)

    #medium_gini, medium_index = redistribute_gini_index(train_data, target_index=0.5, change_step=20, tolerance=0.01, max_iterations=1000)
    #print("\nMedium Gini Index:", medium_index)
    #print("\nNew Dataframe Label Distribution, with total samples: ", medium_gini['shared_emotion'].value_counts())
    # save the dataset to cvs
    #medium_gini.to_csv(os.path.join(WORK_DIR, 'medium_gini.csv'), index=False)

    unbalanced_gini, unbalanced_index = redistribute_gini_index(train_data, target_index=0.8, change_step=10, tolerance=0.01, max_iterations=1000)
    print("\nUnbalanced Gini Index:", unbalanced_index)
    print("\nNew Dataframe Label Distribution, with total samples: ", unbalanced_gini['shared_emotion'].value_counts())
    # save the dataset to cvs
    unbalanced_gini.to_csv(os.path.join(WORK_DIR, 'unbalanced_gini.csv'), index=False)

    # write the balanced_gini, medium_gini, unbalanced_gini to a txt file
    #with open(os.path.join(WORK_DIR, 'indexes.txt'), 'w') as f:
    #    f.write(f'Original Gini Index: {original_gini_index}\n')
    #    f.write(f'Balanced Gini Index: {balanced_index}\n')
    #    f.write(f'Medium Gini Index: {medium_index}\n')
    #    f.write(f'Unbalanced Gini Index: {unbalanced_index}\n')