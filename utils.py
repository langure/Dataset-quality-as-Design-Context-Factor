import pandas as pd
import sqlite3
import numpy as np
from tqdm import tqdm


def split_main_data(validation_fraction=0.15):
    # Connect to the SQLite database
    conn = sqlite3.connect('database/soul_emotions.db')

    # Read data from the database into a pandas DataFrame
    query = "SELECT id, text, shared_emotion FROM soul_emotions WHERE shared_emotion IS NOT NULL;"
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Define the proportion of data to set aside for validation
    validation_fraction = 0.15

    # Get unique shared_emotion values and their counts
    label_distribution = df['shared_emotion'].value_counts()

    # Calculate the number of samples to take for validation for each label
    validation_samples = {label: int(validation_fraction * count) for label, count in label_distribution.items()}

    # Select validation samples randomly for each label
    validation_data = pd.DataFrame()
    for label, count in validation_samples.items():
        label_data = df[df['shared_emotion'] == label]
        validation_data = pd.concat([validation_data, label_data.sample(count, random_state=42)])

    # Remove validation samples from the original dataframe
    train_data = df.drop(validation_data.index)

    return train_data, validation_data


def gini_coefficient(df):
    """Calculate the Gini coefficient for the distribution of 'shared_emotion' labels in the DataFrame."""
    # Get label counts
    label_counts = df['shared_emotion'].value_counts().sort_values()

    # Convert label counts to numpy array
    array = np.array(label_counts)

    # Flatten array
    array = array.flatten()

    # All values are treated equally, arrays must be 1d:
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    gini_index = ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    return gini_index


def redistribute_gini_index(df, target_index, change_step, tolerance, max_iterations):
    """Redistribute the dataframe to achieve the target Gini index."""
    try:
        # Step 1: Create a perfectly balanced dataframe
        label_counts = df['shared_emotion'].value_counts()
        min_label_count = label_counts.min()
        balanced_df = pd.concat(
            [df[df['shared_emotion'] == label].sample(min_label_count) for label in label_counts.index])

        # Step 2: Calculate the initial Gini index
        b_label_counts = balanced_df['shared_emotion'].value_counts()
        current_index = gini_coefficient(balanced_df)

        # Check if the initial index matches the target index
        if abs(current_index - target_index) < tolerance:
            return balanced_df, current_index

        original_data_minus_balanced = df.drop(balanced_df.index)

        # Step 3: Define label order for removal
        label_order = list(label_counts.index)

        # Step 4: Iterate for redistribution
        ptr = 1

        progress_bar = tqdm(total=max_iterations, desc=f"Gini Index: {current_index:.4f}")
        for iteration in range(max_iterations):
            accumulator = ptr
            # Traverse label order
            for label in label_order:
                if accumulator == 0:
                    break
                #remove as many samples as the change step * accumulator from the label
                remove_count = min(label_counts[label], change_step)
                # remove samples from the balanced dataframe from this label, as many as remove_count
                subset_length_before = len(balanced_df[balanced_df['shared_emotion'] == label])
                b_label_counts = balanced_df['shared_emotion'].value_counts()

                to_remove = balanced_df[balanced_df['shared_emotion'] == label].head(remove_count).index
                balanced_df = balanced_df.drop(to_remove)
                subset_length_after = len(balanced_df[balanced_df['shared_emotion'] == label])
                b_label_counts = balanced_df['shared_emotion'].value_counts()
                accumulator -= 1
            # from the original_data_minus_balanced dataframe, make a new dataframe that contains the three most frequent labels
            three_most_frequent_labels = b_label_counts.nlargest(3).index
            for label in three_most_frequent_labels:
                # add samples from the original_data_minus_balanced dataframe to the balanced dataframe
                to_add = original_data_minus_balanced[original_data_minus_balanced['shared_emotion'] == label].head(change_step)
                balanced_df = pd.concat([balanced_df, to_add])
                original_data_minus_balanced = original_data_minus_balanced.drop(to_add.index)



            # Recalculate Gini index
            b_label_counts = balanced_df['shared_emotion'].value_counts()
            current_index = gini_coefficient(balanced_df)

            # Check if within tolerance
            if abs(current_index - target_index) < tolerance:
                progress_bar.close()
                return balanced_df, current_index

            current_diff = abs(current_index - target_index)

            # Update progress bar
            progress_bar.set_description(f"Gini Index: {current_index:.4f}")
            progress_bar.update(1)
            ptr += 1

        # Close progress bar if no convergence
        progress_bar.close()
        return balanced_df, current_index

    except KeyError:
        print("Error: The 'shared_emotion' column does not exist in the dataframe.")
        return None, None
