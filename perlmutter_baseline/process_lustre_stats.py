import os
import csv
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

def get_client_stats_files(directory):
    file_pattern = directory + "/*client_stats_sheet.csv"
    files = glob.glob(file_pattern)
    return files

def load_csv_files(client_stats_files):
    data_dict = {}
    for file_path in client_stats_files:
        # Extract the "nid005802" part from the filename
        key = file_path.split("/")[-1].split("_")[15]

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

        # Store the data in the dictionary using the key
        data_dict[key] = data

    return data_dict

def preprocess_csv_data(csv_data_dict):
    for key, data in csv_data_dict.items():
        # Delete the second sublist
        del data[1]

        # Delete sublists where the fourth value is zero
        data = [sublist for sublist in data if sublist[3] != '0']

        # Update the modified data in the dictionary
        csv_data_dict[key] = data

    return csv_data_dict

def save_preprocessed_data(preprocessed_data_dict, directory_path):
    # Create the 'preprocessed_client_stats' directory
    target_directory = os.path.join(directory_path, 'preprocessed_client_stats')
    os.makedirs(target_directory, exist_ok=True)

    # Save the preprocessed data as CSV files
    for key, data in preprocessed_data_dict.items():
        file_name = key + '.csv'
        file_path = os.path.join(target_directory, file_name)

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

def convert_to_numpy_arrays(preprocessed_data_dict):
    numpy_data_dict = {}
    for key, data in preprocessed_data_dict.items():
        numpy_data = np.array(data)
        numpy_data_dict[key] = numpy_data

    return numpy_data_dict

def preprocess_numpy_data(numpy_data_dict):
    preprocessed_data_dict = {}
    for key, data in numpy_data_dict.items():
        preprocessed_data = np.copy(data)

        # Convert the first column value to int (except the value in the first row)
        preprocessed_data[1:, 0] = preprocessed_data[1:, 0].astype(int)

        # Convert all other columns value to float (while skipping the first row)
        preprocessed_data[1:, 1:] = preprocessed_data[1:, 1:].astype(float)

        # Assign the preprocessed data to the dictionary with the key
        preprocessed_data_dict[key] = preprocessed_data

    return preprocessed_data_dict

def remove_files(target_directory):
    # Delete the contents of 'client_stat_figures' directory
    for filename in os.listdir(target_directory):
        file_path = os.path.join(target_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def normalize_and_plot(preprocessed_data_dict, target_directory, columns_to_normalize, x_axis_index):
    for key, data in preprocessed_data_dict.items():
        # Extract the normalized values
        normalized_data = data[1:, columns_to_normalize].astype(float)
        
        # Check if the array is empty
        if normalized_data.size == 0:
            continue  # Skip normalization for empty arrays

        # Filter out rows with negative values in any column
        row_mask = np.any(normalized_data < 0, axis=1)
        normalized_data = normalized_data[~row_mask]

        min_vals = normalized_data.min(axis=0)
        max_vals = normalized_data.max(axis=0)

        # Check for zero ranges in columns
        zero_range_columns = (max_vals - min_vals) == 0.0

        # Adjust the normalization range for zero-range columns
        max_vals[zero_range_columns] = min_vals[zero_range_columns] + 1.0
        normalized_data = (normalized_data - min_vals) / (max_vals - min_vals)

        for i in range(len(columns_to_normalize)):
            if i == x_axis_index:
                continue
            
            # Create the plot
            plt.figure(figsize=(15, 6))

            x_values = normalized_data[:, x_axis_index]
            y_values = normalized_data[:, i]

            # Sort x_values and rearrange y_values accordingly
            sorted_indices = np.argsort(x_values)
            x_values = x_values[sorted_indices]
            y_values = y_values[sorted_indices]

            # Plot the line
            plt.plot(x_values, y_values)

            # Set plot title and labels
            plt.title(key + ' different stats comparison for: ' + data[0, columns_to_normalize[x_axis_index]] + " vs " + data[0, columns_to_normalize[i]])
            plt.xlabel('Normalized(0-1) ' + data[0, columns_to_normalize[x_axis_index]])
            plt.ylabel('Normalized(0-1) ' + data[0, columns_to_normalize[i]])

            # Save the plot as a figure
            file_name = key + '_' + data[0, columns_to_normalize[x_axis_index]] + "_vs_" + data[0, columns_to_normalize[i]] + '.png'
            file_path = os.path.join(target_directory, file_name)
            plt.savefig(file_path)
            plt.close()

def create_plot(directory_path):
    # print("Plot creation started for: ", directory_path)
    client_stats_files = get_client_stats_files(directory_path)
    csv_data_dict = load_csv_files(client_stats_files)
    preprocessed_data_dict = preprocess_csv_data(csv_data_dict)
    # save_preprocessed_data(preprocessed_data_dict, directory_path)
    
    numpy_data_dict = convert_to_numpy_arrays(preprocessed_data_dict)
    preprocessed_data_dict = preprocess_numpy_data(numpy_data_dict)

    # Create the 'client_stat_figures' directory
    # target_directory = os.path.join(directory_path, 'client_stat_figures')
    # os.makedirs(target_directory, exist_ok=True)
    # remove_files(target_directory)

    # normalize_and_plot(preprocessed_data_dict, target_directory, [2, 3, 4, 5, 7, 9, 11, 13], 3)
    # normalize_and_plot(preprocessed_data_dict, target_directory, [2, 3, 4, 5, 7, 9, 11, 13], 4)
    # normalize_and_plot(preprocessed_data_dict, target_directory, [2, 3, 4, 6, 8, 10, 12, 14], 3)
    # normalize_and_plot(preprocessed_data_dict, target_directory, [2, 3, 4, 6, 8, 10, 12, 14], 4)
    # print("Plot creation successful for: ", target_directory)

    return preprocessed_data_dict

def merge_processed_data(preprocessed_data_dict):
    new_dict = {}

    for key, data in preprocessed_data_dict.items():
        # Extract the key based on the first column (except the first row)
        column_1_values = data[1:, 0]
        rest_of_columns = data[1:, 1:]
        
        for i, value in enumerate(column_1_values):
            if value in new_dict:
                # Check if any value in rest_of_columns[i] is negative
                if np.any(np.less(rest_of_columns[i].astype(float), 0)):
                    continue  # Skip averaging for negative values
                
                # Average the rest of the columns over the existing values
                existing_values = new_dict[value].astype(float)
                averaged_values = (existing_values + rest_of_columns[i].astype(float)) / 2.0
                new_dict[value] = averaged_values
            else:
                # Check if any value in rest_of_columns[i] is negative
                if np.any(np.less(rest_of_columns[i].astype(float), 0)):
                    continue  # Skip averaging for negative values
                
                # Create a new key-value pair
                new_dict[value] = rest_of_columns[i].astype(float)

   # Sort the keys and values based on the sorted keys
    sorted_keys = sorted(new_dict, key=lambda x: int(x))
    sorted_values = np.array([new_dict[key] for key in sorted_keys])

     # Create the final NumPy array
    first_column = np.array(sorted_keys)[:, np.newaxis]
    numpy_data = np.hstack((first_column, sorted_values))
    
    first_row = preprocessed_data_dict[list(preprocessed_data_dict.keys())[0]][0]
    numpy_array = np.vstack((first_row, numpy_data))

    return numpy_array

def save_numpy_array_as_csv(numpy_array, directory_path, file_name):
    # Create the 'client_merged_stats' directory
    target_directory = os.path.join(directory_path, 'client_merged_stats')
    os.makedirs(target_directory, exist_ok=True)

    # Create the file path for saving the CSV
    file_path = os.path.join(target_directory, file_name)

    # Save the NumPy array as a CSV file
    header = ','.join(numpy_array[0].astype(str))
    np.savetxt(file_path, numpy_array[1:], delimiter=',', header=header, comments='', fmt='%s')

    print(f"CSV file saved successfully at: {file_path}")

def plot_merged_data(numpy_data, directory_path, columns_to_normalize, x_axis_index):
    # Create the 'client_merged_stats' directory
    target_directory = os.path.join(directory_path, 'client_merged_stats')
    os.makedirs(target_directory, exist_ok=True)

    # Extract the normalized values
    normalized_data = numpy_data[1:, columns_to_normalize].astype(float)
    
    # Check if the array is empty
    if normalized_data.size == 0:
        return  # Skip normalization for empty arrays

    min_vals = normalized_data.min(axis=0)
    max_vals = normalized_data.max(axis=0)

    # Check for zero ranges in columns
    zero_range_columns = (max_vals - min_vals) == 0.0

    # Adjust the normalization range for zero-range columns
    max_vals[zero_range_columns] = min_vals[zero_range_columns] + 1.0
    normalized_data = (normalized_data - min_vals) / (max_vals - min_vals)

    for i in range(len(columns_to_normalize)):
        if i == x_axis_index:
            continue
        
        # Create the plot
        plt.figure(figsize=(15, 6))

        x_values = normalized_data[:, x_axis_index]
        y_values = normalized_data[:, i]

        # Sort x_values and rearrange y_values accordingly
        sorted_indices = np.argsort(x_values)
        x_values = x_values[sorted_indices]
        y_values = y_values[sorted_indices]

        # Plot the line
        plt.plot(x_values, y_values)

        # Set plot title and labels
        plt.title('merged stats comparison for: ' + numpy_data[0, columns_to_normalize[x_axis_index]] + " vs " + numpy_data[0, columns_to_normalize[i]])
        plt.xlabel('Normalized(0-1) ' + numpy_data[0, columns_to_normalize[x_axis_index]])
        plt.ylabel('Normalized(0-1) ' + numpy_data[0, columns_to_normalize[i]])

        # Save the plot as a figure
        file_name = 'merged_stats_' + numpy_data[0, columns_to_normalize[x_axis_index]] + "_vs_" + numpy_data[0, columns_to_normalize[i]] + '.png'
        file_path = os.path.join(target_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def create_merged_plot(directory_path, preprocessed_data_dict):
    print("Plot creation started for: ", directory_path)
    numpy_data = merge_processed_data(preprocessed_data_dict)
    save_numpy_array_as_csv(numpy_data, directory_path, "all_client_stats_merged.csv")

    plot_merged_data(numpy_data, directory_path, [2, 3, 4, 5, 7, 9, 11, 13], 3)
    plot_merged_data(numpy_data, directory_path, [2, 3, 4, 5, 7, 9, 11, 13], 4)
    plot_merged_data(numpy_data, directory_path, [2, 3, 4, 6, 8, 10, 12, 14], 3)
    plot_merged_data(numpy_data, directory_path, [2, 3, 4, 6, 8, 10, 12, 14], 4)
    print("Plot creation successful for: ", directory_path)

def traverse_and_plot(root_path):
    for node_type in ["cpu"]:
        core_cnt="4"
        if node_type == "cpu":
            core_cnt="64"
        
        for node_cnt in ["8"]:
            for io_burst_size in ["262144k"]:
                for stripe_format in ["large", "small"]:
                    for buf_size in ["33554432", "1048576"]:
                        for aggr_cnt in ["16", "1"]:
                            for itrn_cnt in ["1", "2", "3", "4", "5"]:
                                directory_path = root_path + node_type + "_" + node_cnt + "/core_" + core_cnt + "/io_burst_" + io_burst_size + "/stripe_" + stripe_format + "/buf_size_" + buf_size + "/aggr_" + aggr_cnt + "/itrn_" + itrn_cnt
                                for item in os.listdir(directory_path):
                                    item_path = os.path.join(directory_path, item)
                                    if os.path.isdir(item_path):
                                        preprocessed_data_dict = create_plot(os.path.join(item_path, "lustre_stats"))
                                        create_merged_plot(os.path.join(item_path, "lustre_stats"), preprocessed_data_dict)

if __name__ == "__main__":
    # Example usage:
    root_path = "/Users/mrashid2/DirLab/perlmutter_test_data/expr_subset_8_tests/node/"
    directory_path = "/Users/mrashid2/DirLab/perlmutter_test_data/expr_subset_8_tests/node/cpu_8/core_64/io_burst_262144k/stripe_large/buf_size_1048576/aggr_1/itrn_1/9951951/lustre_stats"
    traverse_and_plot(root_path)
    