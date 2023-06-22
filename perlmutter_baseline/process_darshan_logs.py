import os
import csv
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from intervaltree import IntervalTree, Interval
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

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

def get_timestep_diff(timestamp1, timestamp2):
    # Convert the timestep values into datetime objects
    base_datetime = datetime(1970, 1, 1)  # Unix epoch time
    time_point1 = base_datetime + timedelta(seconds=timestamp1)
    time_point2 = base_datetime + timedelta(seconds=timestamp2)

    # Calculate the timestep difference in seconds
    return (time_point2 - time_point1).total_seconds()

def preprocess_numpy_data(preprocessed_data_dict):
    numpy_data_dict = {}

    for key, data in preprocessed_data_dict.items():
        # Delete the second sublist
        del data[1]

        preprocessed_data = np.array(data)

        # Convert the first column value to int (except the value in the first row)
        preprocessed_data[1:, 0] = preprocessed_data[1:, 0].astype(int)

        # Convert all other columns value to float (while skipping the first row)
        preprocessed_data[1:, 1:] = preprocessed_data[1:, 1:].astype(float)

        # Add two new columns at the end
        preprocessed_data = np.hstack((preprocessed_data, np.zeros((preprocessed_data.shape[0], 2), dtype=preprocessed_data.dtype)))
        preprocessed_data[0, -2:] = 'start_time', 'end_time'
        preprocessed_data[1, -2:] = float('0.0'), float('1.0')

        # Repeat the assignment for the rest of the rows
        for i in range(2, preprocessed_data.shape[0]):
            preprocessed_data[i, -2] = float(preprocessed_data[i-1, -1])  # Assign start_time values
            preprocessed_data[i, -1] = float(preprocessed_data[i-1, -1]) + get_timestep_diff(float(preprocessed_data[i-1, 1]), float(preprocessed_data[i, 1]))  # Assign end_time values

        # Assign the preprocessed data to the dictionary with the key
        numpy_data_dict[key] = preprocessed_data

    return numpy_data_dict

def save_numpy_array_as_csv(numpy_data_dict, directory_path):
    # Create the 'client_merged_stats' directory
    target_directory = os.path.join(directory_path, 'preprocessed_client_stats')
    os.makedirs(target_directory, exist_ok=True)

    # Save the preprocessed data as CSV files
    for key, data in numpy_data_dict.items():
        # Create the file path for saving the CSV
        file_name = key + '.csv'
        file_path = os.path.join(target_directory, file_name)

        # Save the NumPy array as a CSV file
        header = ','.join(data[0].astype(str))
        np.savetxt(file_path, data[1:], delimiter=',', header=header, comments='', fmt='%s')
        print(f"CSV file saved successfully at: {file_path}")

def extract_lustre_stat(directory_path):
    client_stats_files = get_client_stats_files(directory_path)
    csv_data_dict = load_csv_files(client_stats_files)
    numpy_data_dict = preprocess_numpy_data(csv_data_dict)

    return numpy_data_dict

def parse_darshan_dxt_trace(filename):
    client_dict = dict()

    with open(filename) as f:
        lines = f.readlines()
        rank = None
        hostname = None

        for line in lines:
            if 'hostname' in line:
                rank = line.split(',')[1].split(':')[1].strip()
                hostname = line.split(',')[2].split(':')[1].strip()
                if hostname in client_dict:
                    if rank in client_dict[hostname]:
                        continue
                    else:
                        client_dict[hostname][rank] = list()
                else:
                    client_dict[hostname] = dict()
                    client_dict[hostname][rank] = list()

            if 'X_POSIX' in line:
                io_req = line.split()
                client_dict[hostname][rank].append(io_req)

    return client_dict

def extract_data(item_path, item):
    lustre_stat_dict = extract_lustre_stat(os.path.join(item_path, "lustre_stats"))

    darshan_f_path = os.path.join(item_path, item+"_darshan_logs", item+"_darshan_dxt_log_0")
    if os.path.exists(darshan_f_path):
        darshan_f_dict = parse_darshan_dxt_trace(darshan_f_path)
    else:
        darshan_f_dict = {}

    darshan_r_path = os.path.join(item_path, item+"_darshan_logs", item+"_darshan_dxt_log_1")
    if os.path.exists(darshan_r_path):
        darshan_r_dict = parse_darshan_dxt_trace(darshan_r_path)
    else:
        darshan_r_dict = {}

    return lustre_stat_dict, darshan_f_dict, darshan_r_dict

# client_app_io_dict_f ==> contains all ranks IO flush request per client
# client_app_io_dict_r ==> contains all ranks IO read request per client
def map_application_IO_per_client(client_stat_data, client_app_io_dict_f, client_app_io_dict_r):
    # Add six new columns at the end
    client_stat_data = np.hstack((client_stat_data, np.zeros((client_stat_data.shape[0], 6), dtype=client_stat_data.dtype)))
    client_stat_data[0, -6:] = 'read_req_cnt', 'write_req_cnt', 'read_req_avg_size', 'write_req_avg_size', 'read_req_bw', 'write_req_bw'

    # Create two empty Interval Trees
    w_interval_tree = IntervalTree()
    r_interval_tree = IntervalTree()

    for rank, data in client_app_io_dict_f.items():
        # Populate the Interval Tree with event time ranges and associated values
        for io_req in client_app_io_dict_f[rank]:
            value = int(io_req[5]) #length of IO request in bytes
            start_time = float(io_req[6])
            end_time = float(io_req[7])
            if start_time == end_time:
                # Handle the case where start_time and end_time are the same
                w_interval_tree[start_time:(start_time+0.0000001)] = value
            else:
                w_interval_tree[start_time:end_time] = value

    highest_end_time = 0.0
    for rank, data in client_app_io_dict_f.items():
        if len(client_app_io_dict_f[rank]) == 0:
            continue

        if float(client_app_io_dict_f[rank][-1][7]) > highest_end_time:
            highest_end_time = float(client_app_io_dict_f[rank][-1][7])

    for rank, data in client_app_io_dict_r.items():
        # Populate the Interval Tree with event time ranges and associated values
        for io_req in client_app_io_dict_r[rank]:
            value = int(io_req[5]) #length of IO request in bytes
            start_time = highest_end_time + float(io_req[6])
            end_time = highest_end_time + float(io_req[7])
            if start_time == end_time:
                # Handle the case where start_time and end_time are the same
                r_interval_tree[start_time:(start_time+0.0000001)] = value
            else:
                r_interval_tree[start_time:end_time] = value

    # Iterate through each data point in monitoring data
    for i in range(1, client_stat_data.shape[0]):
        start_time = float(client_stat_data[i, -8])
        end_time = float(client_stat_data[i, -7])

        # Query the Interval Tree for overlapping intervals
        w_overlapping_intervals = w_interval_tree[start_time:end_time]

        # Count the number of overlapping intervals
        write_req_cnt = len(w_overlapping_intervals)

        # Calculate the sum of associated values
        write_req_total_length = 0.0
        write_req_total_time = 0.0
        write_req_bw = 0.0
        if write_req_cnt != 0:
            write_req_total_length = float(sum(interval.data for interval in w_overlapping_intervals))
            write_req_total_time = float(sum(interval.end - interval.begin for interval in w_overlapping_intervals))
            write_req_bw = round(float(((write_req_total_length / write_req_total_time) / (1024.0 * 1024.0))), 2)

        write_req_avg_size = 0.0
        if write_req_cnt != 0:
            write_req_avg_size = float(write_req_total_length / write_req_cnt)

        r_overlapping_intervals = r_interval_tree[start_time:end_time]

        # Count the number of overlapping intervals
        read_req_cnt = len(r_overlapping_intervals)

        # Calculate the sum of associated values
        read_req_total_length = 0.0
        read_req_total_time = 0.0
        read_req_bw = 0.0
        if read_req_cnt != 0:
            read_req_total_length = float(sum(interval.data for interval in r_overlapping_intervals))
            read_req_total_time = float(sum(interval.end - interval.begin for interval in r_overlapping_intervals))
            read_req_bw = round(float(((read_req_total_length / read_req_total_time) / (1024.0 * 1024.0))), 2)

        read_req_avg_size = 0.0
        if read_req_cnt != 0:
            read_req_avg_size = float(read_req_total_length / read_req_cnt)

        client_stat_data[i, -6] = read_req_cnt
        client_stat_data[i, -5] = write_req_cnt
        client_stat_data[i, -4] = read_req_avg_size
        client_stat_data[i, -3] = write_req_avg_size
        client_stat_data[i, -2] = read_req_bw
        client_stat_data[i, -1] = write_req_bw

    return client_stat_data

def integrate_app_IO(item_path, item):
    lustre_stat_dict, darshan_f_dict, darshan_r_dict = extract_data(item_path, item)

    for client, data in lustre_stat_dict.items():
        if len(darshan_f_dict) == 0 and len(darshan_r_dict) == 0:
            lustre_stat_dict[client] = map_application_IO_per_client(lustre_stat_dict[client], {}, {})
        elif len(darshan_f_dict) == 0:
            lustre_stat_dict[client] = map_application_IO_per_client(lustre_stat_dict[client], {}, darshan_r_dict[client])
        elif len(darshan_r_dict) == 0:
            lustre_stat_dict[client] = map_application_IO_per_client(lustre_stat_dict[client], darshan_f_dict[client], {})
        else:
            lustre_stat_dict[client] = map_application_IO_per_client(lustre_stat_dict[client], darshan_f_dict[client], darshan_r_dict[client])

    save_numpy_array_as_csv(lustre_stat_dict, os.path.join(item_path, "lustre_stats"))

    lustre_stat_dict.clear()
    for inner_dict in darshan_f_dict.values():
        inner_dict.clear()
    darshan_f_dict.clear()
    for inner_dict in darshan_r_dict.values():
        inner_dict.clear()
    darshan_r_dict.clear()

def get_preprocessed_client_stats_files(item_path):
    target_directory = os.path.join(item_path, 'lustre_stats', 'preprocessed_client_stats')
    files = []
    for item in os.listdir(target_directory):
        files.append(os.path.join(target_directory, item))
    return files

def load_preprocessed_csv_files(client_stats_files):
    data_dict = {}
    for file_path in client_stats_files:
        # Extract the "nid005802" part from the filename
        key = file_path.split("/")[-1].split(".")[0]

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

        # Store the data in the dictionary using the key
        data_dict[key] = data

    return data_dict

def convert_to_numpy_arrays(preprocessed_data_dict):
    numpy_data_dict = {}
    for key, data in preprocessed_data_dict.items():
        numpy_data = np.array(data)
        numpy_data_dict[key] = numpy_data

    return numpy_data_dict

def extract_processed_stat(item_path):
    client_stats_files = get_preprocessed_client_stats_files(item_path)
    preprocessed_data_dict = load_preprocessed_csv_files(client_stats_files)
    numpy_data_dict = convert_to_numpy_arrays(preprocessed_data_dict)

    return numpy_data_dict

def merge_processed_client_stats(numpy_data_dict):
    new_dict = {}

    for key, data in numpy_data_dict.items():
        # Extract the key based on the first column (except the first row)
        column_1_values = data[1:, 0]
        rest_of_columns = data[1:, 1:]

        for i, value in enumerate(column_1_values):
            if value in new_dict:
                # Average the rest of the columns over the existing values
                existing_values = new_dict[value].astype(float)
                new_column = rest_of_columns[i].astype(float)

                # Check for zero values in new_column
                zero_values = new_column == 0.0

                # Divide only by non-zero values
                averaged_values = np.where(zero_values, existing_values, (existing_values + new_column) / 2.0)

                new_dict[value] = averaged_values
            else:
                # Create a new key-value pair
                new_dict[value] = rest_of_columns[i].astype(float)

    # Sort the keys and values based on the sorted keys
    sorted_keys = sorted(new_dict, key=lambda x: int(x))
    sorted_values = np.array([new_dict[key] for key in sorted_keys])

    # Create the final NumPy array
    first_column = np.array(sorted_keys)[:, np.newaxis]
    numpy_data = np.hstack((first_column, sorted_values))

    first_row = numpy_data_dict[list(numpy_data_dict.keys())[0]][0]
    numpy_array = np.vstack((first_row, numpy_data))

    return numpy_array

def save_merged_stats_as_csv(numpy_array, directory_path, file_name):
    # Create the 'client_merged_stats' directory
    target_directory = os.path.join(directory_path, 'client_merged_stats')
    os.makedirs(target_directory, exist_ok=True)

    # Create the file path for saving the CSV
    file_path = os.path.join(target_directory, file_name)

    # Save the NumPy array as a CSV file
    header = ','.join(numpy_array[0].astype(str))
    np.savetxt(file_path, numpy_array[1:], delimiter=',', header=header, comments='', fmt='%s')

    print(f"CSV file saved successfully at: {file_path}")

def generate_store_merged_data(item_path):
    numpy_data_dict = extract_processed_stat(item_path)
    numpy_array = merge_processed_client_stats(numpy_data_dict)
    save_merged_stats_as_csv(numpy_array, os.path.join(item_path, "lustre_stats"), "all_client_stats_merged.csv")

def plot_merged_data(numpy_data, target_directory, columns_to_normalize, bw_type):
    # Extract the normalized values
    normalized_data = numpy_data[1:, :].astype(float)

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
        # Create the plot
        plt.figure(figsize=(15, 6))

        x_values = normalized_data[:, columns_to_normalize[i]]
        y_values = normalized_data[:, bw_type]

        # Sort x_values and rearrange y_values accordingly
        sorted_indices = np.argsort(x_values)
        x_values = x_values[sorted_indices]
        y_values = y_values[sorted_indices]

        # Plot with points
        plt.scatter(x_values, y_values)

        # Set plot title and labels
        plt.title('merged stats comparison for: ' + numpy_data[0, bw_type] + " vs " + numpy_data[0, columns_to_normalize[i]])
        plt.xlabel('Normalized(0-1) ' + numpy_data[0, columns_to_normalize[i]])
        plt.ylabel('Normalized(0-1) ' + numpy_data[0, bw_type])

        # Save the plot as a figure
        file_name = 'merged_stats_' + numpy_data[0, bw_type] + "_vs_" + numpy_data[0, columns_to_normalize[i]] + '.png'
        file_path = os.path.join(target_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def plot_merged_data_timeline(numpy_data, target_directory, columns_to_normalize, bw_type):
    # Extract the normalized values
    normalized_data = numpy_data[1:, :].astype(float)

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
        # Create the plot
        plt.figure(figsize=(15, 6))

        # Plot with line
        plt.plot(numpy_data[1:, 0], normalized_data[:, bw_type], label=numpy_data[0, bw_type])
        plt.plot(numpy_data[1:, 0], normalized_data[:, columns_to_normalize[i]], label=numpy_data[0, columns_to_normalize[i]])

        plt.legend()

        # Set plot title and labels
        plt.title('merged stats comparison for: ' + numpy_data[0, bw_type] + " vs " + numpy_data[0, columns_to_normalize[i]])
        plt.xlabel(numpy_data[0, 0])
        plt.ylabel('Normalized Value(0-1): ' + ', '.join(numpy_data[0, [bw_type, columns_to_normalize[i]]]))

        # Configure tick frequency on the X-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust the number of ticks as desired

        # Save the plot as a figure
        file_name = 'timeline_merged_stats_' + numpy_data[0, bw_type] + "_vs_" + numpy_data[0, columns_to_normalize[i]] + '.png'
        file_path = os.path.join(target_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def plot_merged_non_normalized_data_timeline(numpy_data, target_directory, lustre_bw, app_bw):
    plt.figure(figsize=(15, 6))

    # Plot with line
    plt.plot(numpy_data[1:, 0], numpy_data[1:, app_bw].astype(float), label=numpy_data[0, app_bw])
    plt.plot(numpy_data[1:, 0], numpy_data[1:, lustre_bw].astype(float), label=numpy_data[0, lustre_bw])

    plt.legend()

    # Set plot title and labels
    plt.title('merged stats comparison for: ' + numpy_data[0, app_bw] + " vs " + numpy_data[0, lustre_bw])
    plt.xlabel(numpy_data[0, 0])
    plt.ylabel('Normalized Value(0-1): ' + ', '.join(numpy_data[0, [app_bw, lustre_bw]]))

    # Configure tick frequency on the X-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust the number of ticks as desired
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust the number of ticks as desired

    # Save the plot as a figure
    file_name = 'timeline_merged_non_normalized_stats_' + numpy_data[0, app_bw] + "_vs_" + numpy_data[0, lustre_bw] + '.png'
    file_path = os.path.join(target_directory, file_name)
    plt.savefig(file_path)
    plt.close()

def remove_png_files(target_directory):
    for filename in os.listdir(target_directory):
        if filename.endswith(".png"):
            file_path = os.path.join(target_directory, filename)
            os.remove(file_path)

def extract_plot_merged_data(item_path):
    target_directory = os.path.join(item_path, "lustre_stats", "client_merged_stats")
    remove_png_files(target_directory)
    file_path = os.path.join(target_directory, "all_client_stats_merged.csv")

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Store the data in the dictionary using the key
    numpy_data = np.array(data)

    plot_merged_data(numpy_data, target_directory, [2, 3, 4, 6, 8, 10, 12, 14], -1)
    plot_merged_data(numpy_data, target_directory, [2, 3, 4, 5, 7, 9, 11, 13], -2)
    plot_merged_data_timeline(numpy_data, target_directory, [2, 3, 4, 6, 8, 10, 12, 14], -1)
    plot_merged_data_timeline(numpy_data, target_directory, [2, 3, 4, 5, 7, 9, 11, 13], -2)
    plot_merged_non_normalized_data_timeline(numpy_data, target_directory, 6, -1)
    plot_merged_non_normalized_data_timeline(numpy_data, target_directory, 5, -2)
    print("Plot creation successful for: ", file_path)

def remove_existing_files(root_path):
    # Delete the files of 'root_path' directory
    for filename in os.listdir(root_path):
        file_path = os.path.join(root_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def derive_corr(item_path, root_path):
    target_directory = os.path.join(item_path, "lustre_stats", "client_merged_stats")
    file_path = os.path.join(target_directory, "all_client_stats_merged.csv")

    # create dataframe from file
    dataframe = pd.read_csv(file_path)

    # Drop the specified columns from the dataframe
    columns_to_drop = ["observation_no", "timestamp", "actual_record_duration", "start_time", "end_time"]
    dataframe = dataframe.drop(columns=columns_to_drop)

    # use corr() method on dataframe to make correlation matrix
    matrix = dataframe.corr()

    # Create a DataFrame for the file_path
    file_path_df = pd.DataFrame([file_path], columns=["File Path"])

    # Concatenate the file_path DataFrame with the matrix DataFrame
    merged_df = pd.concat([file_path_df, matrix])

    output_path = os.path.join(root_path, "correlation_output.csv")

    # Save the merged DataFrame as a CSV file
    merged_df.to_csv(output_path, mode='a', index=True)

    print("Matrix save successfully for: ", file_path)

def adf_test(timeseries):
    try:
        dftest = adfuller(timeseries, autolag='AIC')
        result = pd.Series(dftest[0:4], index=['Test Statistic', 'P-value', 'Lags Used', 'No of Observations'])

        for key, value in dftest[4].items():
            result['Critical Value (%s)' % key] = value

        return result

    except ValueError as e:
        print("Error occurred:", e)
        return pd.Series(dtype=object)

def kpss_test(timeseries):
    try:
        dftest = kpss(timeseries, regression='c')
        result = pd.Series(dftest[0:3], index=['Test Statistic', 'P-value', 'Lags Used'])

        for key, value in dftest[3].items():
            result['Critical Value (%s)' % key] = value

        return result

    except ValueError as e:
        print("Error occurred:", e)
        return pd.Series(dtype=object)

def perform_adf_test(item_path, root_path):
    target_directory = os.path.join(item_path, "lustre_stats", "client_merged_stats")
    file_path = os.path.join(target_directory, "all_client_stats_merged.csv")

    # create dataframe from file
    dataframe = pd.read_csv(file_path)

    # Drop the specified columns from the dataframe
    columns_to_drop = ["observation_no", "timestamp", "actual_record_duration", "start_time", "end_time"]
    dataframe = dataframe.drop(columns=columns_to_drop)
    result = dataframe.apply(adf_test, axis = 0)

     # Create a DataFrame for the file_path
    file_path_df = pd.DataFrame([file_path], columns=["File Path"])

    # Concatenate the file_path DataFrame with the matrix DataFrame
    merged_df = pd.concat([file_path_df, result])
    output_path = os.path.join(root_path, "adf_output.csv")

    # Save the merged DataFrame as a CSV file
    merged_df.to_csv(output_path, mode='a', index=True)
    print("ADF Test save successfully for: ", file_path)

def perform_kpss_test(item_path, root_path):
    target_directory = os.path.join(item_path, "lustre_stats", "client_merged_stats")
    file_path = os.path.join(target_directory, "all_client_stats_merged.csv")

    # create dataframe from file
    dataframe = pd.read_csv(file_path)

    # Drop the specified columns from the dataframe
    columns_to_drop = ["observation_no", "timestamp", "actual_record_duration", "start_time", "end_time"]
    dataframe = dataframe.drop(columns=columns_to_drop)
    result = dataframe.apply(kpss_test, axis = 0)

     # Create a DataFrame for the file_path
    file_path_df = pd.DataFrame([file_path], columns=["File Path"])

    # Concatenate the file_path DataFrame with the matrix DataFrame
    merged_df = pd.concat([file_path_df, result])
    output_path = os.path.join(root_path, "kpss_output.csv")

    # Save the merged DataFrame as a CSV file
    merged_df.to_csv(output_path, mode='a', index=True)
    print("KPSS Test save successfully for: ", file_path)

def traverse_and_plot(root_path):
    # remove_existing_files(root_path)

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
                                        # integrate_app_IO(item_path, item)
                                        # generate_store_merged_data(item_path)
                                        extract_plot_merged_data(item_path)
                                        # derive_corr(item_path, root_path)
                                        # perform_adf_test(item_path, root_path)
                                        # perform_kpss_test(item_path, root_path)

if __name__ == "__main__":
    # Example usage:
    root_path = "/Users/mrashid2/DirLab/perlmutter_test_data/expr_subset_8_tests/node/"
    traverse_and_plot(root_path)