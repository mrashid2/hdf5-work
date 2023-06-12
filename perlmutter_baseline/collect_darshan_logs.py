import os
import re
import subprocess

def process_files(directory):
    file_dict = {}
    pattern = '^(\w+)id(\d+)-(\w+)-(\w+)-(\w+)-(\w+)\.(\w+)'

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            match = re.search(pattern, filename)
            if match:
                key = match.group(2)
                if key in file_dict:
                    file_dict[key].append(filename)
                else:
                    file_dict[key] = [filename]

    return file_dict

def execute_shell_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return_code = process.returncode
    return stdout.decode('utf-8'), stderr.decode('utf-8'), return_code

def construct_shell_command(file_dict):
    for key, file_list in file_dict.items():
        command = f"mkdir -p {key}_darshan_logs"
        stdout, stderr, return_code = execute_shell_command(command)
        for i, filename in enumerate(file_list):
            command = f"darshan-dxt-parser {filename} > {key}_darshan_logs/{key}_darshan_dxt_log_{i}"
            stdout, stderr, return_code = execute_shell_command(command)
            command = f"cp {filename} {key}_darshan_logs"
            stdout, stderr, return_code = execute_shell_command(command)

def transfer_darshan_logs(root_path, darshan_logs_path):
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
                                        command = f"cp -rf {darshan_logs_path}/{item}_darshan_logs {item_path}"
                                        stdout, stderr, return_code = execute_shell_command(command)

if __name__ == "__main__":
    construct_shell_command(process_files(os.getcwd()))

    # root_path = "/Users/mrashid2/DirLab/perlmutter_test_data/expr_subset_8_tests/node/"
    # darshan_logs_path = "/Users/mrashid2/DirLab/perlmutter_test_data/expr_subset_8_tests/jun_5_40_executions/"
    # transfer_darshan_logs(root_path, darshan_logs_path)