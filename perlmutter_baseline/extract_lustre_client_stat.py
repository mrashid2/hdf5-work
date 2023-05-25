import os
import re
import csv
import sys
import copy
import time
import subprocess

stat_name_list = [
    'observation_no',
    'avg_cur_dirty_bytes',
    'avg_cur_grant_bytes',
    'avg_waittime',
    'avg_read_rpc_bw',
    'avg_write_rpc_bw',
    'avg_read_rpc_latency',
    'avg_write_rpc_latency',
    'avg_no_of_read_rpcs',
    'avg_no_of_write_rpcs',
    'avg_size_of_read_rpcs',
    'avg_size_of_write_rpcs',
    'avg_no_of_in_flight_read_rpcs',
    'avg_no_of_in_flight_write_rpcs',
    'actual_record_duration'
]

stat_summary_list = [
    'which observation it is (start from 1)',
    'average of all IO active OSCs current amount of dirty pages (in bytes)',
    'average of all IO active OSCs current amount of grant (in bytes)',
    'average of all IO active OSCs all type of transferred RPCs average waittime (in usec)',
    'average of all read IO active OSCs transferred read RPCs during sample duration bandwidth (in MB/s)',
    'average of all write IO active OSCs transferred write RPCs during sample duration bandwidth (in MB/s)',
    'average of all read IO active OSCs transferred read RPCs during sample duration latency (in microseconds)',
    'average of all write IO active OSCs transferred write RPCs during sample duration latency (in microseconds)',
    'average no. of all read IO active OSCs transferred read RPCs during sample duration',
    'average no. of all write IO active OSCs transferred write RPCs during sample duration',
    'average size of all read IO active OSCs transferred read RPCs during sample duration (in bytes)',
    'average size of all write IO active OSCs transferred write RPCs during sample duration (in bytes)',
    'average value of all read IO active OSCs transferred in flight read RPCs during sample duration',
    'average value of all write IO active OSCs transferred in flight write RPCs during sample duration',
    'actual snapshot record duration'
]

stat_name_list_per_osc = [
    'observation_no',
    'cur_dirty_bytes',
    'cur_grant_bytes',
    'avg_waittime',
    'avg_read_rpc_bw',
    'avg_write_rpc_bw',
    'avg_read_rpc_latency',
    'avg_write_rpc_latency',
    'no_of_read_rpcs',
    'no_of_write_rpcs',
    'avg_size_of_read_rpcs',
    'avg_size_of_write_rpcs',
    'avg_no_of_in_flight_read_rpcs',
    'avg_no_of_in_flight_write_rpcs',
    'actual_record_duration'
]

stat_summary_list_per_osc = [
    'which observation it is (start from 1)',
    'current amount of dirty pages (in bytes)',
    'current amount of grant (in bytes)',
    'average of all type of transferred RPCs average waittime (in usec)',
    'average of transferred read RPCs during sample duration bandwidth (in MB/s)',
    'average of transferred write RPCs during sample duration bandwidth (in MB/s)',
    'average of transferred read RPCs during sample duration latency (in microseconds)',
    'average of transferred write RPCs during sample duration latency (in microseconds)',
    'no. of transferred read RPCs during sample duration',
    'no. of transferred write RPCs during sample duration',
    'average size of transferred read RPCs during sample duration (in bytes)',
    'average size of transferred write RPCs during sample duration (in bytes)',
    'average value of transferred in flight read RPCs during sample duration',
    'average value of transferred in flight write RPCs during sample duration',
    'actual snapshot record duration'
]

client_page_size_in_bytes = 4096
megabytes_to_bytes = 1024*1024

snap_record_duration = 1

# Tunable parameters
mppr_str = 'max_pages_per_rpc'
mdm_str = 'max_dirty_mb'
mrif_str = 'max_rpcs_in_flight'

# Future parameters to tune
mram = 'max_read_ahead_mb'
mrapfm = 'max_read_ahead_per_file_mb'
mrawm = 'max_read_ahead_whole_mb'

# Far in the future
mcm = 'max_cached_mb'

# Each path is cmmented with what parameters we will consider and
# also includes what parameters may be useful in future.

# rpc_stats, max_pages_per_rpc, osc_cached_mb, osc_stats
osc_proc_path = '/proc/fs/lustre/osc/'

# contention_seconds, cur_dirty_bytes, max_dirty_mb,
# max_rpcs_in_flight
osc_sys_fs_path = '/sys/fs/lustre/osc/'

# changing params
obsvn_cnt = -1

hostname = subprocess.run(['hostname'], stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()[0]

class OSC_Snapshot:
    def __init__(self, osc_name):
        self.osc_name = osc_name

        self.cur_read_rif = 0
        self.cur_write_rif = 0
        self.pending_read_pages = 0
        self.pending_write_pages = 0
        self.read_ppr_dist = dict()
        self.write_ppr_dist = dict()
        self.read_rif_dist = dict()
        self.write_rif_dist = dict()
        self.total_read_rpc_cnt = 0
        self.total_write_rpc_cnt = 0

        self.avg_waittime = 0
        self.read_bytes_per_rpc = 0
        self.write_bytes_per_rpc = 0
        # The latency's are also per RPC
        self.read_rpc_latency = 0
        self.write_rpc_latency = 0
        self.read_rpc_bw = 0
        self.write_rpc_bw = 0

        self.cur_dirty_bytes = 0
        self.cur_grant_bytes = 0

        # The following parameters value represent what we observed during the duration of the last sample.
        self.sample_read_rpc_latency = 0
        self.sample_write_rpc_latency = 0
        self.sample_read_rpc_bw = 0
        self.sample_write_rpc_bw = 0
        self.sample_read_rpc_cnt = 0
        self.sample_write_rpc_cnt = 0
        self.sample_read_rpc_avg_size = 0
        self.sample_write_rpc_avg_size = 0
        self.sample_read_rpc_avg_in_flight_cnt = 0
        self.sample_write_rpc_avg_in_flight_cnt = 0

    def save_cur_rif(self, cur_read_rif, cur_write_rif):
        self.cur_read_rif = cur_read_rif
        self.cur_write_rif = cur_write_rif

    def save_pending_pages(self, pending_read_pages, pending_write_pages):
        self.pending_read_pages = pending_read_pages
        self.pending_write_pages = pending_write_pages

    def save_ppr_dist(self, read_ppr_dist, write_ppr_dist):
        self.read_ppr_dist = read_ppr_dist
        self.write_ppr_dist = write_ppr_dist

    def save_rif_dist(self, read_rif_dist, write_rif_dist):
        self.read_rif_dist = read_rif_dist
        self.write_rif_dist = write_rif_dist

    def reset_sample_variables(self):
        self.sample_read_rpc_latency = 0
        self.sample_write_rpc_latency = 0
        self.sample_read_rpc_bw = 0
        self.sample_write_rpc_bw = 0
        self.sample_read_rpc_cnt = 0
        self.sample_write_rpc_cnt = 0
        self.sample_read_rpc_avg_size = 0
        self.sample_write_rpc_avg_size = 0
        self.sample_read_rpc_avg_in_flight_cnt = 0
        self.sample_write_rpc_avg_in_flight_cnt = 0

class Client_Snapshot:
    def __init__(self):
        self.osc_names = subprocess.run(['ls', '-1', osc_proc_path], stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()

        self.osc_snapshots = dict()
        for osc_name in self.osc_names:
            self.osc_snapshots[osc_name] = OSC_Snapshot(osc_name)

    def extract_dicts_from_stat_distribution(self, line_list, stat_matchmaker):
        read_dist = dict()
        write_dist = dict()

        for i in range(len(line_list)):
            try:
                line = line_list[i]
                stat_anchor = re.search('^' + stat_matchmaker + '(\s)*rpcs(\s)*% cum % \|(\s)*rpcs(\s)*% cum %$', line, re.IGNORECASE)
                if stat_anchor == None:
                    continue

                for j in range(i+1, len(line_list)):
                    try:
                        line_match = line_list[j]
                        dist_row_match = re.match('^(\d+):(\s)*(\d+)(\s)*(\d+)(\s)*(\d+)(\s)*\|(\s)*(\d+)(\s)*(\d+)(\s)*(\d+)$', line_match, re.IGNORECASE)
                        if dist_row_match == None:
                            return read_dist, write_dist

                        read_dist[int(dist_row_match.group(1))] = (int(dist_row_match.group(3)), int(dist_row_match.group(5)), int(dist_row_match.group(7)))
                        write_dist[int(dist_row_match.group(1))] = (int(dist_row_match.group(10)), int(dist_row_match.group(12)), int(dist_row_match.group(14)))

                    except AttributeError:
                        return read_dist, write_dist

            except AttributeError:
                stat_anchor = re.search('^' + stat_matchmaker + '(\s)*rpcs(\s)*% cum % \|(\s)*rpcs(\s)*% cum %$', line, re.IGNORECASE)

        return read_dist, write_dist

    def extract_single_stat_data_from_stats(self, line_list, pattern, group_no):
        for line in line_list:
            try:
                attr_match = re.search(pattern, line, re.IGNORECASE)
                value_attr = int(float(attr_match.group(group_no)))
                return value_attr

            except AttributeError:
                attr_match = re.search(pattern, line, re.IGNORECASE)

        return 0

    def extract_mult_stat_float_data_from_stats(self, line_list, pattern, group_no):
        val_list = []
        for line in line_list:
            try:
                attr_match = re.search(pattern, line, re.IGNORECASE)
                if len(group_no) > 1:
                    val_list.append(float(attr_match.group(group_no[0]) + '.' + attr_match.group(group_no[1])))
                else:
                    val_list.append(int(attr_match.group(group_no[0])))

            except AttributeError:
                attr_match = re.search(pattern, line, re.IGNORECASE)

        return val_list

    def check_avg_type(self, line_list, pattern):
        for line in line_list:
            try:
                attr_match = re.search(pattern, line, re.IGNORECASE)
                if attr_match:
                    return True

            except AttributeError:
                attr_match = re.search(pattern, line, re.IGNORECASE)

        return False

    def set_dist_rpc_count(self):
        rpc_count = 0

        for key in self.read_ppr_dist:
            rpc_count += self.read_ppr_dist[key][0]

        self.total_read_rpc_cnt = rpc_count

        rpc_count = 0

        for key in self.write_ppr_dist:
            rpc_count += self.write_ppr_dist[key][0]

        self.total_write_rpc_cnt = rpc_count

        return

    def save_osc_rpc_dist_stats_data(self, osc_name):
        # rpc_stats_lines = subprocess.run(['cat', osc_proc_path + osc_name + '/rpc_stats'], stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()
        rpc_stats_lines = []
        with open(osc_proc_path + osc_name + '/rpc_stats') as f:
            rpc_stats_lines = f.readlines()

        cur_read_rif = self.extract_single_stat_data_from_stats(rpc_stats_lines, '^read RPCs in flight:(\s)+(\d)+', 2)
        cur_write_rif = self.extract_single_stat_data_from_stats(rpc_stats_lines, '^write RPCs in flight:(\s)+(\d)+', 2)
        self.osc_snapshots[osc_name].save_cur_rif(cur_read_rif, cur_write_rif)

        pending_read_pages = self.extract_single_stat_data_from_stats(rpc_stats_lines, '^pending read pages:(\s)+(\d)+', 2)
        pending_write_pages = self.extract_single_stat_data_from_stats(rpc_stats_lines, '^pending write pages:(\s)+(\d)+', 2)
        self.osc_snapshots[osc_name].save_pending_pages(pending_read_pages, pending_write_pages)

        read_dist, write_dist = self.extract_dicts_from_stat_distribution(rpc_stats_lines, 'pages per rpc')
        self.osc_snapshots[osc_name].save_ppr_dist(read_dist, write_dist)

        read_dist, write_dist = self.extract_dicts_from_stat_distribution(rpc_stats_lines, 'rpcs in flight')
        self.osc_snapshots[osc_name].save_rif_dist(read_dist, write_dist)

        self.set_dist_rpc_count()

    def save_osc_import_data(self, osc_name):
        # import_lines = subprocess.run(['cat', osc_proc_path + osc_name + '/import'], stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()
        import_lines = []
        with open(osc_proc_path + osc_name + '/import') as f:
            import_lines = f.readlines()

        avg_waittime = self.extract_single_stat_data_from_stats(import_lines, '^(\s+)avg_waittime:(\s+)(\d+)(\s+)usec', 3)
        self.osc_snapshots[osc_name].avg_waittime = avg_waittime

        read_bytes_per_rpc = 0
        write_bytes_per_rpc = 0
        val_list = self.extract_mult_stat_float_data_from_stats(import_lines, '^(\s+)bytes_per_rpc:(\s+)(\d+)\.(\d+)', [3])

        if len(val_list) == 0:
            read_bytes_per_rpc = 0
            write_bytes_per_rpc = 0
        elif len(val_list) == 1:
            if self.check_avg_type(import_lines, '^(\s+)read_data_averages:'):
                read_bytes_per_rpc = val_list[0]
            else:
                write_bytes_per_rpc = val_list[0]
        else:
            read_bytes_per_rpc = val_list[0]
            write_bytes_per_rpc = val_list[1]

        self.osc_snapshots[osc_name].read_bytes_per_rpc = read_bytes_per_rpc
        self.osc_snapshots[osc_name].write_bytes_per_rpc = write_bytes_per_rpc

        read_rpc_latency = 0
        write_rpc_latency = 0
        val_list = self.extract_mult_stat_float_data_from_stats(import_lines, '^(\s+)usec_per_rpc:(\s+)(\d+)\.(\d+)', [3])

        if len(val_list) == 0:
            read_rpc_latency = 0
            write_rpc_latency = 0
        elif len(val_list) == 1:
            if self.check_avg_type(import_lines, '^(\s+)read_data_averages:'):
                read_rpc_latency = val_list[0]
            else:
                write_rpc_latency = val_list[0]
        else:
            read_rpc_latency = val_list[0]
            write_rpc_latency = val_list[1]

        self.osc_snapshots[osc_name].read_rpc_latency = read_rpc_latency
        self.osc_snapshots[osc_name].write_rpc_latency = write_rpc_latency

        read_rpc_bw = 0
        write_rpc_bw = 0
        val_list = self.extract_mult_stat_float_data_from_stats(import_lines, '^(\s+)MB_per_sec:(\s+)(\d+)\.(\d+)', [3, 4])

        if len(val_list) == 0:
            read_rpc_bw = 0
            write_rpc_bw = 0
        elif len(val_list) == 1:
            if self.check_avg_type(import_lines, '^(\s+)read_data_averages:'):
                read_rpc_bw = val_list[0]
            else:
                write_rpc_bw = val_list[0]
        else:
            read_rpc_bw = val_list[0]
            write_rpc_bw = val_list[1]

        self.osc_snapshots[osc_name].read_rpc_bw = read_rpc_bw
        self.osc_snapshots[osc_name].write_rpc_bw = write_rpc_bw

    def save_osc_params_data(self, osc_name):
        # cur_dirty_bytes = int(subprocess.run(['cat', osc_sys_fs_path + osc_name + '/' + 'cur_dirty_bytes'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
        cur_dirty_bytes = 0
        with open(osc_sys_fs_path + osc_name + '/' + 'cur_dirty_bytes') as f:
            cur_dirty_bytes = int(f.read())
        self.osc_snapshots[osc_name].cur_dirty_bytes = cur_dirty_bytes

        # cur_grant_bytes = int(subprocess.run(['cat', osc_proc_path + osc_name + '/' + 'cur_grant_bytes'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
        cur_grant_bytes = 0
        with open(osc_proc_path + osc_name + '/' + 'cur_grant_bytes') as f:
            cur_grant_bytes = int(f.read())
        self.osc_snapshots[osc_name].cur_grant_bytes = cur_grant_bytes

    def populate_snapshot(self):
         # Possible improvement by implementing threading here
        for osc_name in self.osc_names:
            self.save_osc_rpc_dist_stats_data(osc_name)
            self.save_osc_import_data(osc_name)
            self.save_osc_params_data(osc_name)
            self.osc_snapshots[osc_name].reset_sample_variables()

    def get_avg_cur_dirty_bytes(self, osc_names):
        if len(osc_names) == 0:
            return [0]

        total_dirty_bytes = 0

        for osc_name in osc_names:
            total_dirty_bytes += self.osc_snapshots[osc_name].cur_dirty_bytes

        return [round((total_dirty_bytes / len(osc_names)), 2)]

    def get_avg_cur_grant_bytes(self, osc_names):
        if len(osc_names) == 0:
            return [0]

        total_grant_bytes = 0

        for osc_name in osc_names:
            total_grant_bytes += self.osc_snapshots[osc_name].cur_grant_bytes

        return [round((total_grant_bytes / len(osc_names)), 2)]

    def get_avg_waittime(self, osc_names):
        if len(osc_names) == 0:
            return [0]

        total_avg_waittime = 0

        for osc_name in osc_names:
            total_avg_waittime += self.osc_snapshots[osc_name].avg_waittime

        return [round((total_avg_waittime / len(osc_names)), 2)]

    def get_avg_read_rpc_bw(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_read_rpc_bw = 0

        for osc_name in osc_names:
            diff_in_total_bytes = (self.osc_snapshots[osc_name].total_read_rpc_cnt * self.osc_snapshots[osc_name].read_bytes_per_rpc) - (prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt * prev_snap.osc_snapshots[osc_name].read_bytes_per_rpc)
            diff_in_total_usec = (self.osc_snapshots[osc_name].total_read_rpc_cnt * self.osc_snapshots[osc_name].read_rpc_latency) - (prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt * prev_snap.osc_snapshots[osc_name].read_rpc_latency)
            self.osc_snapshots[osc_name].sample_read_rpc_bw = round((diff_in_total_bytes / diff_in_total_usec), 2)
            total_read_rpc_bw += self.osc_snapshots[osc_name].sample_read_rpc_bw

        return [round((total_read_rpc_bw / len(osc_names)), 2)]

    def get_avg_write_rpc_bw(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_write_rpc_bw = 0

        for osc_name in osc_names:
            diff_in_total_bytes = (self.osc_snapshots[osc_name].total_write_rpc_cnt * self.osc_snapshots[osc_name].write_bytes_per_rpc) - (prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt * prev_snap.osc_snapshots[osc_name].write_bytes_per_rpc)
            diff_in_total_usec = (self.osc_snapshots[osc_name].total_write_rpc_cnt * self.osc_snapshots[osc_name].write_rpc_latency) - (prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt * prev_snap.osc_snapshots[osc_name].write_rpc_latency)
            self.osc_snapshots[osc_name].sample_write_rpc_bw = round((diff_in_total_bytes / diff_in_total_usec), 2)
            total_write_rpc_bw += self.osc_snapshots[osc_name].sample_write_rpc_bw

        return [round((total_write_rpc_bw / len(osc_names)), 2)]

    def get_avg_read_rpc_latency(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_read_rpc_latency = 0

        for osc_name in osc_names:
            diff_in_total_usec = (self.osc_snapshots[osc_name].total_read_rpc_cnt * self.osc_snapshots[osc_name].read_rpc_latency) - (prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt * prev_snap.osc_snapshots[osc_name].read_rpc_latency)
            diff_in_rpc_cnt = self.osc_snapshots[osc_name].total_read_rpc_cnt - prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt
            self.osc_snapshots[osc_name].sample_read_rpc_latency = round((diff_in_total_usec / diff_in_rpc_cnt), 2)
            total_read_rpc_latency += self.osc_snapshots[osc_name].sample_read_rpc_latency

        return [round((total_read_rpc_latency/ len(osc_names)), 2)]

    def get_avg_write_rpc_latency(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_write_rpc_latency= 0

        for osc_name in osc_names:
            diff_in_total_usec = (self.osc_snapshots[osc_name].total_write_rpc_cnt * self.osc_snapshots[osc_name].write_rpc_latency) - (prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt * prev_snap.osc_snapshots[osc_name].write_rpc_latency)
            diff_in_rpc_cnt = self.osc_snapshots[osc_name].total_write_rpc_cnt - prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt
            self.osc_snapshots[osc_name].sample_write_rpc_latency = round((diff_in_total_usec / diff_in_rpc_cnt), 2)
            total_write_rpc_latency += self.osc_snapshots[osc_name].sample_write_rpc_latency

        return [round((total_write_rpc_latency / len(osc_names)), 2)]

    def get_avg_no_of_read_rpcs(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_no_of_read_rpcs = 0

        for osc_name in osc_names:
            self.osc_snapshots[osc_name].sample_read_rpc_cnt = self.osc_snapshots[osc_name].total_read_rpc_cnt - prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt
            total_no_of_read_rpcs += self.osc_snapshots[osc_name].sample_read_rpc_cnt

        return [round((total_no_of_read_rpcs/ len(osc_names)), 2)]

    def get_avg_no_of_write_rpcs(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_no_of_write_rpcs = 0

        for osc_name in osc_names:
            self.osc_snapshots[osc_name].sample_write_rpc_cnt = self.osc_snapshots[osc_name].total_write_rpc_cnt - prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt
            total_no_of_write_rpcs += self.osc_snapshots[osc_name].sample_write_rpc_cnt

        return [round((total_no_of_write_rpcs/ len(osc_names)), 2)]

    def get_avg_size_of_read_rpcs(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_size_of_read_rpcs = 0

        for osc_name in osc_names:
            diff_in_total_bytes = (self.osc_snapshots[osc_name].total_read_rpc_cnt * self.osc_snapshots[osc_name].read_bytes_per_rpc) - (prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt * prev_snap.osc_snapshots[osc_name].read_bytes_per_rpc)
            diff_in_rpc_cnt = self.osc_snapshots[osc_name].total_read_rpc_cnt - prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt
            self.osc_snapshots[osc_name].sample_read_rpc_avg_size = round((diff_in_total_bytes / diff_in_rpc_cnt), 2)
            total_size_of_read_rpcs += self.osc_snapshots[osc_name].sample_read_rpc_avg_size

        return [round((total_size_of_read_rpcs/ len(osc_names)), 2)]

    def get_avg_size_of_write_rpcs(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_size_of_write_rpcs = 0

        for osc_name in osc_names:
            diff_in_total_bytes = (self.osc_snapshots[osc_name].total_write_rpc_cnt * self.osc_snapshots[osc_name].write_bytes_per_rpc) - (prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt * prev_snap.osc_snapshots[osc_name].write_bytes_per_rpc)
            diff_in_rpc_cnt = self.osc_snapshots[osc_name].total_write_rpc_cnt - prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt
            self.osc_snapshots[osc_name].sample_write_rpc_avg_size = round((diff_in_total_bytes / diff_in_rpc_cnt), 2)
            total_size_of_write_rpcs += self.osc_snapshots[osc_name].sample_write_rpc_avg_size

        return [round((total_size_of_write_rpcs/ len(osc_names)), 2)]

    def get_avg_of_diff_in_rpcs_in_flight_per_osc(self, cur_dist, prev_dist):
        rpc_in_flight_cnt = 0
        diff_in_rpcs_in_flight = 0

        for key in self.cur_dist:
            try:
                rpc_in_flight_cnt += cur_dist[key][0] - prev_dist[key][0]
                diff_in_rpcs_in_flight += (cur_dist[key][0] - prev_dist[key][0]) * key
            except KeyError:
                rpc_in_flight_cnt += cur_dist[key][0]
                diff_in_rpcs_in_flight += cur_dist[key][0] * key

        return round((diff_in_rpcs_in_flight / rpc_in_flight_cnt), 2)

    def get_avg_no_of_in_flight_read_rpcs(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_no_of_in_flight_read_rpcs = 0

        for osc_name in osc_names:
            self.osc_snapshots[osc_name].sample_read_rpc_avg_in_flight_cnt = self.get_avg_of_diff_in_rpcs_in_flight_per_osc(self.osc_snapshots[osc_name].read_rif_dist, prev_snap.osc_snapshots[osc_name].read_rif_dist)
            total_no_of_in_flight_read_rpcs += self.osc_snapshots[osc_name].sample_read_rpc_avg_in_flight_cnt

        return [round((total_no_of_in_flight_read_rpcs/ len(osc_names)), 2)]

    def get_avg_no_of_in_flight_write_rpcs(self, osc_names, prev_snap):
        if len(osc_names) == 0:
            return [0]

        total_no_of_in_flight_write_rpcs = 0

        for osc_name in osc_names:
            self.osc_snapshots[osc_name].sample_write_rpc_avg_in_flight_cnt = self.get_avg_of_diff_in_rpcs_in_flight_per_osc(self.osc_snapshots[osc_name].write_rif_dist, prev_snap.osc_snapshots[osc_name].write_rif_dist)
            total_no_of_in_flight_write_rpcs += self.osc_snapshots[osc_name].sample_write_rpc_avg_in_flight_cnt

        return [round((total_no_of_in_flight_write_rpcs/ len(osc_names)), 2)]

    def get_read_active_osc_names(self, prev_snap):
        read_active_osc_names = []

        for osc_name in self.osc_names:
            if self.osc_snapshots[osc_name].total_read_rpc_cnt > prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt:
                read_active_osc_names.append(osc_name)

        return read_active_osc_names

    def get_write_active_osc_names(self, prev_snap):
        write_active_osc_names = []

        for osc_name in self.osc_names:
            if self.osc_snapshots[osc_name].total_write_rpc_cnt > prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt:
                write_active_osc_names.append(osc_name)

        return write_active_osc_names

    def get_io_active_osc_names(self, prev_snap):
        io_active_osc_names = []

        for osc_name in self.osc_names:
            if self.osc_snapshots[osc_name].total_read_rpc_cnt > prev_snap.osc_snapshots[osc_name].total_read_rpc_cnt or self.osc_snapshots[osc_name].total_write_rpc_cnt > prev_snap.osc_snapshots[osc_name].total_write_rpc_cnt:
                io_active_osc_names.append(osc_name)

        return io_active_osc_names

    def construct_params_list(self, record_dur, prev_snap):
        read_active_osc_names = self.get_read_active_osc_names(prev_snap)
        write_active_osc_names = self.get_write_active_osc_names(prev_snap)
        io_active_osc_names = self.get_io_active_osc_names(prev_snap)

        global obsvn_cnt

        params_list = []
        params_list = params_list + [obsvn_cnt]
        params_list = params_list + self.get_avg_cur_dirty_bytes(io_active_osc_names)
        params_list = params_list + self.get_avg_cur_grant_bytes(io_active_osc_names)
        params_list = params_list + self.get_avg_waittime(io_active_osc_names)
        params_list = params_list + self.get_avg_read_rpc_bw(read_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_write_rpc_bw(write_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_read_rpc_latency(read_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_write_rpc_latency(write_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_no_of_read_rpcs(read_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_no_of_write_rpcs(write_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_size_of_read_rpcs(read_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_size_of_write_rpcs(write_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_no_of_in_flight_read_rpcs(read_active_osc_names, prev_snap)
        params_list = params_list + self.get_avg_no_of_in_flight_write_rpcs(write_active_osc_names, prev_snap)
        params_list = params_list + [record_dur]

        return params_list

    def construct_params_list_per_osc(self, record_dur, osc_name):
        global obsvn_cnt

        params_list = []
        params_list = params_list + [obsvn_cnt]
        params_list = params_list + self.osc_snapshots[osc_name].cur_dirty_bytes
        params_list = params_list + self.osc_snapshots[osc_name].cur_grant_bytes
        params_list = params_list + self.osc_snapshots[osc_name].avg_waittime
        params_list = params_list + self.osc_snapshots[osc_name].sample_read_rpc_bw
        params_list = params_list + self.osc_snapshots[osc_name].sample_write_rpc_bw
        params_list = params_list + self.osc_snapshots[osc_name].sample_read_rpc_latency
        params_list = params_list + self.osc_snapshots[osc_name].sample_write_rpc_latency
        params_list = params_list + self.osc_snapshots[osc_name].sample_read_rpc_cnt
        params_list = params_list + self.osc_snapshots[osc_name].sample_write_rpc_cnt
        params_list = params_list + self.osc_snapshots[osc_name].sample_read_rpc_avg_size
        params_list = params_list + self.osc_snapshots[osc_name].sample_write_rpc_avg_size
        params_list = params_list + self.osc_snapshots[osc_name].sample_read_rpc_avg_in_flight_cnt
        params_list = params_list + self.osc_snapshots[osc_name].sample_write_rpc_avg_in_flight_cnt
        params_list = params_list + [record_dur]

        return params_list

    def write_params_list_to_csv(self, result_folder_path, wld_name, params_list):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        osc_csv_filename = os.path.join(script_dir, result_folder_path, wld_name + "_" + hostname + "_client_stats_sheet.csv")

        with open(osc_csv_filename, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([param for param in params_list])

    def write_params_list_to_csv_per_osc(self, result_folder_path, wld_name, params_list, osc_name):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        osc_csv_filename = os.path.join(script_dir, result_folder_path, wld_name + "_" + hostname + "_" + osc_name + "_osc_stats_sheet.csv")

        with open(osc_csv_filename, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([param for param in params_list])

    def include_params_header_and_summary(self, result_folder_path, wld_name):
        self.write_params_list_to_csv(result_folder_path, wld_name, stat_name_list)
        self.write_params_list_to_csv(result_folder_path, wld_name, stat_summary_list)

    def include_params_header_and_summary_per_osc(self, result_folder_path, wld_name, osc_name):
        self.write_params_list_to_csv_per_osc(result_folder_path, wld_name, stat_name_list_per_osc, osc_name)
        self.write_params_list_to_csv_per_osc(result_folder_path, wld_name, stat_summary_list_per_osc, osc_name)

    def store_params_to_csv(self, result_folder_path, wld_name, record_dur, prev_snap):
        global obsvn_cnt
        obsvn_cnt = obsvn_cnt + 1
        if obsvn_cnt == 0:
            self.include_params_header_and_summary(result_folder_path, wld_name)

        self.write_params_list_to_csv(result_folder_path, wld_name, self.construct_params_list(record_dur, prev_snap))

    def store_params_to_csv_per_osc(self, result_folder_path, wld_name, record_dur, prev_snap):
        global obsvn_cnt
        io_active_osc_names = self.get_io_active_osc_names(prev_snap)

        for osc_name in io_active_osc_names:
            if obsvn_cnt == 0:
                self.include_params_header_and_summary_per_osc(result_folder_path, wld_name, osc_name)

            self.write_params_list_to_csv_per_osc(result_folder_path, wld_name, self.construct_params_list_per_osc(record_dur, osc_name), osc_name)

    def store_params(self, result_folder_path, wld_name, record_dur, prev_snap):
        self.store_params_to_csv(result_folder_path, wld_name, record_dur, prev_snap)

        # 'store_params_to_csv_per_osc' method MUST be called after 'store_params_to_csv' method
        self.store_params_to_csv_per_osc(result_folder_path, wld_name, record_dur, prev_snap)

if __name__ == "__main__":
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # relative path from the script
    result_folder_path = sys.argv[1]
    wld_name = sys.argv[2]

    cur_snap = Client_Snapshot()
    # print(cur_snap.osc_names)
    prev_snap = Client_Snapshot()
    prev_snap.populate_snapshot()

    begin_time = time.time()
    time.sleep(snap_record_duration)

    while True:
        end_time = time.time()
        record_dur = round((end_time - begin_time), 4)

        cur_snap.populate_snapshot()
        begin_time = time.time()
        print('Snapshot Generation Time: ', int((time.time() - end_time) * 1000), ' miliseconds')

        cur_snap.store_params(result_folder_path, wld_name, record_dur, prev_snap)
        prev_snap = copy.deepcopy(cur_snap)
        print('Processing Time (Before Sleeping): ', int((time.time() - end_time) * 1000), ' miliseconds')

        time.sleep(snap_record_duration)
