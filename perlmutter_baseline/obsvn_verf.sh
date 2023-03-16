#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p premium
# #SBATCH --qos=premium
# #SBATCH -p debug
#SBATCH -A m2621
#SBATCH -t 00:00:30
#SBATCH -C cpu
#SBATCH -L SCRATCH # needs the parallel file system
#SBATCH -J obsvn_1node
#SBATCH -o o%j.obsvn_1node
#SBATCH -e o%j.obsvn_1node

VDIR=${SCRATCH}/verf_data
cat /proc/fs/lustre/osc/*/import >> $VDIR/import.txt
cat /proc/fs/lustre/osc/*/rpc_stats >> $VDIR/rpc_stats.txt
cat /proc/fs/lustre/osc/*/cur_grant_bytes >> $VDIR/cur_grant_bytes.txt
cat /sys/fs/lustre/osc/*/cur_dirty_bytes >> $VDIR/cur_dirty_bytes.txt