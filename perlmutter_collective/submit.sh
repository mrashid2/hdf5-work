#!/bin/bash

curdir=$(pwd)

for node_type in "cpu"; do
    if [ "$node_type" = "cpu" ]; then
        core_cnt=64
    else
        core_cnt=4
    fi
    for node_cnt in 8; do
        for io_burst_size in 262144k; do
            for stripe_format in "large" "small"; do
                for buf_size in 33554432 1048576; do
                    for aggr_cnt in 16 1; do
                        for itrn_cnt in $(seq 1 1 5); do
                            tmp_dir=$curdir/node/"$node_type"_"$node_cnt"/core_$core_cnt/io_burst_$io_burst_size/stripe_$stripe_format/buf_size_$buf_size/aggr_$aggr_cnt/itrn_$itrn_cnt
                            cd $tmp_dir
                            filename=$tmp_dir/run.sh

                            echo "Submitting $filename"
                            job=`sbatch $filename`
                            echo "$job"

                            sleeptime=$[ ( $RANDOM % 10 ) ]
                            echo "Sleeping $sleeptime seconds before submitting next job."
                            sleep $sleeptime
                        done
                    done
                done
            done
        done
    done
done