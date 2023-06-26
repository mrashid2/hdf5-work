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
            for stripe_cnt in 128; do
                for stripe_size in 268435456; do
                    for mult_cnt in 8; do
                        for aggr_cnt in 128; do
                            for itrn_cnt in 1; do
                                tmp_dir=$curdir/node/"$node_type"_"$node_cnt"/core_$core_cnt/io_burst_$io_burst_size/stripe_cnt_$stripe_cnt/stripe_size_$stripe_size/mult_cnt_$mult_cnt/aggr_$aggr_cnt/itrn_$itrn_cnt
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
done