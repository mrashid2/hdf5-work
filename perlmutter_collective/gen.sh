#!/bin/bash

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
                                tmp_dir=./node/"$node_type"_"$node_cnt"/core_$core_cnt/io_burst_$io_burst_size/stripe_cnt_$stripe_cnt/stripe_size_$stripe_size/mult_cnt_$mult_cnt/aggr_$aggr_cnt/itrn_$itrn_cnt
                                mkdir -p $tmp_dir
                                myfile=./$tmp_dir/run.sh
                                cp -f ./template.sh $myfile

                                sed -i "s/NODETYPE/$node_type/g" $myfile
                                sed -i "s/CORECNT/$core_cnt/g" $myfile
                                sed -i "s/NNODE/$node_cnt/g" $myfile

                                excl_proc_cnt=$(( ($node_cnt*$core_cnt)+$node_cnt ))
                                sed -i "s/EXCLSVPROC/$excl_proc_cnt/g" $myfile

                                sed -i "s/IOBURST/$io_burst_size/g" $myfile
                                sed -i "s/STRIPECNT/$stripe_cnt/g" $myfile
                                sed -i "s/STRIPESIZE/$stripe_size/g" $myfile
                                sed -i "s/MULTCNT/$mult_cnt/g" $myfile
                                sed -i "s/AGGRCNT/$aggr_cnt/g" $myfile
                                sed -i "s/ITRNCNT/$itrn_cnt/g" $myfile
                                chmod a+rwx $myfile
                            done
                        done
                    done
                done
            done
        done
    done
done
