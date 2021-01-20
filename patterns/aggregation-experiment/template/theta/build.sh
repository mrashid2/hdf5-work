#!/bin/bash

nodes="4"
user=$USER

repetitions=9
per_write=3

cdir=~/hdf5-work/patterns/aggregation-experiment/patterns/benchmark_pattern/
machine=theta
idir=$cdir/$machine
#determine how many jobs in the queue
count=`qstat|grep $USER|grep IOR|wc -l`
curr_dir=`pwd`
num_running_jobs=1

node(){
    local node=$1
    ndir=node${node}
    pdir=$idir/$ndir
    
    mkdir -p $ndir
    cp -rf $pdir/hints $ndir/.
    cp template.sh $ndir/.
    touch $ndir/complete

    patterns=`ls $pdir|sed "s/hints//"`   
    cd $ndir

    #check completed pattern
    pattern(){
        local per=$1
        core=`echo $per | cut -d'_' -f 2`
        size=`echo $per | cut -d'_' -f 3`
 
        if [[ $count -lt $num_running_jobs ]]; then
            read_line="${core}_${size} r"
            write_line="${core}_${size} w"
	    read_count=`cat complete|grep "$read_line"|wc -l`
	    write_count=`cat complete|grep "$write_line"|wc -l`

            echo "write count $write_count; read count $read_count"

            if [[ $write_count -lt $repetitions || $read_count -lt $repetitions ]]; then
                local timestamp=`date +%s`
                local name=${per}_${timestamp}.sh

                cp template.sh $name
                cat $pdir/$per>>$name
                # echo "rm -rf /tmp/jsm.login1.4069">>${name}

                if [ "$node" -gt "16" ]
                then 
                    sed -i "s/ALLOCNODE/128/g" $name 
                    sed -i "s/NNODE/$node/g" $name 
                    sed -i "s/MYQUEUE/default/g" $name 
                else
                    sed -i "s/ALLOCNODE/$node/g" $name 
                    sed -i "s/NNODE/$node/g" $name 
                    sed -i "s/MYQUEUE/debug-cache-quad/g" $name 
                fi



                last=`sqs|grep $user|grep "IOR"|tail -1|awk '{print $1}'`
                if [[ -z $last ]]; then
                    echo "Submitting $per in node${node}"
                    # bsub ${name}
                    job=`qsub $name`
                    count=$(($count+1))
                else
                    echo "Submitting $per in node${node} after $job"
                    # bsub ${name}
                    job=`qsub --dependencies ${job: -6} $name`
                    count=$(($count+1))
                fi
            fi

        fi
    }   
    for per in $patterns; do
        pattern $per
    done
 
}
for node in $nodes; do
    node $node
done

cd $curr_dir
