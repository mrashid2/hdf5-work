#!/bin/bash -l
#SBATCH -N NNODE
#SBATCH -n EXCLSVPROC
#SBATCH -q regular
#SBATCH -A m2621
#SBATCH -t 00:05:00
#SBATCH -C NODETYPE
#SBATCH -L SCRATCH # needs the parallel file system
#SBATCH -J IOR_NNODEnode
#SBATCH -o o%j.ior_NNODEnode
#SBATCH -e o%j.ior_NNODEnode

SDIR=${SCRATCH}
DDIR=$SDIR/ior_data/$SLURM_JOBID
EXEC=${HOME}/benchmarks/ior/src/ior

# trap the signal to the main BATCH script here.
sig_handler()
{
    echo "BATCH interrupted"
    rm -rf $DDIR
    wait # wait for all children, this is important!
}

trap 'sig_handler' SIGINT SIGTERM SIGCONT

module unload cray-hdf5/1.12.2.3
module unload cpe-cuda/23.03

module load cpe-cuda/23.03
module load cray-hdf5-parallel/1.12.2.3
module load darshan/3.4.0-hdf5
module load python

#export LD_LIBRARY_PATH=${HOME}/perlmutter/hdf5-1.10.6/build/hdf5/lib:$LD_LIBRARY_PATH
export MPICH_MPIIO_STATS=1
export MPICH_MPIIO_HINTS_DISPLAY=1
export MPICH_MPIIO_TIMERS=1
export MPICH_MPIIO_AGGREGATOR_PLACEMENT_DISPLAY=1
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DXT_ENABLE_IO_TRACE=4

#print romio hints
export ROMIO_PRINT_HINTS=1

export LLOGDIR="$SDIR/node/NODETYPE_NNODE/core_CORECNT/io_burst_IOBURST/stripe_STRIPETYPE/buf_size_BUFSIZE/aggr_AGGRCNT/itrn_ITRNCNT/$SLURM_JOBID"
export SLOGDIR="$SDIR/node/NODETYPE_NNODE/core_CORECNT/io_burst_IOBURST/stripe_STRIPETYPE/buf_size_BUFSIZE/aggr_AGGRCNT/itrn_ITRNCNT/$SLURM_JOBID/lustre_stats"
export LOCALDIR="/dev/shm/$SLURM_JOBID"
mkdir -p "$LLOGDIR"
mkdir -p "$SLOGDIR"

run_cmd="srun -N NNODE -n NNODE --ntasks-per-node 1 --exclusive"
excn_marker="NODETYPE_node_NNODE_ncore_CORECNT_ioburst_IOBURST_stripe_STRIPETYPE_bufsize_BUFSIZE_aggr_AGGRCNT_itrn_ITRNCNT"

# Define a timestamp function
timestamp() {
    date +"%Y%m%d_%H%M%S" # current time
}

ior(){
    #check file size to determine alignment setting
    size=$(echo "IOBURST" | sed 's/k//g')
    fileSize=$(($size*CORECNT*NNODE/1024))
    echo "size in KB: $size; fileSize in MB:$fileSize"
    if [[ $fileSize -ge 16 ]]; then
        align=16m
    else
        align=1m
    fi

    #check file size to determine default lustre striping, by following nersc recommendation
    stripe_count=1
    if [[ ! -d $DDIR ]]; then
        mkdir -p $DDIR
        default="-c 1 -S 1m $DDIR"
        small="-c 8 -S 1m $DDIR"
        medium="-c 24 -S 1m $DDIR"
        large="-c 64 -S 1m $DDIR"
        Lustre_Default=$default

        if [ STRIPETYPE = "small" ]; then
            Lustre_Default=$small
            stripe_count=8
        elif [ STRIPETYPE = "medium" ]; then
            Lustre_Default=$medium
            stripe_count=24
        elif [ STRIPETYPE = "large" ]; then
            Lustre_Default=$large
            stripe_count=64
        fi

        lfs setstripe $Lustre_Default
        echo "lustre default, "$excn_marker", $Lustre_Default"
    fi

    #track results
    rdir="$LLOGDIR"/ior_excn_outputs
    mkdir -p $rdir

    ior_write(){
        col_write(){
            naggr=$((NNODE*AGGRCNT))
            multiplier_cnt=1
            if [[ $naggr -gt $stripe_count ]]; then
                multiplier_cnt=$(( $naggr/$stripe_count ))
            fi

            export MPICH_MPIIO_HINTS="*:cb_nodes=$naggr:cb_buffer_size=BUFSIZE:cray_cb_nodes_multiplier=$multiplier_cnt:cray_cb_write_lock_mode=2:romio_cb_write=enable:romio_cb_read=enable:cb_config_list=#*:AGGRCNT"
            echo $MPICH_MPIIO_HINTS

            #flush data in data transfer, before file close
            NPROC=$((NNODE*CORECNT))
            export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
            srun -N NNODE -n $NPROC --exclusive $EXEC -b IOBURST -t IOBURST -i 1 -v -v -v -k -a HDF5 --hdf5.setAlignment=$align -c -e -w -o $DDIR/col_"$excn_marker"_f&>>$rdir/col_"$excn_marker"_f
            export LD_PRELOAD=""
            lfs getstripe $DDIR/col_"$excn_marker"_f
        }

        col_write
    }

    ior_read(){
        col_read(){
	        naggr=$((NNODE*AGGRCNT))
            multiplier_cnt=1
            if [[ $naggr -gt $stripe_count ]]; then
                multiplier_cnt=$(( $naggr/$stripe_count ))
            fi

            export MPICH_MPIIO_HINTS="*:cb_nodes=$naggr:cb_buffer_size=BUFSIZE:cray_cb_nodes_multiplier=$multiplier_cnt:cray_cb_write_lock_mode=2:romio_cb_write=enable:romio_cb_read=enable:cb_config_list=#*:AGGRCNT"
            echo $MPICH_MPIIO_HINTS

            NPROC=$((NNODE*CORECNT))
            export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
            srun -N NNODE -n $NPROC --exclusive $EXEC -b IOBURST -t IOBURST -i 1 -v -v -v -k -a HDF5 --hdf5.setAlignment=$align -c -r -o $DDIR/col_"$excn_marker"_f&>>$rdir/col_"$excn_marker"_r
            export LD_PRELOAD=""
        }

        col_read
    }

    aggr_data_thrshld_bytes=$(( 32*1024*1024*1024 ))
    aggr_data_size_bytes=$(( $fileSize*1024*1024 ))
    echo "aggr_data_thrshld_bytes: $aggr_data_thrshld_bytes; aggr_data_size_bytes: $aggr_data_size_bytes"

    loop_needed=1
    if [[ $aggr_data_thrshld_bytes -gt $aggr_data_size_bytes ]]; then
        loop_needed=$(( $aggr_data_thrshld_bytes/$aggr_data_size_bytes ))
    fi
    echo "loop_needed: $loop_needed"

    for loop_cnt in $(seq 1 1 $loop_needed); do
        ior_write
        ior_read
        echo "loop $loop_cnt ior_write+ior_read done!"
    done

    rm -rf $DDIR
}

# Create the local directory in /dev/shm, using one process per node
$run_cmd mkdir -p "$LOCALDIR"


echo "Starting python background process"
timestamp

pdir="$LLOGDIR"/pyscript_logs
mkdir -p $pdir
$run_cmd mkdir -p "$LOCALDIR"/"$excn_marker"
$run_cmd python -u $SDIR/extract_lustre_client_stat.py "$LOCALDIR"/"$excn_marker" "$excn_marker" $SLOGDIR &>> $pdir/"$excn_marker"_pylog &
pid=$!
echo "Started python background process"
timestamp

ior
echo "Finished IOR execution"
timestamp

# Cancel the srun job
echo "Cancelling srun job with PID $pid"
scancel $pid
timestamp

date
echo "====Done===="