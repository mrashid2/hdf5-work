#!/bin/bash -l
#SBATCH -N NNODE
#SBATCH -n 264
#SBATCH -q debug
#SBATCH -A m2621
#SBATCH -t 00:05:00
#SBATCH -C cpu
#SBATCH -L SCRATCH # needs the parallel file system
#SBATCH -J IOR_NNODEnode
#SBATCH -o o%j.ior_NNODEnode
#SBATCH -e o%j.ior_NNODEnode

SDIR=${SCRATCH}
EXEC=${HOME}/benchmarks/ior/src/ior

module unload craype-accel-nvidia80
module unload cpe-cuda/23.02
module unload cray-hdf5-parallel/1.12.2.3
module unload cray-hdf5/1.12.2.3
module unload darshan/3.4.0-hdf5
module unload python

module load cpe-cuda/23.02
module load cray-hdf5-parallel/1.12.2.3
module load darshan/3.4.0-hdf5
module load python

#export LD_LIBRARY_PATH=${HOME}/perlmutter/hdf5-1.10.6/build/hdf5/lib:$LD_LIBRARY_PATH
export MPICH_MPIIO_STATS=1
export MPICH_MPIIO_HINTS_DISPLAY=1
export MPICH_MPIIO_TIMERS=1
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DXT_ENABLE_IO_TRACE=4

#print romio hints
export ROMIO_PRINT_HINTS=1

export OUTDIR="$SCRATCH/outputs/$SLURM_JOBID"
export LOCALDIR="/dev/shm/$SLURM_JOBID"
mkdir -p "$OUTDIR"

#two varying parameters: 1. number of aggregators, 2. stripe size 
half_aggr=$((NNODE/2))
quat_aggr=$((NNODE/4))
eqal_aggr=NNODE
doul_aggr=$((NNODE*2))
qudr_aggr=$((NNODE*4))

naggrs="$doul_aggr $qudr_aggr $eqal_aggr $half_aggr $quat_aggr"
stripe_sizes="1m 2m 4m 8m 16m 32m 64m 128m"

run_cmd="srun -N NNODE -n NNODE  --ntasks-per-node 1 --exclusive"

# Define a timestamp function
timestamp() {
  date +"%Y%m%d_%H%M%S" # current time
}

ior(){
    local i=$1
    local ncore=$2
    local burst=$3
 
    #check file size to determine alignment setting
    size="${burst//k}"
    fileSize=$(($size*$ncore*NNODE/1024))
    if [[ $fileSize -ge 16 ]]; then
        align=16m
    else
        align=1m
    fi

    #check file size to determine default lustre striping, by following nersc recommendation 
    DDIR=$SCRATCH/ior_data/ior_${ncore}_${burst}_default
    if [[ ! -d $DDIR ]]; then
        mkdir -p $DDIR
        fGB=$(($fileSize/1024))
        small="-c 8 -S 1m $DDIR"
        medium="-c 24 -S 1m $DDIR"
        large="-c 72 -S 1m $DDIR"
        if [[ $fGB -le 10 ]]; then
            Lustre_Default=$small 
        elif [[ $fGB -gt 10 && $fGB -le 100 ]]; then
            Lustre_Default=$medium
        elif [[ $fGB -gt 10 && $fGB -le 100 ]]; then
            Lustre_Deafult=$large 
        fi 

        lfs setstripe $Lustre_Default
    	echo "lustre default, ${ncore}_${burst}, $fGB, $Lustre_Default"
    fi

    #track results
    rdir=result_${ncore}_${burst}
    mkdir -p $rdir

    ior_write(){
        col_write(){
            local naggr=$1
            local stripe_size=$2
	        CDIR=$SCRATCH/ior_data/ior_${ncore}_${burst}_${stripe_size}
            mkdir -p $CDIR
	
	        #load romio hints
            # export ROMIO_HINTS=$hfile
            hfile=$rdir/aggr_${naggr}
            cp hints/aggr_${naggr} $hfile  
            hvalue=`cat $hfile`
            echo "$hvalue"
            export MPICH_MPIIO_HINTS="*:$hvalue"
            echo $MPICH_MPIIO_HINTS

            #flush data in data transfer, before file close 
            let NPROC=NNODE*$ncore
            export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
            srun -N NNODE -n $NPROC --exclusive $EXEC -b $burst -t $burst -i 1 -v -v -v -k -a HDF5 --hdf5.setAlignment=$align -c -e -w -o $CDIR/col_${i}_${ncore}_${burst}_${naggr}_${stripe_size}_f&>>$rdir/col_${ncore}_${burst}_${naggr}_${stripe_size}_f
            export LD_PRELOAD=""
        }

        for naggr in $quat_aggr; do
            for stripe_size in 16m; do 
                    col_write $naggr $stripe_size
            done
        done
    }
    
    ior_read(){
        col_read(){ 
            local naggr=$1
            local stripe_size=$2
	        CDIR=$SCRATCH/ior_data/ior_${ncore}_${burst}_${stripe_size}
            mkdir -p $CDIR
	        
            #load romio hints
            # export ROMIO_HINTS=$rdir/aggr_${naggr}_${buffer}
            hfile=$rdir/aggr_${naggr}
            cp hints/aggr_${naggr} $hfile  
            hvalue=`cat $rdir/aggr_${naggr}`
            echo "$hvalue"
            export MPICH_MPIIO_HINTS="*:$hvalue"
            echo $MPICH_MPIIO_HINTS

            let NPROC=NNODE*$ncore
            export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
            srun -N NNODE -n $NPROC --exclusive $EXEC -b $burst -t $burst -i 1 -v -v -v -k -a HDF5 --hdf5.setAlignment=$align -c -r -o $CDIR/col_${i}_${ncore}_${burst}_${naggr}_${stripe_size}_f&>>$rdir/col_${ncore}_${burst}_${naggr}_${stripe_size}_r
            export LD_PRELOAD=""
        }

        for naggr in $quat_aggr; do
            for stripe_size in 16m; do 
                    col_read $naggr $stripe_size
            done
        done
    }
    
    ior_write
    echo "ior_write done!" 
    ior_read
    echo "ior_read done!"

    # CDIR=${SCRATCH}/ior_data
    # rm -rf $CDIR
}

# Create the local directory in /dev/shm, using one process per node
$run_cmd mkdir -p "$LOCALDIR"

echo "Before Loop"
for i in 1; do
    echo "i: $i"
    for ncore in 16; do
        echo "ncore: $ncore"
        for burst in 2721k; do
            echo "burst: $burst"
            echo "Starting python background process"
            timestamp

            $run_cmd mkdir -p "$LOCALDIR"/${i}_ncore_${ncore}_burst_${burst}_aggr_$quat_aggr
            $run_cmd python -u $SDIR/extract_lustre_client_stat.py "$LOCALDIR"/${i}_ncore_${ncore}_burst_${burst}_aggr_$quat_aggr ${i}_ncore_${ncore}_burst_${burst}_aggr_$quat_aggr &>> ${i}_${ncore}_${burst}${unit}_pylog &
            pid=$!
            echo "Started python background process"
            timestamp

            ior $i $ncore $burst
            echo "Finished IOR execution"
            timestamp

            # Cancel the srun job
            echo "Cancelling srun job with PID $pid"
            scancel $pid
        done
    done
echo "Iter $i Done"
timestamp
done

# Send one "collecting" process to archive all local directories into separate archives
# We have to use 'bash -c' because 'hostname' needs to be interpreted on each node separately
$run_cmd bash -c 'tar -cf "$OUTDIR/output_$(hostname).tar" -C "$LOCALDIR" .'

date
echo "====Done===="