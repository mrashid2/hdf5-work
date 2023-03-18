#!/bin/bash -l
#SBATCH -N NNODE
#SBATCH -p premium
# #SBATCH --qos=premium
# #SBATCH -p debug
#SBATCH -A m2621
#SBATCH -t 00:01:00
#SBATCH -C cpu
#SBATCH -L SCRATCH # needs the parallel file system
#SBATCH -J IOR_NNODEnode
#SBATCH -o o%j.ior_NNODEnode
#SBATCH -e o%j.ior_NNODEnode

let NPROC=NNODE
SDIR=${SCRATCH}
CDIR=${SCRATCH}/ior_data
EXEC=${HOME}/benchmarks/ior/src/ior

module unload cpe-cuda/23.02
module unload cray-hdf5-parallel/1.12.2.3
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


#for the hdf5 setting with specific alignment value, the api is HDF5+alignment_setting_value, for the runs with hdf5 setting we perform the ior with no collective i/o for hdf5 metadata, and with collective i/o for hdf5 metadata.
apis="POSIX MPIIO HDF5 HDF51m HDF54m HDF516m HDF564m HDF5256m"

sizes="1 16 256"
units="k m"

sizeg="1"
unitg="g"

run_cmd="srun -N NNODE -n $NPROC"

timestamp=$(date +%Y-%m-%d_%H-%M-%S)

ior(){
    local i=$1
    local api=$2
    local aggr=$3
    local unit=$4


    mkdir -p $CDIR
    lfs setstripe -c 248 -S 16m $CDIR

    if [[ $api == "POSIX" || $api == "MPIIO" || $api == "HDF5" ]]; then
        #flush data in data transfer
        export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
        $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $api -e -w -o $CDIR/${NPROC}p_${i}_${api}_f&>>${api}_${aggr}${unit}_f
        #read
        $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $api  -r -C -o $CDIR/${NPROC}p_${i}_${api}_f &>>${api}_${aggr}${unit}_r
        export LD_PRELOAD=""
    elif [[ $api == "HDF5C" ]]; then
        local input="HDF5"
        export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
        #flush data in data transfer
        $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $input -J $alignment -e -w -o $CDIR/${NPROC}p_${i}_${api}_f&>>${api}_${aggr}${unit}_f --hdf5.collectiveMetadata
        #read
        $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $input -J $alignment  -r -C -o $CDIR/${NPROC}p_${i}_${api}_f &>>${api}_${aggr}${unit}_r --hdf5.collectiveMetadata
        export LD_PRELOAD=""
    else
        local input=`echo "$api"|cut -c1-4`
        local residual=`echo "$api"|cut -c5-9`
        if [[ $residual != *"C"* ]]; then
            local alignment=$residual
            export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
            #flush data in data transfer
            $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $input -J $alignment -e -w -o $CDIR/${NPROC}p_${i}_${api}_f&>>${api}_${aggr}${unit}_f
            #read
            $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $input -J $alignment  -r -C -o $CDIR/${NPROC}p_${i}_${api}_f &>>${api}_${aggr}${unit}_r
            export LD_PRELOAD=""
        else
            local alignment=$(echo $residual|sed "s/C//")
            export LD_PRELOAD=/global/common/software/nersc/pm-2022q3/sw/darshan/3.4.0-hdf5/lib/libdarshan.so
            #flush data in data transfer
            $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $input -J $alignment -e -w -o $CDIR/${NPROC}p_${i}_${api}_f&>>${api}_${aggr}${unit}_f --hdf5.collectiveMetadata
            #read
            $run_cmd $EXEC -b ${aggr}${unit} -t ${aggr}${unit} -i 1 -v -v -v -k -a $input -J $alignment  -r -C -o $CDIR/${NPROC}p_${i}_${api}_f &>>${api}_${aggr}${unit}_r --hdf5.collectiveMetadata
            export LD_PRELOAD=""
        fi
    fi

    rm -rf $CDIR

}

echo "Before Loop"
for i in 1; do
    echo "i: $i"
    for api in HDF5; do
        echo "api: $api"
        for aggr in 24; do
            echo "aggr: $aggr"
            for unit in g; do
                echo "Starting python background process"
                $timestamp
                $run_cmd python -u $SDIR/extract_lustre_client_stat.py $SDIR ${i}_${api}_${aggr}${unit} &>> ${i}_${api}_${aggr}${unit}_pylog &
                pid=$!
                echo "Started python background process"
                $timestamp
                ior $i $api $aggr $unit &
                echo "Finished IOR execution"
                $timestamp
                # Cancel the srun job
                echo "Cancelling srun job with PID $pid"
                scancel $pid
            done
        done
    done
echo "Iter $i Done"
$timestamp
done

date
echo "====Done===="
