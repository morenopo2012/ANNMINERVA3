#!/bin/bash

clear

SING_DIR="/lfstev/e-938/jbonilla/sing_imgs" # Directory with singularity
SINGLRTY="${SING_DIR}/joaocaldeira-singularity_imgs-master-py3_tf114.simg" # Singularity image with python3

EXE=./matplot.py

PREDICTIONS=$1
HDF5=/lfstev/e-938/jbonilla/hdf5/me1Fmc/* #hadmultkineimgs_127x94_me1Fmc-part1.hdf5
TARGET=hadro_data/n_hadmultmeas_100mev
DIS=$2

echo $HDF5

ARGS="--predictions $PREDICTIONS"
ARGS+=" --hdf5 ${HDF5}"
if [ $DIS == "DIS" ] ; then
    ARGS+=" --DIS"
fi

# Show the command to be executed
cat << EOF
$EXE $ARGS
EOF

# Execute the command
singularity exec --nv $SINGLRTY python3 $EXE $ARGS
