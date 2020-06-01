#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=VF

echo "started "`date`" "`date +%s`""

nvidia-smi -L

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GENERAL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Directory with singularity image
SING_DIR="/lfstev/e-938/jbonilla/sing_imgs"
# Singularity image
SINGLRTY="${SING_DIR}/joaocaldeira-singularity_imgs-master-py3_tf114.simg"
#SINGLRTY="${SING_DIR}/LuisBonillaR-singularity-master-py3_tfstable_luis.simg"

# Would you like to train or predict?
#STAGE="train"
STAGE="predict"

# Python script executing tensorflow
EXE=estimator_hadmult_simple.py
# Directory where the checkpoint/models will be saved
#MODEL_DIR="/data/minerva/omorenop/tensorflow/models/vtx_data/VFplanecodes-194-8epoch"
#MODEL_DIR="/data/minerva/omorenop/tensorflow/models/vtx_data/ResNetplanecodes-213-10epochDSCALzLowCorrected/" #
#With the 0.01 z step improved
#MODEL_DIR="/data/minerva/omorenop/tensorflow/models/vtx_data/ResNetplanecodes-213-10epochDSCALzLowzHighCorrected/" #April
#Whole detector
MODEL_DIR="/data/minerva/omorenop/tensorflow/models/vtx_data/ResNetplanecodes-213-10epochWholeMix/"

# Number of classes that will be predicted (6 for hadron mult)
NCLASSES=214
# Number of events per batch that will be sent to train and eval
BATCH_SIZE=100
# Directory with our hdf5 files
DATA_DIR=/lfstev/e-938/omorenop/hdf5
# The name of our training/validation/prediction target:
# 'hadro_data/n_hadmultmeas', 'vtx_data/planecodes', etc
#TARGET=vtx_data/planecodesDScal
TARGET=vtx_data/planecodes

# Neural network architecture. Default 'VFNet' is vertex finding
#NET=VFNet
NET=ResNet

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# How many epochs do we want to train?
EPOCHS=10
# How many steps cover an epoch?
#STEPS_EPOCH=8708 #To 80-20%
#STEPS_EPOCH=8624  #To just ECAL
#STEPS_EPOCH=8784  #To DSCAL
#STEPS_EPOCH=151590 #To 90-10%
#STEPS_EPOCH=29190
#To DSCAL last modified
#STEPS_EPOCH=31570
#To whole Detector
STEPS_EPOCH=61343
# Epochs we want to train in step number
let TRAIN_STEPS=${EPOCHS}*${STEPS_EPOCH}
# How often  dowe want to save checkpoints/models?
let SAVE_STEPS=${STEPS_EPOCH}/2
# How many models should we keep (keeps latest)
SAVEDMODELS=10  #Usually 10
# Files for training. Can be multiple files. To especify files manually,
# separete them with a 'space': TRAIN_FILES=FILE1 FILE2 FILE3 etc
#TRAIN_FILES=${DATA_DIR}/me1Nmc/*
#TRAIN_FILES="/lfstev/e-938/omorenop/hdf5/me1Amc_All-626499/hadmultkineimgs_127x94_me1Amc.hdf5"
#TRAIN_FILES="/lfstev/e-938/omorenop/hdf5/NukeECAL-Train/hadmultkineimgs_127x94_me1Amc.hdf5" #JustECAL and 4 tracker modules
#TRAIN_FILES="/lfstev/e-938/omorenop/hdf5/NukeECAL-Train-lattice/hadmultkineimgs_127x94_me1Amc.hdf5" #Lattice fixed
#TRAIN_FILES="/data/omorenop/minerva/hdf5/201911/Nuke119-Train-040G/hadmultkineimgs_127x94_me1Amc.hdf5"
#TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train/hadmultkineimgs_127x94_me1AmcDSCALTrain.hdf5'
#TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train-Corrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train-RawCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train-VtxPlaneCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train-zLowHighApril/hadmultkineimgs_127x94_me1Amc.hdf5'
#Whole Detector
#TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train-WholeDetector/hadmultkineimgs_127x94_me1Amc.hdf5'
TRAIN_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Train-WholeDetectorMix/hadmultkineimgs_127x94_me1Amc.hdf5'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~VALIDATION-TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Number of steps for validation/testing
if [ $STAGE == "train" ] ; then
  VALID_STEPS=756 #500
elif [ $STAGE == "predict" ] ; then
  VALID_STEPS=7560 #Before 5000
fi
# model to use for prediction
#MODEL=model.ckpt-395350
#MODEL=model.ckpt-129360 #Used for ResNet 0.47 loss
#MODEL=model.ckpt-56056 #7 epochs Resnet 0.48 loss
#MODEL=model.ckpt-52704 #208 categories Resnet 0.917
#MODEL=model.ckpt-48312 #209 categories Res 0.916 loss 10 epochs
#MODEL=model.ckpt-52704 #210 Cat Res 0.946 loss 9 epochs (RawCorrected)
#MODEL=model.ckpt-57096 #198 Cat 0.949 loss 10 epochs
#MODEL=model.ckpt-57096 #32 Cat 0.95 loss 10 epochs

#MODEL=model.ckpt-204330 #
#MODEL=model.ckpt-160545 #For the new version
#MODEL=model.ckpt-205205 #After24 hrs for the prediction with 0.01 mm z accuracy
#MODEL=model.ckpt-552078 #368052 #184026
#MODEL=model.ckpt-398723 #To predict Whole detector
#MODEL=model.ckpt-245368
#MODEL=model.ckpt-337381
#MODEL=model.ckpt-521407
MODEL=model.ckpt-429394 #To predict Whole detector mix
# Files for validation/test. Can be multiple files. Can especify several files
# like in TRAIN_FILES
#EVAL_FILES=${DATA_DIR}/me1Fmc/*

#To evaluate
#EVAL_FILES="/data/omorenop/minerva/hdf5/201911/Nuke119-Evaluate-040G/hadmultkineimgs_127x94_me1Amc.hdf5"
#EVAL_FILES="/lfstev/e-938/omorenop/hdf5/NukeECAL-Evaluate/hadmultkineimgs_127x94_me1Amc.hdf5"
#EVAL_FILES="/lfstev/e-938/omorenop/hdf5/NukeECAL-Evaluate-lattice/hadmultkineimgs_127x94_me1Amc.hdf5"
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate/hadmultkineimgs_127x94_me1AmcDSCALEvaluate.hdf5 '
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-Corrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-RawCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-VtxPlaneCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-zLowHighApril/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-WholeDetector/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Evaluate-WholeDetectorMix/hadmultkineimgs_127x94_me1Amc.hdf5'

#To predict
#EVAL_FILES="/data/omorenop/minerva/hdf5/201911/Nuke119-Test-0402/hadmultkineimgs_127x94_me1Amc.hdf5"
#EVAL_FILES="/lfstev/e-938/omorenop/hdf5/NukeECAL-Test/hadmultkineimgs_127x94_me1Amc.hdf5"
#EVAL_FILES="/lfstev/e-938/omorenop/hdf5/NukeECAL-Test-lattice/hadmultkineimgs_127x94_me1Amc.hdf5"
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test/hadmultkineimgs_127x94_me1AmcDSCALTest.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-Corrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-RawCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-VtxPlaneCorrected/hadmultkineimgs_127x94_me1Amc.hdf5'
#EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-zLowHighApril/hadmultkineimgs_127x94_me1Amc.hdf5'
EVAL_FILES='/lfstev/e-938/omorenop/hdf5/NukeDSCAL-Test-WholeDetectorMix/hadmultkineimgs_127x94_me1Amc.hdf5'

# We create our MODEL_DIR
if [ ! -d "$MODEL_DIR" ]
then
  mkdir $MODEL_DIR
fi

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SET THE SCRIPT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# String with arguments for training, validation or test
ARGS="--batch-size ${BATCH_SIZE}" #100
ARGS+=" --nclasses ${NCLASSES}" #195
ARGS+=" --train-steps ${TRAIN_STEPS}" #epochs * steps_epoch
ARGS+=" --valid-steps ${VALID_STEPS}" #500 to train
ARGS+=" --save-steps ${SAVE_STEPS}" #steps_epoch/2
ARGS+=" --train-files ${TRAIN_FILES}"
ARGS+=" --eval-files ${EVAL_FILES}"
ARGS+=" --target-field ${TARGET}"
ARGS+=" --cnn ${NET}"
ARGS+=" --model-dir ${MODEL_DIR}"
ARGS+=" --saved-models ${SAVEDMODELS}"
ARGS+=" --model ${MODEL}"
if [ $STAGE == "train" ] ; then
  ARGS+=" --do-train"
elif [ $STAGE == "predict" ] ; then
  ARGS+=" --do-test"
else
  echo "STAGE must be 'train' or 'predict'."
  exit 0
fi

# Show the command to be executed
cat << EOF
singularity exec --nv $SINGLRTY python3 $EXE $ARGS
EOF

# Execute the command
singularity exec --nv $SINGLRTY python3  $EXE $ARGS
#-m cProfile --sort cumulative
nvidia-smi

echo "finished "`date`" "`date +%s`""
exit 0

# Singularity containers
#SINGLRTY='/lfstev/e-938/jbonilla/sing_imgs/LuisBonillaR-singularity-master-py3_tfstable_luis.simg'
#SINGLRTY='/data/aghosh12/local-withdata.simg'
#SINGLRTY='/data/perdue/singularity/gnperdue-singularity_imgs-master-py2_tf18.simg '
#SINGLRTY='/lfstev/e-938/jbonilla/sing_imgs/LuisBonillaR-singularity-master-pyhon3_luisb.simg'
