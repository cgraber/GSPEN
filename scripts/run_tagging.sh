# These scripts run the tagging experiments

# Make sure to set this to the desired GPU
GPU=0


IMG_PATH='data/tagging/mirflickr/'
LABELS_PATH='data/tagging/mirflickr25k_annotations_v080/'
UNARY_DATA_ARGS="--img_dir ${IMG_PATH} --labels_dir ${LABELS_PATH}"
STRUCT_DATA_ARGS="--img_dir ${IMG_PATH} --labels_dir ${LABELS_PATH}"

# OPTIONAL: if you plan on training many models, you can pre-process the data once and then load it. Just uncomment the following

#TRAIN_UNARY_FILE="data/tagging/train_unary_data"
#TEST_UNARY_FILE="data/tagging/test_unary_data"
#VAL_UNARY_FILE="data/tagging/val_unary_data"
#UNARY_DATA_ARGS="--train_data_file ${TRAIN_UNARY_FILE} --val_data_file ${VAL_UNARY_FILE} --test_data_file ${TEST_UNARY_FILE} --load_data"

#TRAIN_STRUCT_FILE="data/tagging/train_struct_data"
#TEST_STRUCT_FILE="data/tagging/test_struct_data"
#VAL_STRUCT_FILE="data/tagging/val_struct_data"
#STRUCT_DATA_ARGS="--train_data_file ${TRAIN_STRUCT_FILE} --val_data_file ${VAL_STRUCT_FILE} --test_data_file ${TEST_STRUCT_FILE} --load_data"
#RESULTS_DIR='tmp/'
#python -u experiments/train_tagging.py spen $RESULTS_DIR --process_data $UNARY_DATA_ARGS --img_dir ${DATA_PATH} --labels_dir ${LABELS_PATH}
#python -u experiments/train_tagging.py gspen $RESULTS_DIR --process_data $STRUCT_DATA_ARGS --img_dir ${DATA_PATH} --labels_dir ${LABELS_PATH}

WORKING_DIR="tmp_results/tagging/"
mkdir -p $WORKING_DIR

UNARY_PRETRAIN_PATH="/home/cgraber2/Code/NLStruct/tmp_tagging/unary/full_unary_model"
STRUCT_PRETRAIN_PATH="/home/cgraber2/Code/NLStruct/neurips2018_scripts/tagging_models/struct_model"

# SPEN
RESULTS_DIR="${WORKING_DIR}/spen/"
mkdir -p $RESULTS_DIR
COMMON_PARAMS="--num_epochs 500 --val_interval 1 --batch_size 128 --gpu --unary_lr 0 --use_linear_decay --inf_lr 1. --num_inf_itrs 100 --t_version t_v1 --inf_method fw --t mlp --inf_eps 1e-4 --ignore_unary_dropout --t_use_softplus --use_old_unary"
PRETRAIN_ARGS="--load_old_unary ${UNARY_PRETRAIN_PATH}"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_tagging.py spen $RESULTS_DIR $UNARY_DATA_ARGS $COMMON_PARAMS $PRETRAIN_ARGS --t_unary_lr 1e-2 --t_unary_mom 0.9 --use_loss_aug |& tee "${RESULTS_DIR}results.txt"
PRETRAIN_ARGS="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_unary_t ${RESULTS_DIR}unary_t_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_tagging.py spen $RESULTS_DIR $UNARY_DATA_ARGS $COMMON_PARAMS $PRETRAIN_ARGS --test |& tee "${RESULTS_DIR}test_results.txt"


# GSPEN
RESULTS_DIR="${WORKING_DIR}/gspen/"
mkdir -p $RESULTS_DIR
COMMON_PARAMS="--num_epochs 200 --val_interval 1 --batch_size 128 --gpu --combined_lr 0 --use_linear_decay --inf_lr 1. --num_inf_itrs 100 --t_version full_t_v1 --inf_mode mp --mp_eps 0. --mp_itrs 100 --t mlp --inf_eps 1e-4 --pair_finetune --use_combined_struct_old --ignore_unary_dropout --t_use_softplus --no_t_pots"
PRETRAIN_ARGS="--load_old_struct ${STRUCT_PRETRAIN_PATH}"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_tagging.py gspen $RESULTS_DIR $STRUCT_DATA_ARGS $COMMON_PARAMS $PRETRAIN_ARGS --t_lr 1e-2 --t_mom 0.9 --use_loss_aug |& tee "${RESULTS_DIR}results.txt"
PRETRAIN_ARGS="--pretrain_t ${RESULTS_DIR}t_model --load_old_struct ${STRUCT_PRETRAIN_PATH}"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_tagging.py gspen $RESULTS_DIR $STRUCT_DATA_ARGS $COMMON_PARAMS $PRETRAIN_ARGS --test |& tee "${RESULTS_DIR}test_results.txt"

