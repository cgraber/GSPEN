# These scripts run the bookmarks experiments

# Make sure to set this to the desired GPU
GPU=0

DATA_PATH='data/bookmarks/'

WORKING_DIR="tmp_results/bookmarks/"
mkdir -p $WORKING_DIR

# UNARY
RESULTS_DIR="${WORKING_DIR}unary/"
mkdir -p $RESULTS_DIR
COMMON_PARAMS="--num_epochs 10000 --batch_size 128 --gpu --unary_dropout 0.5 --unary_num_layers 3 --flip_prob 0.01"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py unary $DATA_PATH $RESULTS_DIR $COMMON_PARAMS --clip_grad_norm 1 --tune_thresholds --use_cross_ent --unary_lr 1e-2 --unary_mom 0.9  |& tee "${RESULTS_DIR}results.txt"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py unary $DATA_PATH $RESULTS_DIR $COMMON_PARAMS --clip_grad_norm 1 --tune_thresholds --use_cross_ent --test --pretrain_unary "${RESULTS_DIR}unary_model" |& tee "${RESULTS_DIR}test_results.txt"


# STRUCT
RESULTS_DIR="${WORKING_DIR}struct/"
mkdir -p $RESULTS_DIR
UNARY_PRETRAIN="${WORKING_DIR}unary/unary_model"
COMMON_PARAMS="--num_epochs 100 --batch_size 128 --gpu --inf_mode mp --tune_thresholds --unary_dropout 0.5 --unary_num_layers 3 --mp_eps 1. --mp_itrs 5 --val_interval 5 --use_loss_aug --pair_num_layers 2 --pair_hidden_size 1000"
PRETRAIN_ARGS="--pretrain_unary ${UNARY_PRETRAIN}" 
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py struct $DATA_PATH $RESULTS_DIR $COMMON_PARAMS $PRETRAIN_ARGS --unary_lr 0.  --pair_lr 1e-4 --use_adam  |& tee "${RESULTS_DIR}results.txt"
TEST_PRETRAIN="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_pair ${RESULTS_DIR}pair_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py struct $DATA_PATH $RESULTS_DIR $COMMON_PARAMS $TEST_PRETRAIN --test |& tee "${RESULTS_DIR}test_results.txt"


# SPEN
RESULTS_DIR="${WORKING_DIR}spen/"
mkdir -p $RESULTS_DIR
UNARY_PRETRAIN="${WORKING_DIR}unary/unary_model"
COMMON_PARAMS="--num_epochs 500 --batch_size 128 --gpu --inf_lr 1 --use_sqrt_decay --num_inf_itrs 100 --inf_eps 1e-4 --use_entropy --inf_method emd --inf_mode md --t_version t_v1 --tune_thresholds --unary_dropout 0.5 --unary_num_layers 3 --entropy_coef 0.1"
PRETRAIN_ARGS="--pretrain_unary ${UNARY_PRETRAIN}"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py spen  $DATA_PATH $RESULTS_DIR $COMMON_PARAMS $PRETRAIN_ARGS --unary_lr 0 --t_unary_lr 1e-4 --use_adam  |& tee "${RESULTS_DIR}results.txt"
PRETRAIN_ARGS="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_unary_t ${RESULTS_DIR}unary_t_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py spen  $DATA_PATH $RESULTS_DIR $COMMON_PARAMS $PRETRAIN_ARGS --test |& tee "${RESULTS_DIR}test_results.txt"


# GSPEN
RESULTS_DIR="${WORKING_DIR}gspen/"
mkdir -p $RESULTS_DIR
UNARY_PRETRAIN="${WORKING_DIR}spen/unary_model"
UNARY_T_PRETRAIN="${WORKING_DIR}spen/unary_t_model"
COMMON_PARAMS="--num_epochs 100 --batch_size 128 --gpu --inf_lr 1 --num_inf_itrs 100 --inf_eps 1e-4 --inf_method emd --inf_mode md --use_sqrt_decay --t_version t_v1 --tune_thresholds --unary_dropout 0.5 --unary_num_layers 3 --mp_eps 1. --entropy_coef 0.1 --mp_itrs 5 --val_interval 5 --use_loss_aug --use_entropy --pair_num_layers 2 --pair_hidden_size 1000"
PRETRAIN_ARGS="--pretrain_unary ${UNARY_PRETRAIN} --pretrain_unary_t ${UNARY_T_PRETRAIN}" 
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py gspen $DATA_PATH $RESULTS_DIR $COMMON_PARAMS $PRETRAIN_ARGS --unary_lr 0. --t_unary_lr 0. --pair_lr 1e-4 --use_adam  |& tee "${RESULTS_DIR}results.txt"
TEST_PRETRAIN="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_pair ${RESULTS_DIR}pair_model --pretrain_unary_t ${RESULTS_DIR}unary_t_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_bookmarks.py gspen $DATA_PATH $RESULTS_DIR $COMMON_PARAMS $TEST_PRETRAIN --test |& tee "${RESULTS_DIR}test_results.txt"


