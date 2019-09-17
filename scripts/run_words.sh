# These scripts run the word recognition experiments for 1000 train datapoints for an interpolation factor of 0.5

# Make sure to set this to the desired GPU
GPU=0

DATA_DIR="data/words_neurips19_5/"
WORKING_DIR="tmp_results/words/"
mkdir -p $WORKING_DIR

# UNARY
RESULTS_DIR="${WORKING_DIR}unary/"
mkdir -p $RESULTS_DIR
MODEL_PARAMS="--unary_num_layers 3 --unary_dropout 0.5 --unary_hidden_size 200"
TRAINING_PARAMS="--num_epochs 1000 --batch_size 128 --use_cross_ent --clip_grad_norm 1. --unary_lr 1e-4 --use_adam --val_interval 1 --gpu"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py unary small $DATA_DIR $RESULTS_DIR $MODEL_PARAMS $TRAINING_PARAMS |& tee "${RESULTS_DIR}results.txt"
TEST_PRETRAIN_ARGS="--pretrain_unary ${RESULTS_DIR}unary_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py unary small $DATA_DIR $RESULTS_DIR $MODEL_PARAMS $TRAINING_PARAMS --test $TEST_PRETRAIN_ARGS |& tee "${RESULTS_DIR}test_results.txt"

# STRUCT
PRETRAIN_UNARY="${WORKING_DIR}unary/unary_model"
RESULTS_DIR="${WORKING_DIR}struct/"
UNARY_PARAMS="--unary_num_layers 3 --unary_hidden_size 200"
TRAINING_PARAMS="--num_epochs 2000 --batch_size 128 --clip_grad_norm 1. --unary_lr 0 --use_adam --pair_lr 1e-3 --gpu --unary_dropout 0.5"
PRETRAIN_ARGS="--pretrain_unary $PRETRAIN_UNARY"
INF_PARAMS="--inf_mode mp --mp_eps 1. --mp_itrs 10 --use_loss_aug"
MODEL_PARAMS="$UNARY_PARAMS --use_separate_pairs"
mkdir -p $RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py struct small $DATA_DIR $RESULTS_DIR $TRAINING_PARAMS $INF_PARAMS $MODEL_PARAMS $PRETRAIN_ARGS |& tee "${RESULTS_DIR}results.txt"
TEST_PRETRAIN_ARGS="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_pair ${RESULTS_DIR}pair_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py struct small $DATA_DIR $RESULTS_DIR $INF_PARAMS $MODEL_PARAMS $TRAINING_PARAMS --test $TEST_PRETRAIN_ARGS |& tee "${RESULTS_DIR}test_results.txt"

# SPEN
PRETRAIN_UNARY="${WORKING_DIR}unary/unary_model"
RESULTS_DIR="${WORKING_DIR}spen/"
UNARY_PARAMS="--unary_num_layers 3 --unary_hidden_size 200"
TRAINING_PARAMS="--num_epochs 2000 --batch_size 128 --clip_grad_norm 1. --unary_lr 0 --use_adam --t_unary_lr 1e-4 --gpu --unary_dropout 0.5"
PRETRAIN_ARGS="--pretrain_unary $PRETRAIN_UNARY"
INF_PARAMS="--inf_method emd --use_sqrt_decay --inf_lr 1. --num_inf_itrs 10000 --inf_eps 1e-4 --use_entropy --use_loss_aug"
MODEL_PARAMS="$UNARY_PARAMS --t_version t_v1 --t_unary mlp --t_use_softplus --t_hidden_size 1000"
mkdir -p $RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py spen small $DATA_DIR $RESULTS_DIR $TRAINING_PARAMS $INF_PARAMS $MODEL_PARAMS $PRETRAIN_ARGS |& tee "${RESULTS_DIR}results.txt"
TEST_PRETRAIN_ARGS="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_unary_t ${RESULTS_DIR}unary_t_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py spen small $DATA_DIR $RESULTS_DIR $INF_PARAMS $MODEL_PARAMS $TRAINING_PARAMS --test $TEST_PRETRAIN_ARGS |& tee "${RESULTS_DIR}test_results.txt"

# GSPEN
PRETRAIN_UNARY="${WORKING_DIR}/struct/unary_model"
PRETRAIN_PAIR="${WORKING_DIR}/struct/pair_model"
RESULTS_DIR="${WORKING_DIR}spen/"
STRUCT_PARAMS="--unary_num_layers 3 --unary_hidden_size 200 --use_separate_pairs"
TRAINING_PARAMS="--num_epochs 1000 --batch_size 128 --clip_grad_norm 1. --unary_lr 0 --pair_lr 0 --t_lr 1e-4 --use_adam --gpu --unary_dropout 0.5"
PRETRAIN_ARGS="--pretrain_unary $PRETRAIN_UNARY --pretrain_pair $PRETRAIN_PAIR"
INF_PARAMS="--inf_mode md --mp_eps 1. --mp_itrs 10 --num_inf_itrs 100 --inf_lr 1. --use_sqrt_decay --inf_eps 1e-4 --use_loss_aug --use_entropy"
MODEL_PARAMS="$STRUCT_PARAMS --t_version full_t_v1 --t_unary mlp --t_num_layers 2 --t_hidden_size 1000 --t_use_softplus"
mkdir -p $RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py gspen small $DATA_DIR $RESULTS_DIR $TRAINING_PARAMS $INF_PARAMS $MODEL_PARAMS $PRETRAIN_ARGS |& tee "${RESULTS_DIR}results.txt"
TEST_PRETRAIN_ARGS="--pretrain_unary ${RESULTS_DIR}unary_model --pretrain_t ${RESULTS_DIR}t_model --pretrain_pair ${RESULTS_DIR}pair_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/train_words.py gspen small $DATA_DIR $RESULTS_DIR $INF_PARAMS $MODEL_PARAMS $TRAINING_PARAMS --test $TEST_PRETRAIN_ARGS |& tee "${RESULTS_DIR}test_results.txt"
