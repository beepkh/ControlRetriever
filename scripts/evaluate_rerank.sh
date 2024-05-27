export CUDA_VISIBLE_DEVICES=0

BEIR_DIR="PATH_TO_BEIR_DATA"
OUTPUT_DIR="./rerank/topk100/"
MODEL_NAME="PATH_TO_COCODR-LARGE"
MODEL_PATH="./checkpoint/"


POOLING="cls"
BATCH_SIZE=256
SCORE_FUC="dot"
OUTPUT_DIR="./rerank/topk100/"

MAX_D=128
MAX_Q=64
MAX_C=128

datasets=(webis-touche2020)

for dataset in ${datasets[@]}
do
    MAX_D=128
    MAX_Q=64
    MAX_C=128
    if [ $dataset == 'arguana' ]
        then
            MAX_Q=128
            MAX_C=150
    else
            MAX_Q=64
            MAX_C=128
    fi

    if [ $dataset == 'scifact' ]
        then
            MAX_D=256
    else
            MAX_D=128
    fi

    OPTS=""
    OPTS+=" --k_values 10 100"
    OPTS+=" --model_name_or_path ${MODEL_NAME}"
    OPTS+=" --d_model_path ${MODEL_NAME}"
    OPTS+=" --q_model_path ${MODEL_PATH}"
    OPTS+=" --pooling ${POOLING}"
    OPTS+=" --score_function ${SCORE_FUC}"
    OPTS+=" --batch_size ${BATCH_SIZE}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --beir_dir ${BEIR_DIR}"
    OPTS+=" --dataset ${dataset}"
    OPTS+=" --max_seq_length_q ${MAX_Q}"
    OPTS+=" --max_seq_length_d ${MAX_D}"
    OPTS+=" --max_seq_length_c ${MAX_C}"
    OPTS+=" --use_instruction"
    OPTS+=" --result_dir ${RESULT_DIR}"
    echo $OPTS
    python ./evaluate_rerank.py ${OPTS}

done