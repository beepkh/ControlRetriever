export CUDA_VISIBLE_DEVICES=0

BEIR_DIR="PATH_TO_BEIR_DATA"
OUTPUT_DIR="./retrieval"
MODEL_NAME="PATH_TO_COCODR-LARGE"
POOLING="cls"
BATCH_SIZE=256
SCORE_FUC="dot"
OUTPUT_DIR="./retrieval"

OPTS=""
OPTS+=" --model_name_or_path ${MODEL_NAME}"
OPTS+=" --d_model_path ${MODEL_NAME}"
OPTS+=" --pooling ${POOLING}"
OPTS+=" --score_function ${SCORE_FUC}"
OPTS+=" --batch_size ${BATCH_SIZE}"
OPTS+=" --output_dir ${OUTPUT_DIR}"
OPTS+=" --beir_dir ${BEIR_DIR}"
OPTS+=" --result_dir ${RESULT_DIR}"
OPTS+=" --use_instruction"

datasets=(arguana)
MODEL_PATH="./checkpoint/"

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
    OPTSSUB="${OPTS}"
    OPTSSUB+=" --dataset ${dataset}"
    OPTSSUB+=" --q_model_path ${MODEL_PATH}"
    OPTSSUB+=" --max_seq_length_q ${MAX_Q}"
    OPTSSUB+=" --max_seq_length_d ${MAX_D}"
    OPTSSUB+=" --max_seq_length_c ${MAX_C}"
    echo $OPTSSUB
    python ./evaluate_retrieval.py ${OPTSSUB}
done
