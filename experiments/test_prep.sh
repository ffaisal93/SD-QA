#!/bin/bash
#filename input_path output_path
myFileNames=$(ls $1)
inputpath=$1
outputpath=$2
echo $myFileNames
for i in $myFileNames; do
	# NAME="${i:0: -9}"
	NAME=$(basename $i .jsonl.gz)
    INPUT=$inputpath'/'$NAME'.jsonl.gz'
    OUTPUT=$outputpath'/'$NAME'.tfrecord'
    python3 ../baselines/tydiqa/baseline/prepare_tydi_data.py \
    --input_jsonl=$INPUT \
    --output_tfrecord=$OUTPUT \
    --vocab_file=../baselines/tydiqa/baseline/mbert_modified_vocab.txt \
    --is_training=false

done