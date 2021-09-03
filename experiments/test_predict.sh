#!/bin/bash

inppath=$1
OUTFOLDER=$2
tfpath=$3
MPATH=$4
MNAME=$5
bert=$6
myFileNames=$(ls $1)
for i in $myFileNames; do
	NAME=$(basename $i .jsonl.gz)
	# echo $NAME $inppath $OUTFOLDER $tfpath
    PREDFILE=$inppath'/'$NAME'.jsonl.gz'
    TFFILE=$tfpath'/'$NAME'.tfrecord'
    MPATH=$5
    MNAME=$6
    OUTFILE='pred_'$NAME'_'$MNAME'.jsonl'
    echo $NAME
    echo $PREDFILE
    echo $OUTFILE
    echo $OUTFOLDER
    echo $TFFILE
    echo $MPATH
    echo $MNAME

    python3 ../baselines/tydiqa/baseline/run_tydi.py \
    --bert_config_file=$6/bert_config.json \
    --vocab_file=..baselines/tydiqa/baseline/mbert_modified_vocab.txt \
    --init_checkpoint=$MPATH \
    --predict_file=$PREDFILE \
    --precomputed_predict_file=$TFFILE \
    --do_predict \
    --output_dir=$OUTFOLDER \
    --output_prediction_file=$OUTFOLDER/$OUTFILE

done




