# SD-QA

### Data File Structure
```
dev/
    -lang/
        -language.dev.csv
        -language.dev.txt
        -dialect/
            -language.dev.dialect-ASR.txt.jsonl.gz
            -language.dev.dialect-ASR.txt
            -metadata.csv
            -wav_lang/
    	        -ID.wav

test/
    -lang/
        -language.test.csv
        -language.test.txt
        -dialect/
            -language.test.dialect-ASR.txt.jsonl.gz
            -language.test.dialect-ASR.txt
            -metadata.csv
            -wav_lang/
    	        -ID.wav

-asr_metadata/
	-dev/
		-asr_output_with_metadata_lang.csv
	-test/
		-asr_output_with_metadata_lang.csv
```

- lang: eg. eng, ara
- language.dev.csv: language specific csv file containing gold and ASR transcripts for all dialects
- language.dev.txt: language specific text file containing gold data
- language.dev.dialect-ASR.txt.jsonl.gz: Language and dialect specific TyDi-QA format datafile (gold question replaced with transcript)
- metadata.csv: metadata file (example ID-user ID mapping with additional info for each dialect and language)
- wav_lang: folder containing audio files
- asr_output_with_metadata_lang.csv: single language specific csv file containing all metadata, transcripts with word error rate for each example instance 

## WER  based evaluation on ASR outputs

- see [`experiments-wer.ipynb`](https://github.com/ffaisal93/SD-QA/blob/master/experiments/experiments-wer.ipynb)

## Comparative minimal answer predictions for Error analysis 
- see [`error_analysis`](https://github.com/ffaisal93/SD-QA/tree/master/tydiqa_data/error_analysis)

## Baseline-TydiQA 
We train a tydiqa baseline model for the primary task evaluation. Instead of using the original training data, we use the  `discard_dev` version (SDQA development questions are discarded from the training data).

Available model and training data for download:
- [Trained checkpoint](https://drive.google.com/drive/folders/1B0JSZW3PWCXAyZCfZBHWkq4Rh0xUJukL?usp=sharing): Downolad to the folder `trained_models/`
- [Original training/development dataset](https://drive.google.com/drive/folders/1z3G5HJ25m46EykLbt6KkvNPk06wSJs8x?usp=sharing): Download to the folder `tydiqa_data/`
- [`discard_dev` training dataset](https://drive.google.com/drive/folders/1kSMx07w4FGKRKOtlDLHBVKkLv_HTFuQN?usp=sharing): Download to the folder `tydiqa_data/`
- [language specific dev dataset replaced with asr output](https://github.com/ffaisal93/SD-QA/tree/master/dev): Can be constructed using [`construct-tydiqa-datafile.ipynb`](https://github.com/ffaisal93/SD-QA/blob/master/experiments/construct-tydiqa-datafile.ipynb)
- [language specific test dataset replaced with asr output](https://github.com/ffaisal93/SD-QA/tree/master/test): Can be constructed using [`construct-tydiqa-datafile.ipynb`](https://github.com/ffaisal93/SD-QA/blob/master/experiments/construct-tydiqa-datafile.ipynb)

### Experimenting with a primary task baseline
Detailed steps to train a tydiqa primary task baseline model is [here](https://github.com/ffaisal93/SD-QA/tree/master/baselines/tydiqa/baseline) 

##### prepare the training samples:
```
python3 baselines/tydiqa/baseline/prepare_tydi_data.py \
  --input_jsonl=tydiqa_data/tydiqa-v1.0-train-discard-dev.jsonl.gz \
  --output_tfrecord=tydiqa_data/train_tf/train_samples.tfrecord \
  --vocab_file=baselines/tydiqa/baseline/mbert_modified_vocab.txt \
  --record_count_file=tydiqa_data/train_tf/train_samples_record_count.txt \
  --include_unknowns=0.1 \
  --is_training=true
```
##### prepare dev samples from all language-dialect specific asr outputs
```
./experiments/test_prep.sh tydiqa_data/dev tydiqa_data/dev_tf
```
##### prepare test samples from all language-dialect specific asr outputs
```
./experiments/test_prep.sh tydiqa_data/test tydiqa_data/test_tf
```
##### train

```
python3 baselines/tydiqa/baseline/run_tydi.py \
  --bert_config_file=mbert_dir/bert_config.json \
  --vocab_file=baselines/tydiqa/baseline/mbert_modified_vocab.txt \
  --init_checkpoint=mbert_dir/bert_model.ckpt \
  --train_records_file=tydiqa_data/train_tf/train_samples.tfrecord \
  --record_count_file=tydiqa_data/train_tf/train_samples_record_count.txt \
  --do_train \
  --output_dir=trained_models/
```

##### Predict

Once the model is trained, we run inference on the dev/test set:

dev:
```
./experiments/test_predict.sh \
tydiqa_data/dev tydiqa_data/dev_predict tydiqa_data/dev_tf \
trained_models/model.ckpt discard_dev mbert_dir
```

test:
```
./experiments/test_predict.sh \
tydiqa_data/test tydiqa_data/test_predict tydiqa_data/test_tf \
trained_models/model.ckpt discard_dev mbert_dir
```


- to point the trained checkpoint at `--init_checkpoint`, write correct location inplace of `trained_models/model.ckpt`
- write downloaded mbert location inplace of `mbert_dir`


##### Evaluate
- see [`tydiqa_evaluation.ipynb`](https://github.com/ffaisal93/SD-QA/blob/master/experiments/tydiqa_evaluation.ipynb)


## Citation
If you use SD-QA, please cite the "[SD-QA: Spoken Dialectal Question Answering for the Real World](https://arxiv.org/abs/2109.12072)". You can use the following BibTeX entry
~~~
@inproceedings{faisal-etal-21-sdqa,
 title = {{SD-QA}: {S}poken {D}ialectal {Q}uestion {A}nswering for the {R}eal {W}orld},
  author = {Faisal, Fahim and Keshava, Sharlina and ibn Alam, Md Mahfuz and Anastasopoulos, Antonios},
  url={https://arxiv.org/abs/2109.12072},
  year = {2021},
  booktitle = {Findings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP Findings)},
  publisher = {Association for Computational Linguistics},
  month = {November},
}
~~~

We built our augmented dataset and baselines on top of TydiQA. Kindly also make sure to cite the original TyDi QA paper,
~~~
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
journal = {TACL},
year    = {2020}
}
~~~

## License
Both the code and data for SD-QA are availalbe under the [Apache License 2.0](LICENSE).
