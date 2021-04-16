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
```

- lang: eg. eng, ara
- language.dev.csv: language specific csv file containing gold and ASR transcripts for all dialects
- language.dev.txt: language specific text file containing gold data
- language.dev.dialect-ASR.txt.jsonl.gz: Language and dialect specific TyDi-QA format datafile (gold question replaced with transcript)
- metadata.csv: metadata file (example ID-user ID mapping with additional info for each dialect and language)
- wav_lang: folder containing audio files
- asr_output_with_metadata_lang.csv: single language specific csv file containing all metadata, transcripts with word error rate for each example instance 


## Experiments
Code to replicate all experiments will be soon released under the `experiments` directory.
## Citation
If you use SD-QA, please cite the "[SD-QA: Spoken Dialectal Question Answering for the Real World](https://cs.gmu.edu/~antonis/publication/faisal-etal-21-sdqa/SD-QA.pdf)". You can use the following BibTeX entry
~~~
@misc{faisal-etal-21-sdqa,
 title = {{SD-QA}: {S}poken {D}ialectal {Q}uestion {A}nswering for the {R}eal {W}orld},
  author = {Faisal, Fahim and Keshava, Sharlina and ibn Alam, Md Mahfuz and Anastasopoulos, Antonios},
  url={https://cs.gmu.edu/~antonis/publication/faisal-etal-21-sdqa/SD-QA.pdf},
  year = {2021},
  note = {preprint}
}
~~~