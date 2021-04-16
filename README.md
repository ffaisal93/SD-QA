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
- language.dev.csv: language specific csv file containing gold and transcript columns for all dialect
- language.dev.txt: language specific text file containing gold data
- language.dev.dialect-ASR.txt.jsonl.gz: Language and dialect specific TyDi-QA format datafile (gold question replaced with transcript)
- metadata.csv: metadata file (example ID-user ID mapping with additional info for each dialect and language)
- wav_lang: folder containing audio files
- asr_output_with_metadata_lang.csv: single language specific csv file containing all metadata, transcriptions with word error rate for each example instance 