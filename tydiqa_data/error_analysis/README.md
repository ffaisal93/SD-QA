## Comparative minimal answer predictions for Error analysis 

- Predictions on our custom test data.
- Only those predictions are reported, where `gold_min_start >= 0` and prediction not null.
- `{language}-gold` column is for the predictions on original tydiqa questions.
- Column naming: `gbr-en-US` means `dialectal_region`:`gbr`, `google asr unit`:`en-US` (means spoken question was collected from `gbr` region and transcribed using google speech-to-text language unit `en-US`).