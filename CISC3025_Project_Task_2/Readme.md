# CISC3003-Project-Task 2

## Author:
Huang Yanzhen, DC126732 on CISC3025 - Natural Language Processing.

## Results
To view results, please head to the `./output/` directory.
1. [word count](./output/word_count.txt): **First line:** Doc freq for all classes; **Other:** Word freqs in each class for all word types.
2. [word_dict](./output/word_dict.txt): **First line:** Word freq for all classes; **Other:** Selected top 10,000 records from word_count.
3. [word_probability](./output/word_probability.txt): **First line:** Prior probability P(c) for all c in C; **Other:** Posterior probability P(w|c) for all w and c;
4. [classification_result](./output/classification_result.txt): Results for classifying test set.
5. [f_scores](./output/f_scores.txt): Model evaluation using both macro & micro average F-scores.

To view temporary outputs, please head to `./temp_output` directory.
1. [word_probability_log](./temp_output/word_probability_log.txt): `word_probability.txt` in negative logarithmic space.
2. [classification_compare](./temp_output/classification_compare.txt): `classification_result.txt` but all instances have both prediction and actual.

## Programs
To go through the entire process of model construction, model testing and model evaluation: 

Please kindly head to [main.py](main.py) and click "run". 

To run individual steps of model construction, please head to one of the following:
1. [count_word.py](./count_word.py): Writes into [word count](./output/word_count.txt).
2. [feature_selection.py](./feature_selection.py): Writes into [word_dict](./output/word_dict.txt)
3. [word_probability.py](./word_probability.py): Model Construction. Writes into  [word_probability](./output/word_probability.txt).
4. [classification.py](./classification.py): Model testing. Writes into [classification_result](./output/classification_result.txt).
5. [evaluation.py](evaluation.py): Model evaluation. Writes into [f_scores](./output/f_scores.txt).

## Settings
There are some custom settings in [__funcs__.py](__funcs__.py). 
- Don't want to print intermediate data? Set `__funcs__.settings['PRINT_PROCESS']` to `False`.
- Want to run model without writing files? Set `__funcs__.settings['WRITE_FILES']` to `False`.