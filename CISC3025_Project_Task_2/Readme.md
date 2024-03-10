# CISC3003-Project-Task 2

## Results
To view results, please head to the `./output/` directory.
1. [word count](./output/word_count.txt): num(c) for all c in C; num(w) for all w in V.
2. [word_dict](./output/word_dict.txt): num(w in c) for all w in V and c in C; Selected top 10,000 records from word_count.
3. [word_probability](./output/word_probability.txt): P(c) for all c in C; P(w|c) for all w and c;
4. [classification_result](./output/classification_result.txt): Results for classifying test set.
5. [f_scores](./output/f_scores.txt): Model evaluation using both macro & micro average F-scores.

## Programs
To run programs, all you have to do is to click the run button.

Run programs and inspect the terminal for in-process data:
1. [count_word.py](./count_word.py): Writes into [word count](./output/word_count.txt).
2. [feature_selection.py](./feature_selection.py): Writes into [word_dict](./output/word_dict.txt)
3. [word_probability.py](./word_probability.py): Writes into  [word_probability](./output/word_probability.txt).
4. 