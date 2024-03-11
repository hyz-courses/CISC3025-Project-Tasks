# Package Imports
import json
import re
import nltk
import math
import pandas as pd
import numpy as np
from collections import Counter
import datetime

# Local Imports
import __funcs__
from count_word import count_word as _count_word
from feature_selection import feature_selection as _feature_selection
from word_probability import word_probability as _word_probability
from classification import classification as _classification
from evaluation import evaluation as _evaluation

file_paths = {
    "TRAINING_DATA": "./data/train.json",
    "WORD_COUNT": "./output/word_count.txt",
    "WORD_DICT": "./output/word_dict.txt",
    "WORD_PROB": "./output/word_probability.txt",
    "WORD_PROB_LOG": "./temp_output/word_probability_log.txt",
    "TEST_DATA": "./data/test.json",
    "CLASSIFY_RESULTS": "./output/classification_result.txt",
    "CLASSIFY_COMPARE": "./temp_output/classification_compare.txt",
    "F_SCORE": "./output/f_scores.txt"
}


def main():
    __funcs__.console_log_title("Running count_word.py")
    _count_word(file_paths["TRAINING_DATA"], file_paths["WORD_COUNT"])

    __funcs__.console_log_title("Running feature_selection.py")
    _feature_selection(file_paths['WORD_COUNT'], threshold=10000, output_file_path=file_paths['WORD_DICT'])

    __funcs__.console_log_title("Running word_probability.py")
    _word_probability(
        [file_paths['WORD_COUNT'], file_paths['WORD_DICT']],
        output_file=file_paths['WORD_PROB'],
        add_alpha=1,
        use_log=False
        # precision=4   # Control digits after dot.
    )

    _word_probability(
        [file_paths['WORD_COUNT'], file_paths['WORD_DICT']],
        output_file=file_paths['WORD_PROB_LOG'],
        add_alpha=1,
        use_log=True
        # precision=4   # Control digits after dot.
    )

    __funcs__.console_log_title("Running classification.py")
    _classification(file_paths['TEST_DATA'], file_paths['CLASSIFY_RESULTS'], file_paths['CLASSIFY_COMPARE'])

    __funcs__.console_log_title("Running evaluation.py")
    _evaluation(file_paths["CLASSIFY_COMPARE"], file_paths["F_SCORE"])

    print("Current Time: ", end=" ")
    print(datetime.datetime.now())


main()
