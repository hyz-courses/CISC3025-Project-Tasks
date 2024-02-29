import json
import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter


def feature_selection(input_file, threshold, output_file=None):
    # ---------- 1. Preparation ------------#
    # Initialize Tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # Selected Features
    selected_features = []

    # Class Frequency
    classwise_word_freqs_arr = [0, 0, 0, 0, 0]

    # ---------- 2. Read & Extract Data ------------#
    # Open input file, read file line by line.
    # Make sure to run count_word.py before running this code.
    with open(input_file, 'r') as i_file:
        for line_number, line in enumerate(i_file, start=1):
            # Ignore the first row, since they are doc freqs of classes.
            if line_number == 1:
                continue

            # Never exceed the given threshold.
            if line_number > threshold:
                break

            # 2.1 Tokenize cur line.
            cur_line = line.strip()     # example: "apple 10 44 13 1918 809"
            cur_data_set = tokenizer.tokenize(cur_line)     # example: ['apple', '10', '44', '13', '1919','809']

            # 2.2 Separate words and frequencies.
            cur_word = cur_data_set[0]          # Current word
            cur_data_set = cur_data_set[1:]     # Slice the first word, leaving the frequencies.
            cur_data_set = [int(num) for num in cur_data_set]    # Convert string to int

            # Summarize
            # Sample row: ['apple',[10,44,13,1918,809]]
            selected_features.append((cur_word, cur_data_set))

        i_file.close()

    # -------------- 3. Post Process ----------------#
    # Summarize word frequencies into class-wise word frequencies.
    # Given rows in selected_features like: ['apple',[10,44,13,1918,809]]
    for instance in selected_features:
        cur_data_set = instance[1]      # Frequencies of a word.
        for index in range(0, len(cur_data_set)):   # Accumulate freq to class-wise array.
            classwise_word_freqs_arr[index] += cur_data_set[index]

    # --------------- 4. Write Data ----------------#
    # 4.1 Stringify Class Frequency
    classwise_word_freqs_str = ""
    for num in classwise_word_freqs_arr:
        classwise_word_freqs_str += str(num) + " "
    classwise_word_freqs_str += "\n"

    # 4.2 Write Files
    if output_file is None:
        return
    with open(output_file, 'w') as o_file:
        # 4.2.1 Write word frequencies of classes
        o_file.write(classwise_word_freqs_str)
        # 4.2.2 Write word and frequencies
        for instance in selected_features:
            # Sample row: ['apple',[10,44,13,1918,809]]
            cur_word = instance[0]      # Current word
            cur_data_set = instance[1]  # Its frequencies in all docs
            # Initialize write string
            cur_word_data_str = ""
            cur_word_data_str += cur_word + " "
            # Write each data into string
            for num in cur_data_set:
                cur_word_data_str += str(num) + " "
            cur_word_data_str += "\n"
            # Finally, write the processed string into file.
            o_file.write(cur_word_data_str)

        o_file.close()


feature_selection('./output/word_count.txt',  10000,'./output/word_dict.txt')
