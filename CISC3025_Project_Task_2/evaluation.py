import os
import json
import re
import nltk
import math
import pandas as pd
import numpy as np
from collections import Counter

import __funcs__


def evaluation():

    # ------------ 1. Extract list of predict-actual data -------------#
    # 1.1 Initialize tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # 1.2 Get all the predict-actual data
    _, id_predict_actual_list = __funcs__.extract_data_from_txt(
        "./temp_output/classification_compare.txt",
        tokenizer,
        convert_to_int=False)

    # ------------ 2. Organize Data -------------#
    # 2.1 Initialize Precision Recall Matrix
    prec_rec_matrix = pd.DataFrame(np.zeros((5, 5), dtype=int))

    # 2.2 Record Classification Results
    for id_predict_actual in id_predict_actual_list:
        predict_actual = id_predict_actual[1]
        prediction, actual = predict_actual[0], predict_actual[1]
        prec_rec_matrix.loc[
            __funcs__.class_map[prediction],
            __funcs__.class_map[actual]
        ] += 1

    print(prec_rec_matrix)

    # ----------- 3. Precision and Recall -------------- #
    precisions = [0, 0, 0, 0, 0]
    recalls = [0, 0, 0, 0, 0]
    for index, row in prec_rec_matrix.iterrows():
        total_predict_num = sum(row)
        cur_precision = row[index] / total_predict_num
        precisions[index] = cur_precision

    for index, col in prec_rec_matrix.items():
        total_actual_num = sum(col)
        cur_recall = col[index] / total_actual_num
        recalls[index] = cur_recall

    avg_prec = sum(precisions) / len(precisions)
    avg_rec = sum(recalls) / len(recalls)

    print("Precisions: ", end=" ")
    print(precisions)
    print("Recalls: ", end=" ")
    print(recalls)

    print("Avg Prec: ", end=" ")
    print(avg_prec)
    print("Avg Rec: ", end=" ")
    print(avg_rec)

    # ---------- 4. F-Score: Macro-Average -------------- #
    f_scores = [0, 0, 0, 0, 0]

    for index in range(len(precisions)):
        cur_prec = precisions[index]
        cur_rec = recalls[index]
        f_scores[index] = 2 * cur_prec * cur_rec / (cur_prec + cur_rec)

    f_score_macro = sum(f_scores) / len(f_scores)

    print("\nF-Scores: ", end=" ")
    print(f_scores)

    print("Macro-Average F-Score: " + str(f_score_macro))

    # ---------- 5. F-Score: Micro-Average -------------- #
    f_score_micro = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)

    print("Micro-Average F-Score: " + str(f_score_micro))

    # ----------- 6. Write Result ------------ #
    with open('./output/f_scores.txt', 'w') as o_file:
        o_file.write("Precisions: ")
        for prec in precisions:
            o_file.write(str(prec) + " ")
        o_file.write("\n")

        o_file.write("Recalls: ")
        for rec in recalls:
            o_file.write(str(rec) + " ")
        o_file.write("\n")

        o_file.write("Avg Prec: " + str(avg_prec) + "\n")
        o_file.write("Avg Rec: " + str(avg_rec) + "\n")

        o_file.write("F-Scores: ")
        for f_score in f_scores:
            o_file.write(str(f_score) + " ")
        o_file.write("\n")

        o_file.write(
            "Macro-Average F-Score: " + str(f_score_macro) + "\n"
        )

        o_file.write(
            "Micro-Average F-Score: " + str(f_score_micro) + "\n"
        )

evaluation()
