# !pip install matplotlib
# !pip install numpy
# !pip install shifterator
# !pip install nltk
# !pip install pandas

import matplotlib as mpl
import numpy as np
import pandas as pd
#import shifterator
import nltk
import os
#import re
#import collections
nltk.download('punkt')


#take all the .txt files for WoT and compile them into one
def load_files():
    rootdir = "C:/Users/alexp/Documents/GitHub/WoT_SentimentAnalysis/txt2"

    lines = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print(file)
            with open(os.path.join(subdir, file), encoding="utf8") as f:
                lines.append(f.readlines())
    return lines

def tokenize(files):
#ft = fulltext
    ft=""
    for line in lines:
        
        ft += str(line)

    without_line_breaks = ft.replace("\\n', '\\n', '", " ")

    ft_tokenized = nltk.word_tokenize(without_line_breaks)

    one_grams = [word.lower() for word in ft_tokenized]

    return one_grams


print("loading files...")
lines = load_files()
print("converting to one gram vector...")
one_grams = tokenize(lines).to_csv("one_grams.csv")
print("done")


