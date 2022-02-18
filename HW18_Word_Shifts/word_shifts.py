# We use the first book of the Wheel of Time vs the 8th, 
# which in my opinion should be considerably darker

# Lexicon from here: Dodds, Peter Sheridan, Kameron Decker Harris, 
# Isabel M. Kloumann, Catherine A. Bliss, and Christopher M. Danforth. 
# “Temporal patterns of happiness and information in a global social network: 
# Hedonometrics and Twitter.” PLoS ONE 6, no. 12 (2011).

#First we tokenize


from typing import Dict
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import time
import os
import shifterator as sh
plt.style.use('bmh')
from nltk.corpus import webtext
from nltk.probability import FreqDist
from collections import Counter
 

#take all the .txt files for WoT and compile them into one
def load_files(rootdir: str) -> str:
    lines = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print(file)
            with open(os.path.join(subdir, file), encoding="utf8") as f:
                lines.append(f.readlines())
    return lines

def load_file(path_to_file: str) -> str:
    return open(path_to_file, encoding="utf8").readlines()

def tokenize(text):
    #ft = fulltext
    ft=""
    for line in text:
        ft += str(line)

    without_line_breaks = ft.replace("\\n', '\\n', '", " ")
    ft_tokenized = nltk.word_tokenize(without_line_breaks)
    one_grams = [word.lower() for word in ft_tokenized]

    return one_grams


def create_frequency_dictionary(text: str) -> Dict:
    # nltk.download(text)
    # wt_words = webtext.words('testing.txt')
    #data_analysis = nltk.FreqDist(text)
    return Counter(text)
    # Let's take the specific words only if their frequency is greater than 3.
    #return (dict([(m, n) for m, n in data_analysis.items() if len(m) > 3]))


# frequency dictionary for book 1:
rootdir = "C:/Users/alexp/Documents/GitHub/WoT_SentimentAnalysis/subset_WoT_texts/The-Wheel-of-Time-01_-Robert-Jordan-The-Eye-of-the-World.txt"
print("loading files..")
text = load_file(rootdir)
print("tokenizing..")
tokenized = tokenize(text)
print("creating freq dict..")
freq_dict_1 = create_frequency_dictionary(tokenized)
# for key in sorted(freq_dict_1):
#     print("%s: %s" % (key, freq_dict_1[key]))
# data_analysis = nltk.FreqDist(freq_dict_1)
# data_analysis.plot(25, cumulative=False)

# frequency dictionary for book 8
book_8_path = "C:/Users/alexp/Documents/GitHub/WoT_SentimentAnalysis/all_texts_WoT/The-Wheel-of-Time-08_-Robert-Jordan-The-Path-of_Daggers.txt"
print("loading files..")
text = load_file(book_8_path)
print("tokenizing..")
tokenized = tokenize(text)
print("creating freq dict..")
freq_dict_2 = create_frequency_dictionary(tokenized)
# a couple notes before we start making the word shifts:
# stop_lens=[(4,6)] will exclude words with scores that are between 4 and 6 valence
print("creating wordshift")

#print(freq_dict_1)
sentiment_shift = sh.WeightedAvgShift(freq_dict_2,
                                      freq_dict_1,
                                      'labMT_English',
                                      reference_value = "average",
                                      stop_lens=[(5,5)])
sentiment_shift.get_shift_graph(detailed=True, system_names=['The-Path-of_Daggers','The-Eye-of-the-World'])

