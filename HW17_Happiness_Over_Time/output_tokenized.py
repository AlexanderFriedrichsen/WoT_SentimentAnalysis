# !pip install matplotlib
# !pip install numpy
# !pip install shifterator
# !pip install nltk
# !pip install pandas

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
#import shifterator
import nltk
import os
#import re
#import collections
#nltk.download('punkt')


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
tokenized = tokenize(lines)
print("done")


#new stuff
print("loading valence lexicon...")
hedonometer = pd.read_csv("Hedonometer.csv")

shared_words = list(set(list(hedonometer["Word"])).intersection(set(tokenized)))

backwards_word_dict = hedonometer["Word"].to_dict()
word_dict = {y:x for x,y in backwards_word_dict.items()}

happiness_score_dict = {x: hedonometer["Happiness Score"][word_dict[x]] for x in shared_words}


#create happiness time series
sample_text = tokenized

T_list = [10**3, 10**3.5, 10**4] #10**1, 10**1.5, 10**2, 10**2.5]
#T_list = [10**1]
d_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
all_window_happiness = []

x_values_list = []

d = d_list[2]
for T in T_list:
  T = round(T)
  start_window = 0
  stop_window = start_window + T

  #emergency_quit = 0
  window_happinesses = []
  x_values = []

  while start_window < len(sample_text): #or emergency_quit > 10*3:
    window = sample_text[start_window:stop_window]
    #print(window)

    happiness_scores = [happiness_score_dict[w] for w in window if w in happiness_score_dict.keys()]
    happiness_scores = [x for x in happiness_scores if abs(x - 5.0) > d]
    #print(happiness_scores)
    if len(happiness_scores) > 0:
      avg_happiness = sum(happiness_scores)/len(happiness_scores)
    else:
      # sometimes there are no words to score in the window, so for now just use the theoretical average
      avg_happiness = 5.0
    window_happinesses.append(avg_happiness)
    if stop_window < len(sample_text):
      x_values.append(stop_window)
    else:
      x_values.append(len(sample_text))

    start_window += 1
    stop_window = start_window + T
    mid_point = round((stop_window + start_window)/2.0)
    #emergency_quit += 1
  if stop_window < len(sample_text):
    window = sample_text[stop_window:]
    last_happiness = [happiness_score_dict[w] for w in window if w in happiness_score_dict.keys()]
    if len(happiness_scores) > 0:
      avg_happiness = sum(happiness_scores)/len(happiness_scores)
    else:
      # sometimes there are no words to score in the window, so for now just use the theoretical average
      avg_happiness = 5.0
    window_happinesses.append(avg_happiness)
    x_values.append(mid_point)

  all_window_happiness.append(window_happinesses)
  x_values_list.append(x_values)

for i in range(len(all_window_happiness)):
    pyplot.plot(x_values_list[i],all_window_happiness[i])
    ax = pyplot.gca()
    ax.set_ylim([0, 10])
    pyplot.show()
