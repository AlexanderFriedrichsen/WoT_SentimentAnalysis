import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import time
import os
plt.style.use('bmh')

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


BLOCKED_WORDS_DELTA = np.arange(0,4,0.5)
WINDOW_SIZES = [round(x) for x in np.power(10, [1, 1.5, 2, 2.5, 3, 3.5, 4])]
HAPPINESS_SCORES = pd.read_csv("hedonometer.csv").filter(items=['Word','Happiness Score']).set_index('Word')

# raw_text = open("PKD texts/combined.txt").read().lower()
# one_grams = nltk.word_tokenize(raw_text)
one_grams = tokenized


happiness_list = []
for word in one_grams:
    if word in HAPPINESS_SCORES.index:
        happiness_list.append(HAPPINESS_SCORES.loc[word]['Happiness Score'])
overall_happiness = sum(happiness_list)/len(happiness_list)

for delta in BLOCKED_WORDS_DELTA:
    delta_start = time.perf_counter()
    # filtered lens
    filtered_scores = HAPPINESS_SCORES[abs(HAPPINESS_SCORES['Happiness Score']-overall_happiness)>delta]
    # store all data for this filter for plotting
    all_plots = []

    for window_size in WINDOW_SIZES:
        window_start = time.perf_counter()
        # keep track of the rolling average for plotting for this window size
        avg_scores = []
        initial_words = one_grams[:window_size]
        # calculate the average score of the initial window
        happiness_window = []
        for word in initial_words:
            if word in filtered_scores.index:
                happiness_window.append(filtered_scores.loc[word]['Happiness Score'])
        if (len(happiness_window) != 0):
            avg_happiness = sum(happiness_window)/len(happiness_window)
            avg_scores.append([0+(window_size-1)/2, avg_happiness])
        total_happiness = sum(happiness_window)
        num_scored_words_in_window = len(happiness_window)
        # add initial average score to the list for plotting
        for i in range(len(one_grams)-window_size):
            if (i % 5000 == 0):
                print(f'{i/len(one_grams)*100:0.1f}%', end='\r')
            old_word = one_grams[i]
            new_word = one_grams[i+window_size]
            # check if word leaving window is in filter
            if old_word in filtered_scores.index:
                total_happiness -= filtered_scores.loc[old_word]['Happiness Score']
                num_scored_words_in_window -= 1
            # check if word entering window is in filter
            if new_word in filtered_scores.index:
                total_happiness += filtered_scores.loc[new_word]['Happiness Score']
                num_scored_words_in_window += 1
            # add the new average score to the dictionary for plotting
            if (num_scored_words_in_window != 0):
                avg_happiness = total_happiness/num_scored_words_in_window
                avg_scores.append([i+(window_size-1)/2, avg_happiness])
        print()
        all_plots.append(np.transpose(avg_scores))
        window_end = time.perf_counter()
        print(f'Window width = {window_size} done in {window_end-window_start:0.1f} seconds.')

    fig = plt.figure()
    ax = plt.subplot()
    for plot in all_plots:
        ax.plot(plot[0], plot[1])
    ax.legend(["T = " + str(T) for T in WINDOW_SIZES], title="Window Size",bbox_to_anchor=(1, 0.5), loc='center left')
    ax.set_xlabel("Window Center")
    ax.set_ylabel('Average Happiness')
    ax.set_title('Delta = ' + str(delta))
    plt.show()
    delta_end = time.perf_counter()
    print(f'Delta = {delta} done in {delta_end-delta_start:0.1f} seconds.')