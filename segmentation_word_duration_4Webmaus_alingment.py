# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:16:21 2022

@author: Paula Perez
"""

import sys
import textgrids
import os
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#%% StopWordsRemoval
def StopWordsRemoval(text,language='spanish'):
    #Now eliminate stopwords
    clean_sentence= [word for word in text.split() if word.lower() not in stopwords.words(language)]
    clean_sentence=' '.join(clean_sentence)

    return clean_sentence
def StopWordsKeep(text,language='spanish'):
    #Now eliminate stopwords
    clean_sentence= [word for word in text.split() if word.lower() in stopwords.words(language)]
    clean_sentence=' '.join(clean_sentence)

    return clean_sentence



#%%% Language
lg = 'english'
#%% FOR WORDS with stop words
path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/HC/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD','Std_WD', 'Skew_WD', 'Kurt_WD', 'Min_WD', 'Max_WD', 'WD_RT','Avg_WD_RT','Std_WD_RT', 'Skew_WD_RT', 'Kurt_WD_RT', 'Min_WD_RT', 'Max_WD_RT', 'nW' , 'nW_RT']

all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['ORT-MAU']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        print('"{}";{}'.format(label, syll.dur))
        if len(label)!=0:
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\words/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'HC_words_ft_EN.csv', columns=columns, index=None)

#AD
path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/AD/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD','Std_WD', 'Skew_WD', 'Kurt_WD', 'Min_WD', 'Max_WD', 'WD_RT','Avg_WD_RT','Std_WD_RT', 'Skew_WD_RT', 'Kurt_WD_RT', 'Min_WD_RT', 'Max_WD_RT', 'nW' , 'nW_RT']

all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['ORT-MAU']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        print('"{}";{}'.format(label, syll.dur))
        if len(label)!=0:
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\words/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'AD_words_ft_EN.csv', columns=columns, index=None)

#%% FOR WORDS without stop word
path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/HC/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD_nSt','Std_WD_nSt', 'Skew_WD_nSt', 'Kurt_WD_nSt', 'Min_WD_nSt', 'Max_WD_nSt', 'WD_RT_nSt','Avg_WD_RT_nSt','Std_WD_RT_nSt', 
         'Skew_WD_RT_nSt', 'Kurt_WD_RT_nSt', 'Min_WD_RT_nSt', 'Max_WD_RT_nSt', 'nW_nSt' , 'nW_RT_nSt']
#columns=['ID','Avg_WD','Std_WD_St', 'Skew_WD_St', 'Kurt_WD_St', 'Min_WD_St', 'Max_WD_St', 'WD_RT_St','Avg_WD_RT_St','Std_WD_RT_St', 'Skew_WD_RT_St', 'Kurt_WD_RT_St', 'Min_WD_RT_St', 'Max_WD_RT_St', 'nW_St' , 'nW_RT_St']

all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['ORT-MAU']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        print('"{}";{}'.format(label, syll.dur))
        label=StopWordsRemoval(label, language = lg)
        
        if len(label)!=0:
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\words/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'HC_words_nstw_ft_EN.csv', columns=columns, index=None)


#AD

path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/AD/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD_nSt','Std_WD_nSt', 'Skew_WD_nSt', 'Kurt_WD_nSt', 'Min_WD_nSt', 'Max_WD_nSt', 'WD_RT_nSt','Avg_WD_RT_nSt','Std_WD_RT_nSt', 
         'Skew_WD_RT_nSt', 'Kurt_WD_RT_nSt', 'Min_WD_RT_nSt', 'Max_WD_RT_nSt', 'nW_nSt' , 'nW_RT_nSt']
all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['ORT-MAU']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        print('"{}";{}'.format(label, syll.dur))
        label=StopWordsRemoval(label, language = lg)
        
        if len(label)!=0:
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\words/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'AD_words_nstw_ft_EN.csv', columns=columns, index=None)
#%% FOR WORDS only stop word

path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/HC/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD_St','Std_WD_St', 'Skew_WD_St', 'Kurt_WD_St', 'Min_WD_St', 'Max_WD_St', 'WD_RT_St','Avg_WD_RT_St','Std_WD_RT_St', 'Skew_WD_RT_St', 'Kurt_WD_RT_St', 'Min_WD_RT_St', 'Max_WD_RT_St', 'nW_St' , 'nW_RT_St']

all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['ORT-MAU']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        print('"{}";{}'.format(label, syll.dur))
        label=StopWordsKeep(label, language = lg)
        
        if len(label)!=0:
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\words/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'HC_stw_ft_EN.csv', columns=columns, index=None)     

#AD
path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/AD/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD_St','Std_WD_St', 'Skew_WD_St', 'Kurt_WD_St', 'Min_WD_St', 'Max_WD_St', 'WD_RT_St','Avg_WD_RT_St','Std_WD_RT_St', 'Skew_WD_RT_St', 'Kurt_WD_RT_St', 'Min_WD_RT_St', 'Max_WD_RT_St', 'nW_St' , 'nW_RT_St']

all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['ORT-MAU']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        print('"{}";{}'.format(label, syll.dur))
        label=StopWordsKeep(label, language = lg)
        
        if len(label)!=0:
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\words/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'AD_stw_ft_EN.csv', columns=columns, index=None)     
                
#%%
# pt=r'C:/Users/pama_/Documents/PhD/GITA/Adolfo/Features/Prosody/AD_02.TextGrid'

# grid = textgrids.TextGrid(pt)

# for syll in grid['KAN-MAU']:
#     # Convert Praat to Unicode in the label
#     label = syll.text.transcode()
#     # Print label and syllable duration, CSV-like
#     print('"{}";{}'.format(label, syll.dur))

#%% FOR SYLLABLES with stop words


path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/HC/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD_sy','Std_WD_sy', 'Skew_WD_sy', 'Kurt_WD_sy', 'Min_WD_sy', 'Max_WD_sy', 'WD_RT_sy','Avg_WD_RT_sy','Std_WD_RT_sy', 
         'Skew_WD_RT_sy', 'Kurt_WD_RT_sy', 'Min_WD_RT_sy', 'Max_WD_RT_sy', 'nW_sy' , 'nW_RT_sy']

all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid,
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        print(arg)
        print('"{}";{}'.format(label, syll.dur))
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['MAS']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        #print('"{}";{}'.format(label, syll.dur))
        if len(label)!=0 and label!='<p:>':
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\syllables/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'HC_syll_ft_EN.csv', columns=columns, index=None)




#AD
path=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\textgrids/AD/'
files=os.listdir(path)
features=[]
columns=['ID','Avg_WD_sy','Std_WD_sy', 'Skew_WD_sy', 'Kurt_WD_sy', 'Min_WD_sy', 'Max_WD_sy', 'WD_RT_sy','Avg_WD_RT_sy','Std_WD_RT_sy', 
         'Skew_WD_RT_sy', 'Kurt_WD_RT_sy', 'Min_WD_RT_sy', 'Max_WD_RT_sy', 'nW_sy' , 'nW_RT_sy']
all_dur=[]
all_w=[]
for arg in files:
    
    #Empty lists
    duration=[]
    xmax=[]
    fts=[]
    
    
    # Try to open the file as textgrid,
    try:
        grid = textgrids.TextGrid(path+arg)
    # Discard and try the next one
    except:
        print(arg)
        print('"{}";{}'.format(label, syll.dur))
        continue

    # Assume "syllables" is the name of the tier
    # containing syllable information
    for syll in grid['MAS']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration, CSV-like
        #print('"{}";{}'.format(label, syll.dur))
        
        if len(label)!=0 and label!='<p:>':
            duration.append(syll.dur)
            all_dur.append(syll.dur)
            all_w.append(label)
        xmax.append(syll.xmax)
    
        
    
    
    #Features
    
    #Duration words-> Avg, std, skew,kurt, min, max
    fts.append([np.mean(duration), np.std(duration), skew(duration), kurtosis(duration), np.min(duration), np.max(duration)])
    
    
    #Duration word ratios -> avg-duration/total duration,duration word/total duration (Avg, std, skew,kurt, min, max)
    fts.append([np.mean(duration)/xmax[-1], np.mean(np.array(duration)/xmax[-1]),np.std(np.array(duration)/xmax[-1]), skew(np.array(duration)/xmax[-1]), kurtosis(np.array(duration)/xmax[-1]), np.min(np.array(duration)/xmax[-1]), np.max(np.array(duration)/xmax[-1])])
    
    #Number words-> number, number/total duration
    fts.append([len(duration), len(duration)/xmax[-1]])

    features.append([np.hstack((arg[:-9], np.hstack(fts)))])

features=np.vstack(features)

pt=r'C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Code\Features\Speech\Words\syllables/'

df=pd.DataFrame(features, columns=columns, index=None)
df.to_csv(pt+'AD_syll_ft_EN.csv', columns=columns, index=None)