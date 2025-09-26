# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:55:29 2021

@author: Tomas & Paula
"""

import numpy as np
import scipy as sp
from scipy import signal
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read,write #Leer y guardar audios
#from scipy.signal import gaussian
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import sys
#sys.path.append('./Phoneme_features')
#import Phoneme_posteriors.get_phone_features as ph


def eVAD(sig,fs,win=0.025,step=0.01):
    """
    Energy-based Voice Activity Detection
    """
    #Normalize signal
    sig = sig-np.mean(sig)
    sig /=np.max(np.abs(sig))
    
    lsig = len(sig)
    #Add silence to the beginning and end in case the user is an idiot or myself
    
    #Set min threshold base on the energy of the signal
    e = []
    frames = extract_windows(sig,int(win*fs),int(step*fs))

    for seg in frames:
        e.append(10*np.log10(np.sum(np.absolute(seg)**2)/len(seg)))
    e = np.asarray(e)
#    print('e[dB]',np.min(e))
    idx_min = np.where(e==np.min(e))
#    idx_min = np.where(e<=-50)[0]#Threshold in -50dB
    thr = np.min(frames[idx_min])
#    print(np.min(e))

    ext_sil = int(fs)
    esil = int((ext_sil/2)/fs/step)
    new_sig = np.random.randn(lsig+ext_sil)*thr
    new_sig[int(ext_sil/2):lsig+int(ext_sil/2)] = sig
    sig = new_sig

    e = []#energy in dB
    frames = extract_windows(sig,int(win*fs),int(step*fs))
    frames*=np.hanning(int(win*fs))
    for seg in frames:
        e.append(10*np.log10(np.sum(np.absolute(seg)**2)/len(seg)))

    e = np.asarray(e)
    e = e-np.mean(e)
    #Smooth energy contour to remove small energy variations
    gauslen = int(fs*0.01)
    window = signal.gaussian(gauslen, std=int(gauslen*0.05))
    #Convolve signal with Gaussian window for smmothing
    smooth_env = e.copy()
    smooth_env = np.convolve(e, window)
    smooth_env = smooth_env/np.max(smooth_env)
    ini = int(gauslen/2)
    fin = len(smooth_env)-ini
    e = smooth_env[ini:fin]
    e = e/np.max(np.abs(e))
    e = e[esil:int(lsig/fs/step)+esil]

    thr = np.median(e[e < 0])

    cont_sil = np.zeros(lsig)
    cont_vad = np.zeros(lsig)
    itime = 0
    etime = int(win*fs)
    for i in range(len(e)):
        if e[i] <= thr:
            cont_sil[itime:etime] = 1
        else:
            cont_vad[itime:etime] = 1

        itime = i*int(step*fs)
        etime = itime+int(win*fs)

    if np.sum(cont_sil) != 0:
        #Pauses
        dur_sil,seg_sil,time_sil = get_segments(sig,fs,cont_sil,True)
        #Voice
        dur_vad,seg_vad,time_vad = get_segments(sig,fs,cont_vad)
    else:
        dur_sil = [0]
        seg_sil = [0]
        dur_vad = [0]
        seg_vad= [0]

    X_vad = {'Pause_labels':cont_sil,
             'Pause_duration':dur_sil,
             'Pause_segments':seg_sil,
             'Pause_times':time_sil,
             'Speech_labels':cont_vad,
             'Speech_duration':dur_vad,
             'Speech_segments':seg_vad,
             'Speech_times':time_vad}
    return X_vad


def get_segments(sig,fs,segments,lpause=False):
        segments[0] = 0
        segments[-1:] = 0
        yp = segments.copy()
        ydf = np.diff(yp)
        lim_end = np.where(ydf==-1)[0]+1
        lim_ini = np.where(ydf==1)[0]+1
        #Silence segments
        seg_dur = []#Segment durations
        seg_list = []#Segment list
        seg_time = []#Time stamps
        for idx in range(len(lim_ini)):
            #------------------------------------
            tini = lim_ini[idx]/fs
            tend = lim_end[idx]/fs
            seg_dur.append(np.abs(tend-tini))
            seg_list.append(sig[lim_ini[idx]:lim_end[idx]])
            seg_time.append([lim_ini[idx],lim_end[idx]])
            
        seg_dur = np.asarray(seg_dur)
#        if lpause==True:
#            seg_dur = seg_dur[1:-1:]#Eliminate the last two silence segment introduced at the start and end
        seg_time = np.vstack(seg_time)
        return seg_dur,seg_list,seg_time/fs
    
def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)


    return np.vstack(windows)

def duration_feats(x,sig,fs):   
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    

    Returns
    -------
    #Pause/len(sig)
    #speech/len(sig)
    #speech/#pauses
    #Functionals speech
    #Functionals Pauses

    """
    
    pause_len=len(x['Pause_duration'])/(len(sig)/fs)
    speech_len=len(x['Speech_duration'])/(len(sig)/fs)
    sp_ps=len(x['Speech_duration'])/len(x['Pause_duration'])
    speech=np.hstack((np.mean(x['Speech_duration']), np.std(x['Speech_duration']),sp.stats.skew(x['Speech_duration'])
                      ,sp.stats.kurtosis(x['Speech_duration']),np.min(x['Speech_duration']),np.max(x['Speech_duration'])))
    pauses=np.hstack((np.mean(x['Pause_duration']), np.std(x['Pause_duration']),sp.stats.skew(x['Pause_duration']),
                      sp.stats.kurtosis(x['Pause_duration']), np.min(x['Pause_duration']),np.max(x['Pause_duration'])))

    return np.hstack((pause_len,speech_len,sp_ps,speech,pauses))
#%%

'''
def extract_window(file):
    fs, audio_input = read(file)
    y, sr = librosa.load(file, sr=fs)
    scaler = preprocessing.StandardScaler()
    standard_X = np.hstack(scaler.fit_transform(np.vstack(y)))
    X, times = ph.get_features(file, 'ES')
    vowels = times['vowel']  # Phoneme segments
    windows = [standard_X[int(fs*v[0]):int(fs*v[1])] for v in vowels]
    return windows
'''

def main():
    '''
    path='.\\Phoneme_posteriors\\audio\\'
    #Audio path
    files= np.hstack(sorted([".".join(f.split(".")[:-1]) for f in os.listdir(path) ]))
    print(files)
    features=[]
    pbar=tqdm(files)

    new_files=[]
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        fs,sig = read(path+file+'.wav')
        y, sr = librosa.load(path+file+'.wav', sr=fs)
        sig = librosa.resample(y, sr, 16000)
        fs=16000
        sig=sig-np.mean(sig)
        sig /=np.max(np.abs(sig))
        x=eVAD(sig,fs)
        feats=duration_feats(x,sig,fs)
        features.append(np.hstack((file,feats)))

    print(features)

    data=pd.DataFrame(features, index=None, columns=('File Name', 'Pause_len_ratio','Speech_len_ratio','Pause_Speech_ratio',
                                         'Speech_mean','Speech_std', 'Speech_skew', 'Speech_kurt', 'Speech_min',
                                         'Speech_max',
                                         'Pause_mean','Pause_std', 'Pause_skew', 'Pause_kurt', 'Pause_min', 'Pause_max'))

    data.to_csv('Duration_feats_AD_seg.csv', index=None, columns=('File Name', 'Pause_len_ratio','Speech_len_ratio','Pause_Speech_ratio',
                                         'Speech_mean','Speech_std', 'Speech_skew', 'Speech_kurt', 'Speech_min',
                                         'Speech_max',
                                         'Pause_mean','Pause_std', 'Pause_skew', 'Pause_kurt', 'Pause_min', 'Pause_max'))

    #df=pd.DataFrame(time_stamps, columns=(['ini', 'end']), index=None)
    #df.to_csv(,index=False)

    '''

    path=r"C:\Users\pama_\OneDrive\Documents\PostDoc\Papers\2024\Cross_linguistic_Argen-Col\Data\Pitt\audio_preprocessed\Denoised/AD/"
    files = np.hstack(sorted([".".join(f.split(".")[:-1]) for f in os.listdir(path)]))
    features = []
    # pbar.set_description("Processing %s" % file)
    f = []
    pbar = tqdm(files)
    for file in pbar:
        if file != 'desktop':
            fs, sig = read(path + file + '.wav')
            y, sr = librosa.load(path + file + '.wav', sr=fs)
            sig = librosa.resample(y, orig_sr=sr, target_sr=16000)
            fs = 16000
            sig = sig - np.mean(sig)
            sig /= np.max(np.abs(sig))
            #print(sig)
            x = eVAD(sig, fs)
            feats = duration_feats(x, sig, fs)
            f.append(feats)

            #f = np.mean(f)
            #print(f)
            features.append(np.hstack((file, feats)))

    #print(features)

    data = pd.DataFrame(features, index=None,
                        columns=('File', 'lenPause_rt', 'lenSpeech_rt', 'Pause_Speech_rt',
                                 'meanSpeech', 'stdSpeech', 'skewSpeech', 'kurtSpeech', 'minSpeech',
                                 'maxSpeech',
                                 'meanPause', 'stdPause', 'skewPause', 'kurtPause', 'minPause', 'maxPause'))

    data.to_csv('AD_Duration_ft_EN.csv', index=None,
                columns=('File', 'lenPause_rt', 'lenSpeech_rt', 'Pause_Speech_rt',
                         'meanSpeech', 'stdSpeech', 'skewSpeech', 'kurtSpeech', 'minSpeech',
                         'maxSpeech',
                         'meanPause', 'stdPause', 'skewPause', 'kurtPause', 'minPause', 'maxPause'))

    # df=pd.DataFrame(time_stamps, columns=(['ini', 'end']), index=None)
    # df.to_csv(,index=False)

if __name__ == "__main__": main()
'''
plt.figure()
    plt.plot(sig)
    #plt.plot(x['Pause_labels'])
    plt.twiny()
    plt.plot(x['Speech_labels'], color='g')
    plt.show()
    i=1
    plt.figure()
for seg in segs:
    m=2
    n=4
    plt.subplot(m,n,i)
    plt.plot(seg)
    i+=1
plt.show()
'''