''' Functions to process raw audio '''
import os, errno
import numpy as np
import torch 
import librosa
from config import *


def save_audio_to_npy(rawfilepath, npyfilepath, sample_rate=SR, num_samples=NUM_SAMPLES):
    ''' Save audio signal with sr=sample_rate to npy file 
    Args : 
        rawfilepath : path to the MTT audio files 
        npyfilepath : path to save the audio signal 
        sample_rate : sample rate
        num_samples : number of samples to segment 
    Return :
        None 
    '''

    mydir = [path for path in os.listdir(rawfilepath) if path >= '0' and path <= 'f']
    for path in mydir : 
        # create directory with names '0' to 'f' if it doesn't already exist
        try:
            os.mkdir(npyfilepath + path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        audios = [audio for audio in os.listdir(rawfilepath + path) if audio.split(".")[-1] == 'mp3']
        for audio in audios :
            try:
                y,sr = librosa.load(audio, sample_rate=sample_rate)
                if len(y)/num_samples < 10:
                    print ("There are less than 10 segments in this audio")
            except:
                print ("Cannot load audio {}".format(audio))
                continue

            fn = audio.split(".")[0]
            np.save(npyfilepath + path + '/' + fn + '.npy', y)


def get_segment_from_npy(npyfile, segment_idx, num_samples=NUM_SAMPLES):
    ''' Return random segment of length num_samples from the audio 
    Args : 
        npyfile : path to all the npy files each containing audio signals 
        segment_idx : index of the segment to retrieve; max(segment_idx) = total_samples//num_samples
        num_samples : number of samples in one segment 
    Return : 
        segment : audio signal of length num_samples 
    '''
    song = np.load(npyfile)
    # randidx = np.random.randint(10)
    try : 
        segment = song[segment_idx *num_samples : (segment_idx+1)*num_samples]
    except : 
        randidx = np.random.randint(10)
        get_segment_from_npy(npyfile, randidx, num_samples)
    return segment

if __name__ =='__main__':
    # read audio signal and save to npy format 
    save_audio_to_npy(MTT_DIR, AUDIO_DIR, SR, NUM_SAMPLES)
    

