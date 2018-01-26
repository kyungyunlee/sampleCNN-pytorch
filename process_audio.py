import os, errno
import numpy as np
import torch 
import librosa


''' Convert mp3 file into npy file '''
def save_audio_to_npy(rawfilepath, npyfilepath, sample_rate, num_samples):
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
            np.save(npyfilepath + path + '/' + fn + '.npy', segments)


''' Get random segment of length 59049 samples from the audio '''
def get_segment_from_npy(npyfile, segment_idx, num_samples=59049):
    song = np.load(npyfile)
    # randidx = np.random.randint(10)
    try : 
        segment = song[segment_idx *num_samples : (segment_idx+1)*num_samples]
    except : 
        randidx = np.random.randint(10)
        get_segment_from_npy(npyfile, randidx, num_samples)
    return segment


