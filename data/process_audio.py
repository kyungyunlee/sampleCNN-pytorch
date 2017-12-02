import os, errno
import numpy as np
import torch 
import torchaudio
from tempfile import NamedTemporaryFile

RAW_FILE_PATH = 'path_to_raw_audio_data/'
NPY_FILE_PATH = 'path_to_saving_npy_files/'

'''
Preprocessing MTT audio dataset
* 29.1 sec mp3 files
* annotation_final.csv 

Goal : Segment the audio files into 59049 samples -> save all segments/audio in npy file
'''


'''
Changes tempo and gain of the recording with sox and loads it.

I used resampling code from here!
https://github.com/SeanNaren/deepspeech.pytorch
'''
def _resample_audio_with_sox(path, sample_rate):
    with NamedTemporaryFile(suffix=".mp3") as augmented_file:
        augmented_filename = augmented_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename)
        os.system(sox_params)
        sig, sr = torchaudio.load(augmented_filename)
    return sig, sr


'''
Load audio and split into segments  1 segment = 59059 samples)
ex. sample_rate = 22050, n_samples = 59049
'''
def _audio_to_segments (filename, sample_rate, n_samples, center=False):
    try:
        sig, sr = torchaudio.load(filename)
    except:
        print "Could not load file {}".format(filename)
        raise
    if (sr != sample_rate) :
        sig, sr = _resample_audio_with_sox(filename, sample_rate)
    total_samples = len(sig)
    n_segments = total_samples // n_samples
    if center :
        remainder = total_samples % n_samples
        sig = sig[remainder//2 : -remainder//2]
        
    segments = [sig[i * n_samples : (i + 1) * n_samples].numpy() for i in range(n_segments)]
    return segments


'''
Now go through each mp3 file and save each segment into the same npy file
'''
def _save_audio_to_npy(rawfilepath, npyfilepath, sample_rate, n_samples, center=False):
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
                segments = _audio_to_segments(rawfilepath + path + '/' + audio, sample_rate, n_samples, center)
            except:
                print "Exception raised : just pass"
                continue

            # If the audio file length is less than 29.1 sec, do not create a npy file
            if len(segments) < 10 :
                print "Audio file {} length is less than 29.1 sec...pass".format(path + '/' + audio)
                continue

            fn = audio.split(".")[0]
            np.save(npyfilepath + path + '/' + fn + '.npy', segments)


'''
Public function for getting the correct segment within the mp3->npy file
'''
def get_segment_from_npy(npyfilename, segment_index):
    segments = np.load(npyfilename + '.npy')

    try:
        segment = torch.FloatTensor(segments[segment_index])
    except IndexError:
        print ("IndexError: Segment length {} is less than or equal to segment index {}".format(len(segments), segment_index))

    return segment


if __name__ == "__main__":
    # preprocessing code
    _save_audio_to_npy(RAW_FILE_PATH, NPY_FILE_PATH, 22050, 59049)
