import torch
import pandas as pd
from torch.utils.data import Dataset
import process_audio

import warnings
warnings.filterwarnings("ignore")

'''
Load Dataset (divided into train/validate/test sets)
* audio data : saved as segments in npy file
* labels : 50-d labels in csv file
'''

class SampleLevelMTTDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        '''
        Args : 
            csv_file : path to annotations_final_train.csv/annotations_final_val.csv/annotations_final_test.csv
            audio_dir : directory that contains folders 0 - f
        '''
        self.annotations_frame = pd.read_csv(csv_file, index_col=0) # df
        self.labels = self.annotations_frame.drop(['clip_id', 'mp3_path','split','shard', 'segment_id'], axis=1)
        self.audio_dir = audio_dir

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        mp3filename = self.annotations_frame.iloc[index, :]['mp3_path'].split(".")[0]
        segment_index = self.annotations_frame.iloc[index,:]['segment_id']

        segment = process_audio.get_segment_from_npy(self.audio_dir + mp3filename, segment_index)

        label = torch.FloatTensor(self.labels.iloc[index].tolist())

        entry = {'audio': segment, 'label': label}
        return entry
    
    def __len__(self):
        return self.annotations_frame.shape[0]
