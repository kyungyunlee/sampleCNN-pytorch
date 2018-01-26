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
            csvfile : train/val/test csvfiles
            audio_dir : directory that contains folders 0 - f
        '''
        self.annotations_frame = pd.read_csv(csv_file, delimiter='\t') # df
        self.labels = self.annotations_frame.drop(['clip_id', 'mp3_path'], axis=1)
        self.audio_dir = audio_dir

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        idx = index // 10
        segment_idx = index % 10
       
        mp3filename = self.annotations_frame.iloc[idx]['mp3_path'].split('.')[0]+'.npy'
        try :
            segment = process_audio.get_segment_from_npy(self.audio_dir + mp3filename, segment_idx)
        except :
            new_index = idx-1 if idx > 0 else index +1
            return self.__getitem__(new_index)
        
        label = torch.FloatTensor(self.labels.iloc[idx].tolist())
        entry = {'audio': segment, 'label': label}
        return entry
    
    def __len__(self):
        return self.annotations_frame.shape[0] * 10
