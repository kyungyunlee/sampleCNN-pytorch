import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
from process_audio import get_segment_from_npy 
from model import SampleCNN
import argparse


parser= argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, help='WHICH GPU')
args = parser.parse_args()
print (args)
args.cuda = torch.cuda.is_available()


''' Get the human readable ordered list of tags as saved in csv file '''
def get_taglist(csvfile):
    df = pd.read_csv(csvfile, delimiter=',')
    l = list(df)[1:]
    l.remove('clip_id')
    l.remove('mp3_path')
    print(len(l))
    return l

''' Load the trained model '''
def load_model(model_path, saved_state, model, args):
    if os.path.isfile(model_path + saved_state):
        model.load_state_dict(torch.load(model_path+ saved_state))
        if args.cuda:
            model.cuda()
        print ("Model loaded")
        return model
    else :
        print ("Model not found..")
        return

''' Predict tags for the given audio files '''
def predict_topN_tags(model, base_dir, audiofiles, sample_rate, n_samples, tagfile, args, N=5):
    taglist = open(tagfile, 'r').read().split('\n')
    if len(taglist) != 50:
        print ("more than 50 tags? %d"%len(taglist))
        for tag in taglist :
            if tag =='':
                taglist.remove(tag)
        print ("%d tags in total"%len(taglist))

    for song in audiofiles: 
        print ("Evaluating %s"%song)
        y, sr = librosa.load(base_dir + song, sr=sample_rate)
        print ("%d samples with %d sample rate"%(len(y), sr))

        # select middle 29.1secs(10 segments) and average them
        segments = []
        num_segments = 10
        if len(y) < (n_samples * 10) :
            num_segments = y//n_samples
        print ("Number of segments to calculate %d"%num_segments)

        '''
        # random index
        randidx= np.random.randint(len(y)//n_samples)
        segment = y[randidx*n_samples : (randidx+1)*n_samples]
        print ("segment length should be %d and is %d"%(n_samples, len(segment)))
        '''
        
        start_index = len(y)//2 - (n_samples*10)//2
        for i in range(num_segments):
            segments.append(y[start_index + (i*n_samples) : start_index + (i+1) * n_samples])    
        
        # predict value for each segment 
        calculated_val = []
        for segment in segments : 
            segment = torch.FloatTensor(segment)
            segment = segment.view(1, segment.shape[0])
            segment = Variable(segment)
            if args.cuda:
                segment = segment.cuda()

            model.eval()
            out = model(segment)
            sigmoid = nn.Sigmoid()
            out = sigmoid(out)
            out = out.cpu().data[0].numpy()
            calculated_val.append(out)
        
        # average 10 segment values
        calculated_val = np.array(calculated_val)
        print (calculated_val.shape)
        avg_val = np.sum(calculated_val, axis=0) /10
        
        # sort tags
        sorted_tags = np.argsort(avg_val)[::-1][:N]
        print (sorted_tags)
        predicted_tags = []
        for idx in sorted_tags:
            predicted_tags.append(taglist[idx])
        print (predicted_tags)


if __name__ =='__main__':
    model_path = './'
    base_dir = '/media/bach4/kylee/sampleCNN-data/'
    sample_rate= 22050
    n_samples = 59049
    saved_state = 'SampleCNN-singletag.pth'
    tagfile = '50_tags.txt'
    samplecnn_model = SampleCNN(0)
    model = load_model(model_path, saved_state, samplecnn_model, args)
    
    # Predict top 5 tags
    audio_files = ['zenzenzense.mp3', 'MuraMasa-WhatIfIGo.mp3']
    predict_topN_tags(model, base_dir,audio_files, sample_rate, n_samples, tagfile, args)
    


