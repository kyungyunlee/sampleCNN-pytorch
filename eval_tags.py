''' Functions to evaluate the tag predictions '''
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import argparse
from audio_processor import get_segment_from_npy 
from model import SampleCNN
import config
import utils


cuda = torch.cuda.is_available()
print("gpu available :", cuda)
device = torch.device("cuda" if cuda else "cpu")
num_gpu = torch.cuda.device_count()
torch.cuda.manual_seed(5)


parser= argparse.ArgumentParser()
parser.add_argument('--gpus', nargs='+', type=int, default=[])
parser.add_argument('--mp3_file', type=str)
args = parser.parse_args()
print (args)

if len(args.gpus) > 1 :
    multigpu = True
else :
    multigpu = False

utils.handle_multigpu(multigpu, args.gpus, num_gpu)

def get_taglist(csvfile):
    ''' Get the human readable ordered list of tags as saved in csv file 
    '''
    df = pd.read_csv(csvfile, delimiter=',')
    l = list(df)[1:]
    l.remove('clip_id')
    l.remove('mp3_path')
    return l

def load_model(model, saved_state):
    ''' Load the trained model 
    Args : 
        model : initialized model with no state  
        saved_state : path to a specific model state  
    Return 
        model : trained model or None if not existing
    '''
    if os.path.isfile(saved_state):
        model.load_state_dict(torch.load(saved_state)) 
        print ("Model loaded")
        return model
    else :
        print ("Model not found..")
        return

def predict_topN_tags(model, base_dir, song, N=5):
    ''' Predict tags for the given audio files 
    Args : 
        model : path to trained model
        song : path to the song to predict (mp3 file)
        N : number of top N tag predictions to see
    Return : 
        predicted_tags : list of N predicted tags 
    '''
    taglist = open(config.LIST_OF_TAGS, 'r').read().split('\n')
    if len(taglist) != 50:
        print ("more than 50 tags? %d"%len(taglist), "fix..")
        for tag in taglist :
            if tag =='':
                taglist.remove(tag)
        print ("%d tags in total"%len(taglist))
    taglist = np.array(taglist)

    print ("Evaluating %s"%song)
    y, sr = librosa.load(song, sr=config.SR)
    print ("%d samples with %d sample rate"%(len(y), sr))

    # select middle 29.1secs(10 segments) and average them
    segments = []
    num_segments = 10
    if len(y) < (config.NUM_SAMPLES * 10) :
        num_segments = y//config.NUM_SAMPLES
    print ("Number of segments to calculate %d"%num_segments)

    start_index = len(y)//2 - (config.NUM_SAMPLES*10)//2
    for i in range(num_segments):
        segments.append(y[start_index + (i*config.NUM_SAMPLES) : start_index + (i+1) * config.NUM_SAMPLES])    
    
    # predict value for each segment 
    calculated_val = []
    for segment in segments : 
        segment = torch.FloatTensor(segment)
        segment = segment.view(1, segment.shape[0]).to(device)

        model.eval()
        out = model(segment)
        sigmoid = nn.Sigmoid()
        out = sigmoid(out)
        out = out.detach().cpu().numpy()
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
    saved_state = 'SampleCNN-singletag.pth'
    samplecnn_model = SampleCNN()
    model = load_model(samplecnn_model, saved_state)
    if multigpu : 
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    
    model.to(device)

    # Predict top 5 tags
    predict_topN_tags(model, config.BASE_DIR , args.mp3_file)
    


