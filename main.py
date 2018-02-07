import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import model
import train
from data_loader import SampleLevelMTTDataset

# HyperParameters
#SAMPLE_SIZE = 59049
#SAMPLE_RATE = 22050
batch_size = 32
learning_rate = 0.008
dropout_rate = 0.5
num_epochs = 100
num_tags = 50

# Paths
# for accessing original data
data_dir = '/media/bach4/dataset/'
audio_dir = data_dir + 'pubMagnatagatune_mp3s_to_npy/' 
my_dir = '/media/bach4/kylee/sampleCNN-data/' # annotation files saved here

parser = argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, help='WHICH GPU')
args = parser.parse_args()
print (args)
args.cuda = torch.cuda.is_available()


def main() :
    # load data
    train_annotation = my_dir + 'train_50_tags_annotations_final.csv'
    val_annotation = my_dir + 'valid_50_tags_annotations_final.csv'
    test_annotation = my_dir + 'test_50_tags_annotations_final.csv'
    
    tagfile = '50_tags.txt'
    print ("Start loading data...")
    train_data = SampleLevelMTTDataset(train_annotation, audio_dir, tagfile, num_tags)
    val_data = SampleLevelMTTDataset(val_annotation, audio_dir, tagfile, num_tags)
    test_data = SampleLevelMTTDataset(test_annotation, audio_dir, tagfile, num_tags)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    print ("Finished loading data!")
   
    # load model
    print ("Load sampleCNN model")
    sampleCNN_model = model.SampleCNN(dropout_rate)

    if args.cuda:
        sampleCNN_model.cuda(args.device_num)

    # start training
    print ("Start training!!")
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss() # don't use sigmoid layer at the end when using this
    train.train(sampleCNN_model, train_loader, val_loader, criterion, learning_rate, num_epochs,args)
    
    print ("Finished! Hopefully..")

    # test it
    print ("Start testing...")
    train.eval(sampleCNN_model, test_loader, criterion, args)



if __name__ == '__main__':
    main()

