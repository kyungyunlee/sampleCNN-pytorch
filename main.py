import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import model
import train
from data_loader import SampleLevelMTTDataset
from config import * 


parser = argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, help='WHICH GPU')
args = parser.parse_args()
print (args)
device = torch.device("cuda:" + str(args.device_num) if torch.cuda.is_available() else "cpu")


def main() :
    # load data
    train_annotation = BASE_DIR + 'train_50_tags_annotations_final.csv'
    val_annotation = BASE_DIR + 'valid_50_tags_annotations_final.csv'
    test_annotation = BASE_DIR + 'test_50_tags_annotations_final.csv'
    
    print ("Start loading data...")
    train_data = SampleLevelMTTDataset(train_annotation, AUDIO_DIR, LIST_OF_TAGS, NUM_TAGS)
    val_data = SampleLevelMTTDataset(val_annotation, AUDIO_DIR, LIST_OF_TAGS, NUM_TAGS)
    test_data = SampleLevelMTTDataset(test_annotation, AUDIO_DIR, LIST_OF_TAGS, NUM_TAGS)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print ("Finished loading data!")
   
    # load model
    print ("Load sampleCNN model")
    sampleCNN_model = model.SampleCNN(DROPOUT_RATE).to(device)

    # start training
    print ("Start training!!")
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss() # don't use sigmoid layer at the end when using this
    train.train(sampleCNN_model, train_loader, val_loader, criterion, LR, NUM_EPOCHS, device)
    
    print ("Finished! Hopefully..")

    # test it
    print ("Start testing...")
    train.eval(sampleCNN_model, test_loader, criterion, device)



if __name__ == '__main__':
    main()

