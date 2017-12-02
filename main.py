import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import model
import train
import evaluate
from data.data_loader import SampleLevelMTTDataset


# HyperParameters
#SAMPLE_SIZE = 59049
#SAMPLE_RATE = 22050
BATCH_SIZE = 23
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.5
NUM_EPOCHS = 100
TEST_INTERVAL = 1000
SAVE_INTERVAL = 500

# Paths
# for accessing original data
RAW_AUDIO_DATA_PATH = 'path_to_raw_audio_data/'
ORIGINAL_ANNOTATION_PATH = 'path_to_annotation_file/'
# for saving newly processed data
NPY_AUDIO_DATA_PATH = 'path_to_saving_npy_files/'
NEW_ANNOTATION_PATH = 'path_to_saving_new_annotation_files/'
# for saving mdel
SAVE_MODEL_DIR = 'path_to_saving_models/'

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='DISABLE CUDA')
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


def main() :
    # load data
    train_annotation_path = NEW_ANNOTATION_PATH + 'annotations_final_train.csv'
    val_annotation_path = NEW_ANNOTATION_PATH + 'annotations_final_val.csv'
    test_annotation_path = NEW_ANNOTATION_PATH + 'annotations_final_test.csv'
    
    print ("Start loading data...")
    train_data = SampleLevelMTTDataset(train_annotation_path, NPY_AUDIO_DATA_PATH)
    val_data = SampleLevelMTTDataset(val_annotation_path, NPY_AUDIO_DATA_PATH)
    test_data = SampleLevelMTTDataset(test_annotation_path, NPY_AUDIO_DATA_PATH)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print ("Finished loading data!")
   
    # load model
    print ("Load samplCNN model")
    sampleCNN_model = model.SampleCNN(DROPOUT_RATE, BATCH_SIZE)

    if args.cuda:
        sampleCNN_model.cuda()


    # start training
    print ("Start training!!")
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss() # don't use sigmoid layer at the end when using this
    train.train(sampleCNN_model, train_loader, val_loader, criterion, LEARNING_RATE, NUM_EPOCHS, TEST_INTERVAL, SAVE_INTERVAL, SAVE_MODEL_DIR, args)
    
    print ("Finished! Hopefully..")

    # test it
    print ("Start testing...")
    evaluate.eval(sampleCNN_model, test_loader, criterion, args)


if __name__ == '__main__':
    main()

