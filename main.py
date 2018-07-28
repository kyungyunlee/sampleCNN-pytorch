import torch
import model
from data_loader import SampleLevelMTTDataset
import argparse
import model
from solver import Solver

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', nargs='+', type=int, default=[])
args = parser.parse_args()

print ("gpu devices being used: ", args.gpus)

def main() :
    
    dataset = SampleLevelMTTDataset()
    samplecnn = model.SampleCNN()

    # start training
    print ("Start training!!")
    mysolver = Solver(samplecnn, dataset, args)
    mysolver.train()
    
    print ("Finished! Hopefully..")

    # test it
    print ("Start testing...")
    



if __name__ == '__main__':
    main()

