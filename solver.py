import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.optim import lr_scheduler
import utils
import config

cuda = torch.cuda.is_available()
kwargs = {'num_workers':1, 'pin_memory':True} if cuda else {}
print ("gpu available :", cuda)
device = torch.device("cuda" if cuda else "cpu")
num_gpu = torch.cuda.device_count()
torch.cuda.manual_seed(5)

class Solver(object):
    def __init__(self, model, dataset, args):
        self.samplecnn = model
        self.dataset = dataset
        self.args = args
        
        self.curr_epoch = 0

        self.model_savepath = './model'
        if not os.path.exists(self.model_savepath):
            os.makedirs(self.model_savepath)

        # define loss function 
        self.bce = nn.BCEWithLogitsLoss()

        self._initialize()
        self.set_mode('train')


    def _initialize(self):
        self.optimizer = torch.optim.SGD(self.samplecnn.parameters(), lr=config.LR, weight_decay=1e-6, momentum=0.9, nesterov=True)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=2, verbose=True)


        # initialize cuda
        if len(self.args.gpus) > 1:
            self.multigpu = True
        else :
            self.multigpu = False
        
        utils.handle_multigpu(self.multigpu, self.args.gpus, num_gpu)

       
        if self.multigpu :
            self.samplecnn = nn.DataParallel(self.samplecnn, device_ids=self.args.gpus)

        self.samplecnn.to(device)


    def set_mode(self, mode):
        print ("solver mode : ", mode)
        if mode == 'train':
            self.samplecnn.train()
            self.dataset.set_mode('train')

        elif mode == 'valid' :
            self.samplecnn.eval()
            self.dataset.set_mode('valid')

        elif mode == 'test':
            self.samplecnn.eval()
            self.dataset.set_mode('test')

        self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)


    
    def train(self) :
        # Train the network
        for epoch in range(config.NUM_EPOCHS):
            self.set_mode('train') 

            avg_auc1 = []
            avg_ap1 = []
            avg_auc2 = []
            avg_ap2 = []

            for i, data in enumerate(self.dataloader):
                audio = data['audio'].to(device)
                label = data['label'].to(device)

                outputs = self.samplecnn(audio)
                loss = self.bce(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 10 == 0:
                    print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, config.NUM_EPOCHS, i+1, len(self.dataloader), loss.item()))

                    # retrieval 
                    auc1, ap1 = utils.tagwise_aroc_ap(label.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    avg_auc1.append(np.mean(auc1))
                    avg_ap1.append(np.mean(ap1))
                    # annotation
                    auc2, ap2 = utils.itemwise_aroc_ap(label.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    avg_auc2.append(np.mean(auc2))
                    avg_ap2.append(np.mean(ap2))
                    
                    print ("Retrieval : AROC = %.3f, AP = %.3f / "%(np.mean(auc1), np.mean(ap1)), "Annotation : AROC = %.3f, AP = %.3f"%(np.mean(auc2), np.mean(ap2)))


            self.curr_epoch +=1

            print ("Retrieval : Average AROC = %.3f, AP = %.3f / "%(np.mean(avg_auc1), np.mean(avg_ap1)), "Annotation :Average AROC = %.3f, AP = %.3f"%(np.mean(avg_auc2), np.mean(avg_ap2)))
            print ('Evaluating...')
            eval_loss = self.eval()
                
            self.scheduler.step(eval_loss) # use the learning rate scheduler
            curr_lr = self.optimizer.param_groups[0]['lr']
            print ('Learning rate : {}'.format(curr_lr))
            if curr_lr < 1e-7:
                print ("Early stopping")
                break

        torch.save(self.samplecnn.module.state_dict(), self.model_savepath / self.samplecnn.module.__class__.__name__ + '_' + str(self.curr_epoch) + '.pth')
                

    # Validate the network on the val_loader (during training) or test_loader (for checking result)
    # During training use this function for validation data.
    def eval():
        self.set_mode('valid')
        
        eval_loss = 0.0
        avg_auc1 = []
        avg_ap1 = []
        avg_auc2 = []
        avg_ap2 = []
        for i, data in enumerate(self.dataloader):
            audio = data['audio'].to(device)
            label = data['label'].to(device)

            outputs = self.samplecnn(audio)
            loss = self.bce(outputs, label)
            
            auc1, aprec1 = utils.tagwise_aroc_ap(label.cpu().detach().numpy(), outputs.cpu().detach.numpy())
            avg_auc1.append(np.mean(auc1))
            avg_ap1.append(np.mean(aprec1))
            auc2, aprec2 = utils.itemwise_aroc_ap(label.cpu().detach.numpy(), outputs.cpu().detach.numpy())
            avg_auc2.append(np.mean(auc2))
            avg_ap2.append(np.mean(aprec2))

            eval_loss += loss.data[0]

        avg_loss =eval_loss/len(val_loader)
        print ("Retrieval : Average AROC = %.3f, AP = %.3f / "%(np.mean(avg_auc1), np.mean(avg_ap1)), "Annotation : Average AROC = %.3f, AP = %.3f"%(np.mean(avg_auc2), np.mean(avg_ap2)))
        print ('Average loss: {:.4f} \n'. format(avg_loss))
        return avg_loss


if __name__ == '__main__':
    model = SampleCNN()
    model = model.load_state_dict(torch.load('SampleCNN.pth'))
