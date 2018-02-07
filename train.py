import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorboardX import SummaryWriter


def train(model, train_loader, val_loader, criterion, learning_rate, num_epochs, args):
    # Define an optimizer (loss function is defined in main.py)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
    
    # Train the network
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        avg_auc1 = []
        avg_ap1 = []
        avg_auc2 = []
        avg_ap2 = []
        model.train() # training mode
        for i, data in enumerate(train_loader):
            audio = data['audio']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            audio = Variable(audio)
            label = Variable(label)

            if args.cuda:
                audio = audio.cuda(args.device_num)
                label = label.cuda(args.device_num)

            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))

                writer.add_scalar('train/loss', loss.data[0], (epoch+1)*len(train_loader)+ i+1)
                
                # retrieval 
                auc1, aprec1 = aroc_ap(label.data.cpu().numpy(), outputs.data.cpu().numpy())
                avg_auc1.append(np.mean(auc1))
                avg_ap1.append(np.mean(aprec1))
                # annotation
                auc2, aprec2 = aroc_ap2(label.data.cpu().numpy(), outputs.data.cpu().numpy())
                avg_auc2.append(np.mean(auc2))
                avg_ap2.append(np.mean(aprec2))

                print ('Retrieval : AROC = %.3f (%.3f)' % (np.mean(auc1), np.std(auc1) / np.sqrt(len(auc1))))
                print ('Retrieval : AP = %.3f (%.3f)' % (np.mean(aprec1), np.std(aprec1) / np.sqrt(len(aprec1))))
                print ('Annotation : AROC = %.3f (%.3f)' % (np.mean(auc2), np.std(auc2) / np.sqrt(len(auc2))))
                print ('Annotation : AP = %.3f (%.3f)' % (np.mean(aprec2), np.std(aprec2) / np.sqrt(len(aprec2))))
        
        print ("Retrieval : Average AROC = %.3f, AP = %.3f"%(np.mean(avg_auc1), np.mean(avg_ap1)))
        print ("Annotation :Average AROC = %.3f, AP = %.3f"%(np.mean(avg_auc2), np.mean(avg_ap2)))
        print ('Evaluating...')
        eval_loss = eval(model, val_loader, criterion, args)
            
        scheduler.step(eval_loss) # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print ('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-7:
            print ("Early stopping")
            break

    torch.save(model.state_dict(), model.__class__.__name__ + '.pth')
    writer.close()
            

# Validate the network on the val_loader (during training) or test_loader (for checking result)
# During training use this function for validation data.
def eval(model, val_loader, criterion, args):
    model.eval() # eval mode
    
    eval_loss = 0.0
    avg_auc1 = []
    avg_ap1 = []
    avg_auc2 = []
    avg_ap2 = []
    for i, data in enumerate(val_loader):
        audio = data['audio']
        label = data['label']
        audio = Variable(audio)
        label = Variable(label)

        if args.cuda:
            audio = audio.cuda(args.device_num)
            label = label.cuda(args.device_num)

        outputs = model(audio)
        loss = criterion(outputs, label)
        
        auc1, aprec1 = aroc_ap(label.data.cpu().numpy(), outputs.data.cpu().numpy())
        avg_auc1.append(np.mean(auc1))
        avg_ap1.append(np.mean(aprec1))
        auc2, aprec2 = aroc_ap2(label.data.cpu().numpy(), outputs.data.cpu().numpy())
        avg_auc2.append(np.mean(auc2))
        avg_ap2.append(np.mean(aprec2))

        eval_loss += loss.data[0]

    avg_loss =eval_loss/len(val_loader)
    print ("Retrieval : Average AROC = %.3f, AP = %.3f"%(np.mean(avg_auc1), np.mean(avg_ap1)))
    print ("Annotation : Average AROC = %.3f, AP = %.3f"%(np.mean(avg_auc2), np.mean(avg_ap2)))
    print ('Average loss: {:.4f} \n'. format(avg_loss))
    return avg_loss

''' Retrieval : Tag wise calculation '''
def aroc_ap(tags_true_binary, tags_predicted):
    n_tags = tags_true_binary.shape[1]
    auc = list()
    aprec = list()

    for i in range(n_tags):
        if np.sum(tags_true_binary[:, i]) != 0:
            auc.append(roc_auc_score(tags_true_binary[:, i], tags_predicted[:, i]))
            aprec.append(average_precision_score(tags_true_binary[:, i], tags_predicted[:, i]))

    return auc, aprec

''' Annotation : Item wise(row wise) calculation '''
def aroc_ap2(tags_true_binary, tags_predicted):
    n_songs = tags_true_binary.shape[0]
    auc = list()
    aprec = list()

    for i in range(n_songs):
        if np.sum(tags_true_binary[i]) != 0:
            auc.append(roc_auc_score(tags_true_binary[i], tags_predicted[i]))
            aprec.append(average_precision_score(tags_true_binary[i], tags_predicted[i]))

    return auc, aprec

if __name__ == '__main__':
    model = SampleCNN()
    model = model.load_state_dict(torch.load('SampleCNN.pth'))
