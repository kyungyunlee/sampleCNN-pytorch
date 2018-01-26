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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the network
    writer = SummaryWriter()
    epoch_loss = 0.0
    for epoch in range(num_epochs):
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

            epoch_loss += loss.data[0]

            if (i+1) % 10 == 0:
                print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))

                writer.add_scalar('train/loss', loss.data[0], (epoch+1)*len(train_loader)+ i+1)
                
                auc, aprec = aroc_ap(label.data.cpu().numpy(), outputs.data.cpu().numpy())
                # auc2 = aroc_ap2(label.data.cpu().numpy(), outputs.data.cpu().numpy())
                print ('AROC = %.3f (%.3f)' % (np.mean(auc), np.std(auc) / np.sqrt(len(auc))))
                print ('AP = %.3f (%.3f)' % (np.mean(aprec), np.std(aprec) / np.sqrt(len(aprec))))
                # print ('AROC = %.3f (%.3f)' % (np.mean(auc2), np.std(auc2) / np.sqrt(len(auc2))))
                #print 'AP = %.3f (%.3f)' % (np.mean(aprec), np.std(aprec) / np.sqrt(len(aprec)))
                
        print ('Evaluating...')
        eval_loss = eval(model, val_loader, criterion, args)
            
        scheduler.step(eval_loss) # use the learning rate scheduler
        print ('Learning rate : {}'.format(optimizer.param_groups[0]['lr']))

    torch.save(model.state_dict(), model.__class__.__name__ + '.pth')
    writer.close()
            

# Validate the network on the val_loader (during training) or test_loader (for checking result)
# During training use this function for validation data.
def eval(model, val_loader, criterion, args):
    model.eval() # eval mode
    
    eval_loss = 0.0
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

        eval_loss += loss.data[0]

    avg_loss =eval_loss/len(val_loader)
    print ('Average loss: {:.4f} \n'. format(avg_loss))
    return avg_loss

''' Tag wise calculation '''
def aroc_ap(tags_true_binary, tags_predicted):
    n_tags = tags_true_binary.shape[1]
    auc = list()
    aprec = list()

    for i in range(n_tags):
        if np.sum(tags_true_binary[:, i]) != 0:
            auc.append(roc_auc_score(tags_true_binary[:, i], tags_predicted[:, i]))
            aprec.append(average_precision_score(tags_true_binary[:, i], tags_predicted[:, i]))

    return auc, aprec

''' Row wise (item wise) calculation '''
'''
def aroc_ap2(tags_true_binary, tags_predicted):
    n_songs = tags_true_binary.shape[0]
    auc = list()
    #aprec = list()

    for i in range(n_songs):
        if np.sum(tags_true_binary[i]) != 0:
            auc.append(roc_auc_score(tags_true_binary[i], tags_predicted[i]))
            #aprec.append(average_precision_score(tags_true_binary[:, i], tags_predicted[:, i]))

    #return auc, aprec
    return auc
'''

if __name__ == '__main__':
    model = SampleCNN()
    model = model.load_state_dict(torch.load('SampleCNN.pt'))
