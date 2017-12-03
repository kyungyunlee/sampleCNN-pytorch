import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from evaluate import eval, aroc_ap


def train(model, train_loader, val_loader, criterion, learning_rate, num_epochs, test_interval,
          save_interval, save_dir, args):
    # Define an optimizer (loss function is defined in main.py)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # Train the network
    model.train() # training mode
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            audio = data['audio']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            audio = Variable(audio)
            label = Variable(label)

            if args.cuda:
                audio = audio.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))

                auc, aprec = aroc_ap(label.data.numpy(), outputs.data.numpy())
                print 'AROC = %.3f (%.3f)' % (np.mean(auc), np.std(auc) / np.sqrt(len(auc)))
                print 'AP = %.3f (%.3f)' % (np.mean(aprec), np.std(aprec) / np.sqrt(len(aprec)))

            # cross validation
            if (i+1) % test_interval == 0:
                print ('Evaluating...')
                one_batch_val = next(iter(val_loader))
                eval_loss = eval(model, one_batch_val, criterion, args)
                scheduler.step(eval_loss) # use the learning rate scheduler

            if (i+1) % save_interval == 0:
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                save_prefix = os.path.join(save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, i+1)
                torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), 'SampleCNN.pt')
            

