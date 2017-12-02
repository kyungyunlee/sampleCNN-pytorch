import torch
from torch.autograd import Variable
import numpy as np
from model import SampleCNN
from sklearn.metrics import roc_auc_score, average_precision_score


# Validate the network on the val_loader (during training) or test_loader (for checking result)
# During training use this function for validation data.
def eval(model, some_loader, criterion, args):
    model.eval() # eval mode
    eval_loss = 0
    for i, data in enumerate(some_loader):
        audio = data['audio']
        label = data['label']
        audio = Variable(audio)
        label = Variable(label)

        if args.cuda:
            audio = audio.cuda()
            label = label.cuda()

        outputs = model(audio)
        loss = criterion(outputs, label)

        eval_loss += loss.data[0]

        if i+1 == len(some_loader):
            auc, aprec = aroc_ap(label.data.numpy(), outputs.data.numpy())
            print 'AROC = %.3f (%.3f)' % (np.mean(auc), np.std(auc) / np.sqrt(len(auc)))
            print 'AP = %.3f (%.3f)' % (np.mean(aprec), np.std(aprec) / np.sqrt(len(aprec)))

    eval_loss /= len(some_loader)
    print ('\nTotal Eval result : Average loss: {:.4f}, size of eval data : {} \n'. format(eval_loss, len(some_loader)))
    return eval_loss


def aroc_ap(tags_true_binary, tags_predicted):
    n_tags = tags_true_binary.shape[1]
    auc = list()
    aprec = list()

    for i in xrange(n_tags):
        if np.sum(tags_true_binary[:, i]) != 0:
            auc.append(roc_auc_score(tags_true_binary[:, i], tags_predicted[:, i]))
            aprec.append(average_precision_score(tags_true_binary[:, i], tags_predicted[:, i]))

    return auc, aprec

if __name__ == '__main__':
    model = SampleCNN()
    model = model.load_state_dict(torch.load('SampleCNN.pt'))
