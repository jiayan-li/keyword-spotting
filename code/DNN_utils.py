import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  

import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'
dtype = torch.float32

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

#def flatten_label(y):
#    y = y.squeeze()
#    y = y.long()
#    return y 

def column_str_to_numpy(df, colname:str):
    # Given pd.DataFrame df, convert the column colname from string to numpy array.
    if isinstance(df.iloc[0][colname], str):
        df[colname]=df[colname].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            scores = model(x) 
            _, preds = scores.max(1) 
            true_class = y.argmax(dim=1) # True class is the one that has the highest probability in the label vector y.
            num_correct += (preds == true_class).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc 

def check_loss(loader, model):
    '''
    Check the loss in a given loader (e.g. validation loss)
    Args:
        loader: Loader object for tranining, validation or test sets.
        model: Our model (e.g. DNN)
    Returns:
        loss: Loss calculated with the dataset given by loader (e.g. validation dataset).
    '''
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            criterion = nn.KLDivLoss(reduction = "batchmean")
            loss = criterion(F.log_softmax(scores), y) 
            total_loss += loss    
    return total_loss
        

def check_accuracy_12_classes(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            y = flatten(y) # Flatten y to convert dimension from (NxCxH) to (N,-1)
            scores = model(x) 
            _, preds = scores.max(1) 
            true_class = y.argmax(dim=1) # True class is the one that has the highest probability in the label vector y.
            mask = (true_class<12)
            preds = preds[mask]
            true_class = true_class[mask] 
            num_correct += (preds == true_class).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc 
 

def check_accuracy_single_class(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            scores = model(x) 
            _, preds = scores.max(1) 
            #print(preds)
            #print(y)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc 


