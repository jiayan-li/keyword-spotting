import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  

import numpy as np
import matplotlib.pyplot as plt


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

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
            y = flatten(y) # Flatten y to convert dimension from (NxCxH) to (N,-1)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype) 
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
            y = flatten(y)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            criterion = nn.BCEWithLogitsLoss() 
            total_loss = criterion(scores, y)
    
    return total_loss
        
def train(model, optimizer, scheduler, epochs=1, print_every=50):

    """
    Train the model using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - scheduler: Learning rate scheduler

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    val_loss_lst = []
    train_loss_lst = []
    accuracy_val_max = 0
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            y = flatten(y) # Flatten y to convert the dimension from (Nx1) to (N,)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)

            # Compare the output vector with the label vector using BCEwithLogitsLoss.
            criterion = nn.BCEWithLogitsLoss() 
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                print()
                accuracy_val = check_accuracy(loader_val, model)
                val_loss = check_loss(loader_val, model) 
                if accuracy_val > accuracy_val_max:
                    accuracy_val_max = accuracy_val
                    model_params = model.state_dict()
                train_loss_lst.append(loss.item())
                val_loss_lst.append(val_loss.item()) 
        
        # Update the learning rate at every epoch.
        scheduler.step()

    # Plot the accuracy values
    plt.plot(val_loss_lst, label='Validation Loss')
    plt.plot(train_loss_lst, label='Training Loss')

    # Add labels and title to the plot
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Show the plot
    plt.show()
    return train_loss_lst, val_loss_lst
    
    

