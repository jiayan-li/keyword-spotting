import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  

import numpy as np

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def column_str_to_numpy(df, colname:str):
    # Given pd.DataFrame df, convert the column colname from string to numpy array.
    if isinstance(df.iloc[0][colname], str):
        df[colname]=df[colname].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

arr1 = np.zeros(shape = (5,3))
print(arr1)
print(flatten(arr1))


