def flatten(x):
    N = x.shape[0] # read in number of samples N
    return x.view(N, -1)  # "flatten" the other dimensions.



