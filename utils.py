import torch
from torch.autograd import Variable

def idx2onehot(idx, n, idx2=None, alpha = 1):

    assert torch.max(idx).item() < n
    
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
       
    try:
        #print("only go in this one")
        ans = []
        for i in range(idx.data.numpy().size):
            arr = [0., 0., 0., 0., 0., 0., 0, 0., 0., 0.]
            arr[idx.data.numpy()[i][0]] = alpha
            arr[idx2.data.numpy()[i][0]] = 1-alpha
            ans.append(arr)
        onehot = torch.tensor(ans)
    except:
        #print("never go in this one")
        onehot = torch.zeros(idx.size(0), n)
        onehot.scatter_(1, idx, 1)

    #print(onehot)
    return onehot