import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.sparse import dense_to_sparse
import cv2
from model import GAT,CNN
import torch.nn.functional as F

def make_graph(state,target_dim = 50, type = "full" ):
    x = np.zeros((target_dim*target_dim,3))

    img = cv2.resize(state,(target_dim,target_dim))
    adj = np.ones((target_dim,target_dim))
    for i in range(0,target_dim):
        adj[i][i] = 0
    node = 0
    for i in range(0,target_dim):
        for j in range(0,target_dim):
            x[node] = img[i][j]/255
            node+=1

    adj_tensor = torch.tensor(adj,dtype=torch.long)
    x = torch.tensor(x,dtype=torch.float32)
    edge_index,_ = dense_to_sparse(adj_tensor)
    img = np.reshape(img,(3,target_dim,target_dim))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(x=x,edge_index=edge_index,complete=torch.tensor(img/255,dtype=torch.float32)).to(device)
    return data

def attention_calc(alpha_l,alpha_r,i):
    n = alpha_l.shape[0]
    n = int(n)
    mat = np.zeros((n,n))
    temp = np.random.randn(10,10)
    dummy_weights = []
    left = alpha_l[i]
    with torch.no_grad():
        for j in range(0,n):

             if(not j==i):
                 dummy_weights.append(F.leaky_relu(negative_slope=0.02,input = left+alpha_r[j]))
             else:
                 dummy_weights.append(0)
        s = sum(dummy_weights)
        dummy_weights = [(i/s).numpy()[0] for i in dummy_weights]
        mat = np.reshape(np.array(dummy_weights),(int(np.sqrt(n)),int(np.sqrt(n))))
        print(255*mat)
        cv2.imwrite("Attention.png",5e4*mat)
        cv2.imwrite("Temp.png", 1000*temp)
    return mat


if __name__ == '__main__':
    a = np.random.randn(60,60,3)
    t = make_graph(a,50)
    model = CNN()
    model.train()
    print(model(t).detach().numpy())