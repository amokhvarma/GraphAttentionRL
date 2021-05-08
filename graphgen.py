import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.sparse import dense_to_sparse
import cv2
from model import GAT
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
    data = Data(x=x,edge_index=edge_index,complete=img)
    return data



if __name__ == '__main__':
    a = np.random.randn(60,60,3)
    t = make_graph(a,50)
    model = GAT()
    model.train()
    print(model(t).detach().numpy())