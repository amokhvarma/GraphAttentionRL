import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv,SAGEConv

class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        self.hid = 12

        self.input_dim = 50
        self.in_head = 8
        self.out_head = 1
        self.num_features = 3

        self.conv_output = 8
        self.attention = None

        self.conv_output = 8

        self.conv1 = GATConv(self.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, self.conv_output, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.flat = torch.nn.Flatten(0,-1)
        self.fc = torch.nn.Linear(in_features = self.conv_output*self.input_dim**2,out_features=4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.

        x = F.dropout(x, p=0.6, training=self.training)
        x,self.attention = self.conv1(x, edge_index,return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.flat(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.hidden_layers = 2
        self.input_dim = 50
        self.stride = 1
        self.conv1 = torch.nn.Conv2d(3,64,kernel_size = (3,3),stride = (self.stride,self.stride))
        self.pool1 = torch.nn.MaxPool2d((2,2),stride=2)
        self.conv2 = torch.nn.Conv2d(64,64,kernel_size = (3,3),stride = (self.stride,self.stride))
        self.pool2 = torch.nn.MaxPool2d((2,2),stride=2)
        self.flat = torch.nn.Flatten(0,-1)
        self.fc = torch.nn.Linear(in_features=7744,out_features= 4)

    def forward(self,data):
        img = data.complete
        img = img.unsqueeze(0)
        x = self.conv1(img)
        #print(x.shape)
        x = self.pool1(x)
        #x = F.relu(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.flat(x)
        #print(x.shape)
        x = self.fc(x)
        #print("Success")
        return F.log_softmax(x,dim=-1)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.hid = 8
        self.input_dim = 50
        self.num_features = 3
        self.conv_output = 8
        self.conv1 = GCNConv(self.num_features,self.hid, dropout=0.2)
        self.conv2 = GCNConv(self.hid , self.conv_output, dropout=0.2)
        self.flat = torch.nn.Flatten(0,-1)
        self.fc = torch.nn.Linear(in_features = self.conv_output*self.input_dim**2,out_features=4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.

        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.flat(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)



class SAGE(torch.nn.Module):
    def __init__(self):
        super(SAGE, self).__init__()
        self.hid = 8
        self.input_dim = 50
        self.num_features = 3
        self.conv_output = 4
        self.conv1 = SAGEConv(self.num_features,self.hid, dropout=0.6)
        self.conv2 = SAGEConv(self.hid , self.conv_output, dropout=0.6)
        self.flat = torch.nn.Flatten(0,-1)
        self.fc = torch.nn.Linear(in_features = self.conv_output*self.input_dim**2,out_features=4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.

        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.flat(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
