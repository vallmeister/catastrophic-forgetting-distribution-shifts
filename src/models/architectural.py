import numpy as np
import torch
import torch.nn as nn


class FeatBrd1d(nn.Module):
    '''
    Feature Broadcasting Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, n_features)
    Output size is (n_batch, out_channels, n_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        adjacency (Tensor): feature adjacency matrix
    '''

    def __init__(self, in_channels, out_channels, adjacency=None):
        super(FeatBrd1d, self).__init__()
        self.register_buffer('adjacency', adjacency)
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, adj=None):
        if adj is not None:
            return (adj.unsqueeze(1) @ (self.conv(x).unsqueeze(-1))).view(x.size(0), -1, x.size(-1))
        else:
            return (self.adjacency @ (self.conv(x).unsqueeze(-1))).view(x.size(0), -1, x.size(-1))


class FeatTrans1d(nn.Module):
    '''
    Feature Transforming Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''

    def __init__(self, in_channels, in_features, out_channels, out_features):
        super(FeatTrans1d, self).__init__()
        self.out_channels, self.out_features = out_channels, out_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels * out_features, kernel_size=in_features, bias=False)

    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = [self.transform(neighbor[i], adj[i:i + 1]) for i in range(x.size(0))]
        return x, neighbor

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        mm(N,c,f,f @ N,c,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (c_out*f_out,f)
        '''
        return self.conv((adj @ x.unsqueeze(-1)).squeeze(-1)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        fadj = torch.stack([torch.einsum('ca,ncb->cab', x[i], y[i]) for i in range(x.size(0))])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign() * (x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class Net(nn.Module):
    def __init__(self, args, feat_len, num_class, k=2, hidden=2):
        super(Net, self).__init__()
        self.args = args
        self.feat1 = FeatBrd1d(in_channels=1, out_channels=hidden)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(hidden), nn.Softsign())
        self.feat2 = FeatBrd1d(in_channels=hidden, out_channels=hidden)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(hidden), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(feat_len * hidden, num_class))

        self.register_buffer('adj', torch.zeros(1, feat_len, feat_len))
        self.register_buffer('inputs', torch.Tensor(0, 1, feat_len))
        self.register_buffer('targets', torch.LongTensor(0))
        self.neighbor = []
        self.sample_viewed = 0
        self.memory_order = torch.LongTensor()
        self.memory_size = self.args.memory_size

        self.criterion = nn.CrossEntropyLoss()
        exec('self.optimizer = torch.optim.%s(self.parameters(), lr=%f)' % (args.optm, args.lr))

    def forward(self, x, neighbor):
        fadj = self.feature_adjacency(x, neighbor)
        x = self.acvt1(self.feat1(x, fadj))
        x = self.acvt2(self.feat2(x, fadj))
        return self.classifier(x)

    def observe(self, inputs, targets, neighbor, reply=True):
        self.train()
        for i in range(self.args.iteration):
            self.optimizer.zero_grad()
            outputs = self.forward(inputs, neighbor)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        self.sample(inputs, targets, neighbor)
        if reply:
            L = torch.randperm(self.inputs.size(0))
            minibatches = [L[n:n + self.args.batch_size] for n in range(0, len(L), self.args.batch_size)]
            for index in minibatches:
                self.optimizer.zero_grad()
                inputs, targets, neighbor = self.inputs[index], self.targets[index], [self.neighbor[i] for i in
                                                                                      index.tolist()]
                outputs = self.forward(inputs, neighbor)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def feature_adjacency(self, x, y):
        fadj = torch.stack([(x[i].unsqueeze(-1) @ y[i].unsqueeze(-2)).sum(dim=[0, 1]) for i in range(x.size(0))])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    @torch.no_grad()
    def sgnroot(self, x):
        return x.sign() * (x.abs().sqrt())

    @torch.no_grad()
    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x

    @torch.no_grad()
    def uniform_sample(self, inputs, targets, neighbor):
        self.inputs = torch.cat((self.inputs, inputs), dim=0)
        self.targets = torch.cat((self.targets, targets), dim=0)
        self.neighbor += neighbor

        if self.inputs.size(0) > self.args.memory_size:
            idx = torch.randperm(self.inputs.size(0))[:self.args.memory_size]
            self.inputs, self.targets = self.inputs[idx], self.targets[idx]
            self.neighbor = [self.neighbor[i] for i in idx.tolist()]

    @torch.no_grad()
    def sample(self, inputs, targets, neighbor):
        self.sample_viewed += inputs.size(0)
        self.memory_order += inputs.size(0)  # increase the order

        self.targets = torch.cat((self.targets, targets), dim=0)
        self.inputs = torch.cat((self.inputs, inputs), dim=0)
        self.memory_order = torch.cat((self.memory_order, torch.LongTensor(list(range(inputs.size()[0] - 1, -1, -1)))),
                                      dim=0)  # for debug
        self.neighbor += neighbor

        node_len = int(self.inputs.size(0))
        ext_memory = node_len - self.memory_size
        if ext_memory > 0:
            mask = torch.zeros(node_len, dtype=bool)  # mask inputs order targets and neighbor
            reserve = self.memory_size  # reserved memrory to be stored
            seg = np.append(np.arange(0, self.sample_viewed, self.sample_viewed / ext_memory), self.sample_viewed)
            for i in range(len(seg) - 2, -1, -1):
                left = self.memory_order.ge(np.ceil(seg[i])) * self.memory_order.lt(np.floor(seg[i + 1]))
                leftindex = left.nonzero()
                if leftindex.size()[0] > reserve / (i + 1):  # the quote is not enough, need to be reduced
                    leftindex = leftindex[
                        torch.randperm(leftindex.size()[0])[:int(reserve / (i + 1))]]  # reserve the quote
                    mask[leftindex] = True
                else:
                    mask[leftindex] = True  # the quote is enough
                reserve -= leftindex.size()[0]  # deducte the quote
            self.inputs = self.inputs[mask]
            self.targets = self.targets[mask]
            self.memory_order = self.memory_order[mask]
            self.neighbor = [self.neighbor[i] for i in mask.nonzero()]


class PlainNet(nn.Module):
    '''
    Net without memory
    '''

    def __init__(self, feat_len, num_class, hidden=[10, 10], dropout=[0, 0]):
        super(PlainNet, self).__init__()
        self.feat1 = FeatBrd1d(in_channels=1, out_channels=hidden[0])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(hidden[0]), nn.Softsign(), nn.Dropout(dropout[0]))
        self.feat2 = FeatBrd1d(in_channels=hidden[0], out_channels=hidden[1])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(hidden[1]), nn.Softsign(), nn.Dropout(dropout[1]))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(feat_len * hidden[1], num_class))

    def forward(self, x, neighbor):
        fadj = self.feature_adjacency(x, neighbor)
        x = self.acvt1(self.feat1(x, fadj))
        x = self.acvt2(self.feat2(x, fadj))
        return self.classifier(x)

    @torch.no_grad()
    def feature_adjacency(self, x, y):
        fadj = torch.stack([torch.einsum('ca,ncb->ab', x[i], y[i]) for i in range(x.size(0))])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    @torch.no_grad()
    def sgnroot(self, x):
        return x.sign() * (x.abs().sqrt())

    @torch.no_grad()
    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class LGL(nn.Module):
    def __init__(self, feat_len, num_class, hidden=[64, 32], dropout=[0, 0]):
        ## the Flag ismlp will encode without neighbor
        super(LGL, self).__init__()
        c = [1, 4, hidden[1]]
        f = [feat_len, int(hidden[0] / c[1]), 1]

        self.feat1 = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign(), nn.Dropout(p=dropout[0]))
        self.feat2 = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign(), nn.Dropout(p=dropout[1]))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[2] * f[2], num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)
