import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from measures import Result


class NET(torch.nn.Module):
    def __init__(self,
                 model):
        super(NET, self).__init__()

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.01, weight_decay=0.001)

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}
        self.mem_mask = None
        self.mem_task = None

        # hyper-parameters
        self.lambda_l = 10000
        self.lambda_t = 100
        self.beta = 0.1

    def forward(self, features):
        output, elist = self.net(features)
        return output

    def observe(self, dataset, t):
        self.net.train()

        # if new task
        if t != self.current_task:
            self.net.zero_grad()
            self.fisher_loss[self.current_task] = []
            self.fisher_att[self.current_task] = []
            self.optpar[self.current_task] = []

            # computing gradient for the previous task
            output = self.net(self.mem_task)
            loss = self.ce((output[self.mem_mask]), self.mem_task.y[self.mem_mask])
            loss.backward(retain_graph=True)

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher_loss[self.current_task].append(pg)

            for p in self.net.parameters():
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[self.current_task].append(pg)

            self.current_task = t
            self.mem_mask = None
            self.mem_task = None

        if self.mem_mask is None:
            self.mem_mask = dataset.train_mask.data.clone()
        if self.mem_task is None:
            self.mem_task = dataset.clone()

        self.net.zero_grad()
        output = self.net(dataset)
        loss = self.ce(output[dataset.train_mask], dataset.y[dataset.train_mask])

        loss.backward(retain_graph=True)
        grad_norm = 0
        for p in self.net.parameters():
            pg = p.grad.data.clone()
            grad_norm += torch.norm(pg, p=1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss = loss + self.beta * grad_norm
        loss.backward()
        self.opt.step()


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Plain GCN
    plain_model = GCN(128, 16)
    print(device)
    optimizer = torch.optim.Adam(plain_model.parameters(), lr=0.01, weight_decay=0.001)

    data_list = torch.load('./data/csbm/feat_01.pt')
    plain_result = Result(data_list, plain_model, device)
    plain_result.learn()
    print(f'AP: {plain_result.get_average_accuracy():.2f}')
    print(f'AF: {plain_result.get_average_forgetting_measure():.2f}')


    # GCN with TWP regularization module
    twp_model = GCN(128, 16).to(device)
    twp = NET(twp_model)
    accuracy_matrix = torch.empty(10, 10)
    torch.set_printoptions(precision=3, sci_mode=False)
    for task_i, dataset_i in enumerate(data_list):
        dataset_i.to(device)
        for epoch in range(1, 201):
            twp.observe(dataset_i, task_i)
        for t in range(len(data_list)):
            dataset_t = data_list[t].to(device)
            test_mask = dataset_t.test_mask
            twp_model.eval()
            pred = twp_model(dataset_t).argmax(dim=1)
            correct = (pred[test_mask] == dataset_t.y[test_mask]).sum()
            acc = int(correct) / int(test_mask.sum())
            accuracy_matrix[task_i][t] = acc
    result = Result(data_list, twp_model, device)
    result.result_matrix = accuracy_matrix
    print(f'AP: {result.get_average_accuracy():.2f}')
    print(f'AF: {result.get_average_forgetting_measure():.2f}')
