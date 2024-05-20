import torch


class Twp(torch.nn.Module):
    def __init__(self, model):
        super(Twp, self).__init__()

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
        return self.net(features)

    def observe(self, data, t):
        self.net.train()

        # if new task
        if t != self.current_task:
            self.net.zero_grad()
            self.fisher_loss[self.current_task] = []
            self.fisher_att[self.current_task] = []
            self.optpar[self.current_task] = []

            # computing gradient for the previous task
            output = self.net(self.mem_task)
            loss = self.ce(output, self.mem_task.y)
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
            self.mem_task = None

        if self.mem_task is None:
            self.mem_task = data.clone()

        self.net.zero_grad()
        output = self.net(data)
        loss = self.ce(output, data.y)

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
        return loss
