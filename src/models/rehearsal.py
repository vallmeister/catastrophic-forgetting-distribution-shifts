import random

import torch
import torch.nn as nn


class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self):
        super().__init__()

    def forward(self, ids_per_cls_train, budget, feats):
        return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self, ids_per_cls_train, budget, vecs):
        budget_dist_compute = 1000
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(
                ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute, len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0, vecs_1))

            dist_ = torch.cat(dist, dim=-1)  # include distance to all the other classes
            dist_ = torch.mean(dist_, dim=1)
            rank = dist_.sort(descending=True)[1].tolist()
            current_ids_selected = rank[:budget]
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected


class ExperienceReplay(torch.nn.Module):
    """
        ER-GNN baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.

        """

    def __init__(self, model, num_cls):
        super(ExperienceReplay, self).__init__()

        # setup network
        self.net = model
        self.sampler = CM_sampler()
        self.num_classes = num_cls

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.01, weight_decay=0.001)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = 1
        self.aux_data = None
        self.aux_features = None
        self.aux_labels = None
        self.aux_loss_w_ = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data, t):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = data.x
        labels = data.y
        train_mask = data.train_mask
        ids_per_cls = []
        for cls in range(self.num_classes):
            id_mask = train_mask & (labels == cls)
            ids = torch.where(id_mask)[0]
            ids_per_cls.append(ids.tolist())
        self.net.train()
        n_nodes = train_mask.sum().item()
        buffer_size = len(self.buffer_node_ids)
        beta = buffer_size / (buffer_size + n_nodes)

        self.net.zero_grad()
        output = self.net(data)
        output_labels = labels[train_mask]

        n_per_cls = [(output_labels == j).sum() for j in range(self.num_classes)]
        loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class

        loss_w_ = torch.tensor(loss_w_).to(dev)
        loss = self.ce(output[train_mask], labels[train_mask], weight=loss_w_)

        if t != self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls, self.budget, features)
            self.buffer_node_ids.extend(sampled_ids)

            if t > 0:
                buffer_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
                buffer_mask[self.buffer_node_ids] = True
                self.aux_data = data.clone().subgraph(buffer_mask)
                self.aux_features, self.aux_labels = self.aux_data.x, self.aux_data.y
                n_per_cls = [(self.aux_labels == j).sum() for j in range(self.num_classes)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                self.aux_loss_w_ = torch.tensor(loss_w_).to(dev)

        if t != 0:
            # calculate auxiliary loss based on replay if not the first task
            output = self.net(self.aux_data)
            loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)
            loss = beta * loss + (1 - beta) * loss_aux

        loss.backward()
        self.opt.step()
        return loss
