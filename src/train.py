from pathlib import Path

import torch
from sklearn import metrics

from measures import get_average_forgetting_measure, get_average_accuracy
from models import GCN, Twp, ExperienceReplay
from util import EarlyStopping


def train(model, data, task, f1=False):
    es = EarlyStopping(Path('./gcn_backup.pt'), model)
    val_data = data.clone().subgraph(data.val_mask)
    for epoch in range(1, 501):
        model.observe(data, task)
        val_acc = evaluate(model, val_data, f1)
        if es(val_acc):
            model.load_state_dict(torch.load(es.path))
            return epoch
    return 500


def evaluate(model, data, f1=False):
    model.eval()
    pred = model(data).argmax(dim=1)
    if f1:
        return metrics.f1_score(data.y.cpu(), pred.cpu())
    correct = (pred == data.y).sum()
    return int(correct) / int(data.x.size(0))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = torch.load("data/csbm/hom_04.pt.pt")
    num_features = data_list[0].x.size(1)
    num_classes = torch.unique(data_list[0].y[data_list[0].train_mask]).numel()
    t = len(data_list)

    gcn_ret = GCN(num_features, num_classes).to(device)
    ret_mat = torch.empty((t, t), dtype=torch.float)

    reg_gcn = Twp(GCN(num_features, num_classes).to(device))
    reg_mat = torch.empty((t, t), dtype=torch.float)

    er_gcn = ExperienceReplay(GCN(num_features, num_classes).to(device), num_classes)
    er_mat = torch.empty((t, t), dtype=torch.float)

    gcn_list = [gcn_ret, reg_gcn, er_gcn]
    matrix_list = [ret_mat, reg_mat, er_mat]

    for i, data_i in enumerate(data_list):
        for k in range(len(gcn_list)):
            gcn = gcn_list[k]
            matrix = matrix_list[k]
            ep = train(gcn, data_i.clone().to(device), i)
            print(f'Early stopped after {ep} epochs')

            for j, data_j in enumerate(data_list):
                matrix[i][j] = evaluate(gcn, data_j.clone().subgraph(data_j.test_mask).to(device))
        print()
    print("GCN".ljust(20), '|', f'AP: {get_average_accuracy(ret_mat, ):.2f}', '|',
          f'AF:{get_average_forgetting_measure(ret_mat):.2f}')
    print("Regularization".ljust(20), '|', f'AP: {get_average_accuracy(reg_mat):.2f}', '|',
          f'AF:{get_average_forgetting_measure(reg_mat):.2f}')
    print("Experience Replay".ljust(20), '|', f'AP: {get_average_accuracy(er_mat):.2f}', '|',
          f'AF:{get_average_forgetting_measure(er_mat):.2f}')

    torch.set_printoptions(precision=2)
    print(ret_mat, '\n')
    print(reg_mat, '\n')
    print(er_mat, '\n')
    torch.save(ret_mat, "gcn_gll.pt")

    full_graph = data_list[-1]
    train_mask = torch.zeros(full_graph.x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(full_graph.x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(full_graph.x.size(0), dtype=torch.bool)
    for data in data_list:
        indices = torch.where(data.train_mask)[0]
        train_mask[indices] = True

        indices = torch.where(data.val_mask)[0]
        val_mask[indices] = True

        indices = torch.where(data.test_mask)[0]
        test_mask[indices] = True
    full_graph.train_mask = train_mask
    full_graph.val_mask = val_mask
    full_graph.test_mask = test_mask
    gcn = GCN(num_features, num_classes).to(device)
    result = []
    train(gcn, full_graph, 0, False)
