import torch
from models import GCN, Twp, ExperienceReplay
from measures import get_average_forgetting_measure, get_average_accuracy


def evaluate(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    return int(correct) / int(data.test_mask.sum())


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = torch.load("data/csbm/hom_03.pt")
    num_features = data_list[0].x.size(1)
    num_classes = torch.unique(data_list[0].y).numel()
    t = len(data_list)

    gcn_retraining = GCN(num_features, num_classes).to(device)
    retraining_matrix = torch.empty((t, t), dtype=torch.float)

    gcn_no_retraining = GCN(num_features, num_classes).to(device)
    no_retraining_matrix = torch.empty((t, t), dtype=torch.float)

    reg_gcn = Twp(GCN(num_features, num_classes).to(device))
    reg_matrix = torch.empty((t, t), dtype=torch.float)

    er_gcn = ExperienceReplay(GCN(num_features, num_classes).to(device))
    er_matrix = torch.empty((t, t), dtype=torch.float)

    for i, data_i in enumerate(data_list):
        for epoch in range(1, 101):
            gcn_retraining.observe(data_i.clone().to(device))
            reg_gcn.observe(data_i.clone().to(device), i)
            er_gcn.observe(data_i.clone().to(device))
            if i == 0:
                gcn_no_retraining.observe(data_i.clone().to(device))
        for j, data_j in enumerate(data_list):
            retraining_matrix[i][j] = evaluate(gcn_retraining, data_j.clone().to(device))
            no_retraining_matrix[i][j] = evaluate(gcn_no_retraining, data_j.clone().to(device))
            reg_matrix[i][j] = evaluate(reg_gcn, data_j.clone().to(device))
            er_matrix[i][j] = evaluate(er_gcn, data_j.clone().to(device))

    print("GCN".ljust(25), '|', f'AP: {get_average_accuracy(retraining_matrix):.2f}', '|',
          f'AF:{get_average_forgetting_measure(retraining_matrix):.2f}')
    print("GCN cold".ljust(25), '|', f'AP: {get_average_accuracy(no_retraining_matrix):.2f}', '|',
          f'AF:{get_average_forgetting_measure(no_retraining_matrix):.2f}')
    print("Regularization".ljust(25), '|', f'AP: {get_average_accuracy(reg_matrix):.2f}', '|',
          f'AF:{get_average_forgetting_measure(reg_matrix):.2f}')
    print("Experience Replay".ljust(25), '|', f'AP: {get_average_accuracy(er_matrix):.2f}', '|',
          f'AF:{get_average_forgetting_measure(er_matrix):.2f}')
