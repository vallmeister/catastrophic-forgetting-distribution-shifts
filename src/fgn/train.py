import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn import metrics

import measures
from early_stopping import EarlyStopping
from models import LGL, AFGN, PlainNet

## AFGN is LGL with attention; AttnPlainNet is the PlainNet with attention
nets = {'lgl': LGL, 'afgn': AFGN, 'plain': PlainNet}

CSBMs = ['base', 'class', 'feat', 'hom', 'struct', 'zero']
REAL_WORLD = ['dblp', 'elliptic', 'ogbn']

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BATCH_SIZE = 32


def get_neighbor_features(data_i, dev):
    self_loops = torch.tensor([i for i in range(data_i.x.size()[0])], dtype=torch.long).to(dev)
    source = data_i.edge_index[0]
    source = torch.cat((source, self_loops), 0)
    target = data_i.edge_index[1]
    target = torch.cat((target, self_loops), dim=0)
    neighbor_list = [target[source == i].tolist() for i in range(data_i.x.size()[0])]
    return neighbor_list


def train(model, data_i, dev, crit, optim):
    neighbor_feature_list = get_neighbor_features(data_i, dev)
    neighbor = []
    inputs = data_i.x[data_i.train_mask].unsqueeze(1)
    targets = data_i.y[data_i.train_mask]
    for i in torch.where(data_i.train_mask)[0].tolist():
        neighbor.append(torch.stack([data_i.x[j].unsqueeze(0) for j in neighbor_feature_list[i]]))
    train_loss = 0
    logger.info(f'Start training with 500 epochs and batch-size={BATCH_SIZE}')
    es = EarlyStopping(Path('fgn_config.pt'), model)
    for epoch in range(1, 501):
        model.train()
        t_loss = 0
        for i in range(math.ceil(inputs.size(0) / BATCH_SIZE)):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            optim.zero_grad()
            outputs = model(inputs[start:end], neighbor[start:end])
            loss = crit(outputs, targets[start:end])
            loss.backward()
            optim.step()
            t_loss += loss.item()
        train_loss += t_loss
        if epoch % 20 == 0:
            logger.info(f'Epoch: {epoch:03d} loss: {t_loss:.2f}')
        val_acc = validate(model, data_i, dev)
        if es(val_acc):
            logger.info(f'Early stopped after {epoch} epochs')
            model.load_state_dict(torch.load(es.path))
            break
    logger.info(f'Total loss: {train_loss:.2f}')


def validate(model, data_i, dev):
    model.eval()
    val_input = data_i.x[data_i.val_mask].unsqueeze(1)
    neighbor_feature_list = get_neighbor_features(data_i, dev)
    val_indices = torch.where(data_i.val_mask)[0].tolist()
    test_neighbors = [
        torch.stack([data_i.x[j].unsqueeze(0) for j in (set(val_indices) & set(neighbor_feature_list[i]))]) for i in
        val_indices]
    predictions = []
    for i in range(math.ceil(val_input.size(0) / BATCH_SIZE)):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        predictions.append(model(val_input[start:end], test_neighbors[start:end]).argmax(dim=1))
    pred = torch.cat(predictions, dim=0)
    correct = (pred == data_i.y[data_i.val_mask]).sum()
    val_acc = int(correct) / int(data_i.val_mask.sum())
    return val_acc


def evaluate(model, data_i, dev, f1=False):
    model.eval()
    test_input = data_i.x[data_i.test_mask].unsqueeze(1)
    neighbor_feature_list = get_neighbor_features(data_i, dev)
    test_indices = torch.where(data_i.test_mask)[0].tolist()
    test_neighbors = [
        torch.stack([data_i.x[j].unsqueeze(0) for j in (set(test_indices) & set(neighbor_feature_list[i]))]) for i in
        test_indices]
    predictions = []
    for i in range(math.ceil(test_input.size(0) / BATCH_SIZE)):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        predictions.append(model(test_input[start:end], test_neighbors[start:end]).argmax(dim=1))
    pred = torch.cat(predictions, dim=0)
    test_labels = data_i.y[data_i.test_mask]
    if f1:
        return metrics.f1_score(test_labels.cpu(), pred.cpu())
    correct = (pred == test_labels).sum()
    acc = int(correct) / int(data_i.test_mask.sum())
    return acc


def learn(data_list, f1=False):
    tasks = len(data_list)
    logger.info(f'T={tasks}')
    Net = nets['plain']
    fgn = Net(feat_len=data_list[0].x.size()[1], num_class=torch.unique(data_list[0].y).numel()).to(device)
    logger.info(fgn)
    r_matrix = torch.empty((tasks, tasks), dtype=torch.float)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fgn.parameters(), lr=0.01, weight_decay=0.001)
    for i, data_i in enumerate(data_list):
        data_i = data_i.to(device)
        train(fgn, data_i, device, criterion, optimizer)
        for j, data_j in enumerate(data_list):
            data_j = data_j.to(device)
            acc = evaluate(fgn, data_j, device, f1)
            r_matrix[i][j] = acc
    logger.info(f'Results:\n{r_matrix}')
    return r_matrix


if __name__ == '__main__':
    path_name = './fgn_results/'
    os.makedirs(path_name, exist_ok=True)
    torch.set_printoptions(precision=2, sci_mode=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using {device}')

    for dataset in CSBMs:
        pt_path = f'{dataset}_01_result_matrix.pt'
        if pt_path in os.listdir('./fgn_results/'):
            logger.info(f'{pt_path} already processed')
            continue
        dataset_list = torch.load(f'./data/csbm/{dataset}_01.pt')
        result_matrix = learn(dataset_list)
        torch.save(result_matrix, f'./fgn_results/{pt_path}')
        print(dataset.ljust(10), '|',
              f'AP: {measures.get_average_accuracy(result_matrix):.2f}',
              f'AF: {measures.get_average_forgetting_measure(result_matrix):.2f}')

    for dataset in REAL_WORLD:
        BATCH_SIZE = 16
        dataset_list = torch.load(f'./data/real_world/{dataset}_tasks.pt')
        result_matrix = learn(dataset_list, dataset == 'elliptic')
        pt_path = f'{path_name}{dataset}_result_matrix.pt'
        torch.save(result_matrix, pt_path)
        logger.info(f'Saved result matrix R for {dataset}\n')
