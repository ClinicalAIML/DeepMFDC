import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from network_fdbase import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from Sim_Scenario2_1223 import load_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import json


Dataname = 'fd_v2_c3'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=80)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=10)
parser.add_argument("--feature_dim", default=128)
parser.add_argument("--high_feature_dim", default=256)
parser.add_argument("--n_base", default=6)
parser.add_argument('--scaler', default='norm',type=str)
parser.add_argument("--data_num", default=100)
parser.add_argument("--view", default=2)
parser.add_argument("--num_class", default=3)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = np.random.randint(10000)

sample_size = 600
time_grid = 101


def setup_seed(seed):
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _, regs1, regs2 = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
            loss_list.append(regs1[v])
            loss_list.append(regs2[v])
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, regs1, regs2 = model(xs)
        #loss_list = []
        loss_list_feature = []
        loss_list_label = []
        loss_list_recon = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list_feature.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list_label.append(criterion.forward_label(qs[v], qs[w]))
            loss_list_recon.append(mes(xs[v], xrs[v]))
        loss_feature = sum(loss_list_feature)
        loss_label = sum(loss_list_label)
        loss_recon = sum(loss_list_recon)

        #loss = sum(loss_list)
        precision_a = torch.exp(-log_var_a)
        precision_b = torch.exp(-log_var_b)
        precision_c = torch.exp(-log_var_c)
        loss_a = precision_a*loss_feature + log_var_a
        loss_b = precision_b*loss_label + log_var_b
        loss_c = precision_c*loss_recon + log_var_c
        loss = loss_a + loss_b +loss_c
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


for num in range(args.data_num):
    num_exp = 1
    path = f'./results_FDA/v{args.view}_c{args.num_class}_s{sample_size}/exp_{num_exp}/{num}'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/' + f'args_output_{args.view}.json', "w") as file:
        json.dump(vars(args), file, indent=4)


    dataset, dims, view, data_size, class_num = load_data(args.dataset, sample_size, time_grid, args.view, args.num_class, num, path)

    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    # accs = []
    # nmis = []
    # aris = []
    # purs = []
    if not os.path.exists('./models'):
        os.makedirs('./models')
    T = 1
    for i in range(T):
        print("ROUND:{}".format(i+1))
        setup_seed(seed)
        model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, args.n_base, device)
        print(model)
        model = model.to(device)
        log_var_a = torch.nn.Parameter(torch.zeros(1, device=device))
        log_var_b = torch.nn.Parameter(torch.zeros(1, device=device))
        log_var_c = torch.nn.Parameter(torch.zeros(1, device=device))

        params = list(model.parameters()) + [log_var_a, log_var_b, log_var_c]
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

        epoch = 1
        while epoch <= args.mse_epochs:
            pretrain(epoch)
            epoch += 1
        while epoch <= args.mse_epochs + args.con_epochs:
            contrastive_train(epoch)
            if epoch == args.mse_epochs + args.con_epochs:
                acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
                try:
                    with open(path + '/' + f'args_output_{args.view}.json', 'r') as file:
                        data_out = json.load(file)
                except FileNotFoundError:
                    data_out = {}
                #data_out['training_sessions'] = [training_results]
                data_out["seed"] = seed
                data_out["accuracy_nofine"] = acc
                data_out["NMIs_nofine"] = nmi
                data_out["ARIs_nofine"] = ari
                data_out["PURs_nofine"] = pur
                with open(path + '/' + f'args_output_{args.view}.json', 'w') as file:
                    json.dump(data_out, file, indent=4)
            epoch += 1
        new_pseudo_label = make_pseudo_label(model, device)
        while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
            fine_tuning(epoch, new_pseudo_label)
            if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
                acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
                state = model.state_dict()
                torch.save(state, path + '/' + args.dataset + '.pth')
                print('Saving..')

                try:
                    with open(path + '/' + f'args_output_{args.view}.json', 'r') as file:
                        data_out = json.load(file)
                except FileNotFoundError:
                    data_out = {}
                #data_out['training_sessions'] = [training_results]
                data_out["accuracy"] = acc
                data_out["NMIs"] = nmi
                data_out["ARIs"] = ari
                data_out["PURs"] = pur
                with open(path + '/' + f'args_output_{args.view}.json', 'w') as file:
                    json.dump(data_out, file, indent=4)
            epoch += 1
