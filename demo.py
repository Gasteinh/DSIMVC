import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import faiss
import argparse
from meta_network import WNet, SafeNetwork, Online
from network import Network
from dataloader import load_data, MultiviewDataset, RandomSampler
from loss import Loss
from make_mask import get_mask
from evaluation import evaluate
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset', default='bdgp')
parser.add_argument("--view", type=int, default=2)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", type=int, default=128)
parser.add_argument('--lr_wnet', type=float, default=0.0004)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument("--epochs", default=120)
parser.add_argument('--lr_decay_factor', type=float, default=0.2)
parser.add_argument('--lr_decay_iter', type=int, default=20)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--initial_epochs', type=int, default=100)
parser.add_argument('--pretrain_epochs', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--miss_rate', default=0.1, type=float)
parser.add_argument('--T', default=10, type=int)
parser.add_argument('--iterations', default=200, type=int)
args = parser.parse_args()


data_list, Y, dims, total_view, data_size, class_num = load_data(args.dataset)
view = total_view
miss_rate = args.miss_rate
incomplete_loader = None

if args.dataset not in ['ccv']:
    for v in range(total_view):
        min_max_scaler = MinMaxScaler()
        data_list[v] = min_max_scaler.fit_transform(data_list[v])
record_data_list = copy.deepcopy(data_list)


if args.dataset == 'bdgp':
    args.initial_epochs = 30
    args.pretrain_epochs = 100
    args.iterations = 100
if args.dataset == 'mnist_usps':
    args.initial_epochs = 80
    args.pretrain_epochs = 100
    args.iterations = 200
if args.dataset == 'ccv':
    args.initial_epochs = 30
    args.pretrain_epochs = 100
    args.iterations = 300
if args.dataset == 'multi-fashion':
    args.initial_epochs = 100
    args.pretrain_epochs = 200
    args.iterations = 300


def get_model():
    return SafeNetwork(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device)


def pretrain(com_dataset):
    """
    pretraining on complete data
    :return: parameters of the pretraining model
    """
    print("Initializing network parameters...")
    pretrain_model = Online(view, dims, args.feature_dim).to(device)
    loader = DataLoader(com_dataset, batch_size=args.batch_size, shuffle=True)
    opti = torch.optim.Adam(pretrain_model.params(), lr=0.0003)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.pretrain_epochs):
        for batch_idx, (xs, _, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            xrs = pretrain_model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)

            opti.zero_grad()
            loss.backward()
            opti.step()
    return pretrain_model.state_dict()


def bi_level_train(model, criterion, optimizer, class_num, view,
             com_loader, full_loader, mask, incomplete_ind):
    wnet_label = WNet(class_num, 100, 1).to(device)
    memory = Memory()
    memory.bi = True
    wnet_label.train()
    iteration = 0

    optimizer_wnet_label = torch.optim.Adam(wnet_label.params(), lr=args.lr_wnet)

    for com_batch, incomplete_batch in zip(com_loader, incomplete_loader):
        xs, _, _ = com_batch
        incomplete_xs, _, _ = incomplete_batch
        iteration += 1
        for v in range(view):
            xs[v] = xs[v].to(device)
            incomplete_xs[v] = incomplete_xs[v].to(device)

        model.train()
        meta_net = get_model()
        meta_net.load_state_dict(model.state_dict())

        com_hs, com_qs, incomplete_hs, incomplete_qs = meta_net(xs, incomplete_xs)

        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(criterion.forward_label(com_qs[v], com_qs[w]))
        loss_hat = sum(loss_list)

        cost_w_labels = []
        cost_w_features = []
        for v in range(view):
            for w in range(v+1, view):
                l_f, l_l = criterion.forward_feature2(incomplete_hs[v], incomplete_hs[w]), criterion.forward_label(incomplete_qs[v], incomplete_qs[w])
                cost_w_labels.append(l_l)
                cost_w_features.append(l_f)

        weight_label = wnet_label(sum(incomplete_qs)/view)
        norm_label = torch.sum(weight_label)

        for v in range(len(cost_w_labels)):
            if norm_label != 0:
                loss_hat += (torch.sum(cost_w_features[v] * weight_label)/norm_label
                                    + torch.sum(cost_w_labels[v]*weight_label) / norm_label)
            else:
                loss_hat += torch.sum(cost_w_labels[v] * weight_label + cost_w_features[v]*weight_label)

        meta_net.zero_grad()
        grads = torch.autograd.grad(loss_hat, (meta_net.params()), create_graph=True)
        meta_net.update_params(lr_inner=args.meta_lr, source_params=grads)
        del grads

        com_hs, com_qs, _, _ = meta_net(xs, incomplete_xs)

        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(criterion.forward_label(com_qs[v], com_qs[w]))

        l_g_meta = sum(loss_list)

        optimizer_wnet_label.zero_grad()
        l_g_meta.backward()
        optimizer_wnet_label.step()

        com_hs, com_qs, incomplete_hs, incomplete_qs = model(xs, incomplete_xs)

        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(criterion.forward_label(com_qs[v], com_qs[w]))

        loss = sum(loss_list)

        cost_w_labels = []
        cost_w_features = []
        for v in range(view):
            for w in range(v+1, view):
                l_f, l_l = criterion.forward_feature2(incomplete_hs[v], incomplete_hs[w]), criterion.forward_label(incomplete_qs[v], incomplete_qs[w])
                cost_w_labels.append(l_l)
                cost_w_features.append(l_f)

        with torch.no_grad():
            weight_label = wnet_label(sum(incomplete_qs)/view)
            norm_label = torch.sum(weight_label)

        for v in range(len(cost_w_labels)):
            if norm_label != 0:
                loss += (torch.sum(cost_w_labels[v] * weight_label)/norm_label
                                + torch.sum(cost_w_features[v]*weight_label) / norm_label)
            else:
                loss += torch.sum(cost_w_labels[v] * weight_label + cost_w_features[v]*weight_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        memory.update_feature(model, full_loader, mask, incomplete_ind, iteration)

    acc, nmi, pur = valid(model, mask)

    return acc, nmi, pur


def valid(model, mask):
    pred_vec = []
    with torch.no_grad():
        input_data = []
        for v in range(view):
            data_v = torch.from_numpy(record_data_list[v]).to(device)
            input_data.append(data_v)
        output, _ = model.forward_cluster(input_data)
        for v in range(view):
            miss_ind = mask[:, v] == 0
            output[v][miss_ind] = 0
        sum_ind = np.sum(mask, axis=1, keepdims=True)
        output = sum(output)/torch.from_numpy(sum_ind).to(device)
        pred_vec.extend(output.detach().cpu().numpy())

    pred_vec = np.argmax(np.array(pred_vec), axis=1)
    acc, nmi, pur = evaluate(Y, pred_vec)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(acc, nmi, pur))
    return acc, nmi, pur


class Memory:
    def __init__(self):
        self.features = None
        self.alpha = args.alpha
        self.interval = args.interval
        self.bi = False

    def cal_cur_feature(self, model, loader):
        features = []
        for v in range(view):
            features.append([])

        for _, (xs, y, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            with torch.no_grad():
                if self.bi:
                    hs, _, _ = model.forward_xs(xs)
                else:
                    hs, _, _ = model(xs)
                for v in range(view):
                    fea = hs[v].detach().cpu().numpy()
                    features[v].extend(fea)

        for v in range(view):
            features[v] = np.array(features[v])

        return features

    def update_feature(self, model, loader, mask, incomplete_ind, epoch):
        topK = 600
        model.eval()
        cur_features = self.cal_cur_feature(model, loader)
        indices = []
        if epoch == 1:
            self.features = cur_features
            for v in range(view):
                fea = np.array(self.features[v])
                n, dim = fea.shape[0], fea.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(fea)
                _, ind = index.search(fea, topK + 1)  # Sample itself is included
                indices.append(ind[:, 1:])
            return indices
        elif epoch % self.interval == 0:
            for v in range(view):
                f_v = (1-self.alpha)*self.features[v] + self.alpha*cur_features[v]
                self.features[v] = f_v/np.linalg.norm(f_v, axis=1, keepdims=True)

                n, dim = self.features[v].shape[0], self.features[v].shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(self.features[v])
                _, ind = index.search(self.features[v], topK + 1)  # Sample itself is included
                indices.append(ind[:, 1:])
            if self.bi:
                make_imputation(mask, indices, incomplete_ind)
            return indices


def make_imputation(mask, indices, incomplete_ind):
    global data_list

    for v in range(view):
        for i in range(data_size):
            if mask[i, v] == 0:
                predicts = []
                for w in range(view):
                    # only the available views are selected as neighbors
                    if w != v and mask[i, w] != 0:
                        neigh_w = indices[w][i]
                        for n_w in range(neigh_w.shape[0]):
                            if mask[neigh_w[n_w], v] != 0 and mask[neigh_w[n_w], w] != 0:
                                predicts.append(data_list[v][neigh_w[n_w]])
                            if len(predicts) >= args.K:
                                break

                assert len(predicts) >= args.K
                fill_sample = np.mean(predicts, axis=0)
                data_list[v][i] = fill_sample

    global incomplete_loader
    incomplete_data = []
    for v in range(view):
        incomplete_data.append(data_list[v][incomplete_ind])
    incomplete_label = Y[incomplete_ind]
    incomplete_dataset = MultiviewDataset(view, incomplete_data, incomplete_label)
    incomplete_loader = DataLoader(
        incomplete_dataset, args.batch_size, drop_last=True,
        sampler=RandomSampler(len(incomplete_dataset), args.iterations * args.batch_size)
    )


def initial(com_dataset, full_loader, criterion, mask, incomplete_ind):
    print("Initializing neighbors...")
    online_net = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device)
    loader = DataLoader(com_dataset, batch_size=256, shuffle=True, drop_last=True)
    mse_loader = DataLoader(com_dataset, batch_size=256, shuffle=True)
    opti = torch.optim.Adam(online_net.parameters(), lr=0.0003, weight_decay=0.)
    mse = torch.nn.MSELoss()

    memory = Memory()
    memory.interval = 1
    epochs = args.initial_epochs

    # pretraining on complete data

    for e in range(1, 201):
        for xs, _, _ in mse_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)

            xrs = online_net.forward_mse(xs)

            loss_list = []
            for v in range(view):
                loss_list.append(mse(xrs[v], xs[v]))
            loss = sum(loss_list)

            opti.zero_grad()
            loss.backward()
            opti.step()

    for e in range(1, epochs+1):
        for xs, _, _ in loader:
            for v in range(view):
                xs[v] = xs[v].to(device)

            hs, qs, _ = online_net(xs)

            loss_list = []
            for v in range(view):
                for w in range(v+1, view):
                    loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                    loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss = sum(loss_list)

            opti.zero_grad()
            loss.backward()
            opti.step()

    # initial neighbors by the pretrain model
    indices = memory.update_feature(online_net, full_loader, mask, incomplete_ind, epoch=1)
    make_imputation(mask, indices, incomplete_ind)


def main():
    result_record = {"ACC": [], "NMI": [], "PUR": []}
    for t in range(1, args.T+1):
        print("--------Iter:{}--------".format(t))

        data_list = copy.deepcopy(record_data_list)
        mask = get_mask(view, data_size, miss_rate)
        sum_vec = np.sum(mask, axis=1, keepdims=True)
        complete_index = (sum_vec[:, 0]) == view
        mv_data = []
        for v in range(view):
            mv_data.append(data_list[v][complete_index])
        mv_label = Y[complete_index]
        com_dataset = MultiviewDataset(view, mv_data, mv_label)
        com_loader = DataLoader(
            com_dataset, args.batch_size, drop_last=True,
            sampler=RandomSampler(len(com_dataset), args.iterations * args.batch_size)
        )
        full_dataset = MultiviewDataset(view, data_list, Y)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
        incomplete_ind = (sum_vec[:, 0]) != view

        model = get_model()
        state_dict = pretrain(com_dataset)
        model.load_state_dict(state_dict, strict=False)
        optimizer = torch.optim.Adam(model.params(), lr=0.0003, weight_decay=0.)
        criterion = Loss(args.batch_size, class_num, view, device)
        initial(com_dataset, full_loader, criterion, mask, incomplete_ind)
        acc, nmi, pur = bi_level_train(model, criterion, optimizer, class_num, view, com_loader,
                 full_loader, mask, incomplete_ind)
        result_record["ACC"].append(acc)
        result_record["NMI"].append(nmi)
        result_record["PUR"].append(pur)

    print("----------------Training Finish----------------")
    print("----------------Final Results----------------")
    print("ACC (mean) = {:.4f} ACC (std) = {:.4f}".format(np.mean(result_record["ACC"]), np.std(result_record["ACC"])))
    print("NMI (mean) = {:.4f} NMI (std) = {:.4f}".format(np.mean(result_record["NMI"]), np.std(result_record["NMI"])))
    print("PUR (mean) = {:.4f} PUR (std) = {:.4f}".format(np.mean(result_record["PUR"]), np.std(result_record["PUR"])))


if __name__ == '__main__':
    main()
