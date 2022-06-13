from torch.utils.data import Dataset, Sampler
import numpy as np
import torch


class MultiviewDataset(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            data.append(torch.tensor(self.data_list[i][idx].astype('float32')))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


def load_data(name):
    """
    :param name: name of dataset
    :return:
    data_list: python list containing all views, where each view is represented as numpy array
    labels: ground_truth labels represented as numpy array
    dims: python list containing dimension of each view
    num_views: number of views
    data_size: size of data
    class_num: number of category
    """
    data_path = "./data/"
    path = data_path + name + '.npz'
    data = np.load(path)
    num_views = int(data['n_views'])
    data_list = []
    for i in range(num_views):
        x = data[f"view_{i}"]
        if len(x.shape) > 2:
            x = x.reshape([x.shape[0], -1])
        data_list.append(x.astype(np.float32))
    labels = data['labels']
    dims = []
    for i in range(num_views):
        dims.append(data_list[i].shape[1])
    class_num = labels.max() + 1
    data_size = data_list[0].shape[0]

    return data_list, labels, dims, num_views, data_size, class_num


class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
