from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
from tqdm import tqdm

from torch.utils.data.dataset import ConcatDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

imgsize = 224
def read_domainnet_class(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    for i,dom in enumerate(domain_name):
        split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(dom, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(dataset_path, data_path)
                # if int(label)>=60: continue
                label = int(i)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels




def read_domainnet_data(dataset_path, domain_name,split="train",classnum=100):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            if int(label)>=classnum: continue
            data_path = path.join(dataset_path, data_path)
            data_paths.append(data_path)
            data_labels.append(int(label))
    return data_paths, data_labels



def read_domainnet_all(dataset_path, domain_names, split="train"):
    data_paths = []
    data_labels = []
    for i,domain_name in enumerate(domain_names):
        split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels

def read_data_num(dataset_path, domain_name, split="train", maxnum = 10):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    ind = [0]*345
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            if ind[label]==maxnum:
                continue
            else:
                data_paths.append(data_path)
                data_labels.append(label)
                ind[label]+=1
            if sum(ind)>(maxnum*345+5):break
            
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index] 
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(base_path,domain_name, batch_size, num_workers=16):
    dataset_path = path.join(base_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    train_dataset = DomainNet(train_data_paths, train_data_labels, preprocess, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)
    return train_dataset,test_dataset



def get_domainnet_dataset(base_path,domain_name, batch_size,alpha=0.01,clients_for_eachdomain=5,num_workers=8,classnum = 100):
    dataset_path = path.join(base_path)
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train",classnum=classnum)
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test",classnum=classnum)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    # train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)
    train_data_labels  = np.array(train_data_labels)
    train_data_paths = np.array(train_data_paths)
    net_dataidx_map,traindata_cls_counts =partition(alpha, clients_for_eachdomain, train_data_labels,classes=100)
    if domain_name=='clipart': print(traindata_cls_counts)
    clients = [DomainNet(train_data_paths[net_dataidx_map[idxes]], train_data_labels[net_dataidx_map[idxes]], preprocess, domain_name) for idxes in net_dataidx_map]
    clients_loader = [DataLoader(client, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True) \
                      for client in clients]
    test_dloader = DataLoader(test_dataset, batch_size=256, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    
    return clients_loader,test_dloader#[cl.__len__() for cl in clients]


def get_domainnet_dataset_all(domain_names, batch_size,preprocess,alpha=0.01,clients_for_eachdomain=5,num_workers=8):
    dataset_path = path.join('/home/share/DomainNet')
    
    trains = [[] for _ in range(clients_for_eachdomain)]
    trainsloader = []
    tests = []
    nums = [0 for _ in range(clients_for_eachdomain)]
    for d,domain_name in enumerate(domain_names):
        train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
        test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
        
        test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)
        test_dloader = DataLoader(test_dataset, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True,
                                  shuffle=False)
        tests.append(test_dloader)
        
        
        train_data_labels  = np.array(train_data_labels)
        train_data_paths = np.array(train_data_paths)
        net_dataidx_map,_ =partition(alpha, clients_for_eachdomain, train_data_labels)
        
        tmp = []
        for idxes in net_dataidx_map:
            ccc = DomainNet(train_data_paths[net_dataidx_map[idxes]], train_data_labels[net_dataidx_map[idxes]], preprocess, domain_name)
            trains[idxes].append(ccc)
            nums[idxes]+=ccc.__len__()
            
    for data in trains:
        trainsloader.append(DataLoader(ConcatDataset(data), batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True))
        
    
    return trainsloader,tests,nums


def partition(alphaa, n_netss, y_train,classes=345):
    min_size = 0
    n_nets = n_netss
    N = y_train.shape[0]
    net_dataidx_map = {}
    alpha = alphaa
    K=classes
    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map,traindata_cls_counts

def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts