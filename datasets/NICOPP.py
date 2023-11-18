from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from tqdm import tqdm
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

imgsize = 224
def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            b = line.split('/')
            b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
            data_path =f"{'/'.join(b)}"
            # data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return np.array(data_paths), np.array(data_labels)


def read_domainnet_split(dataset_path, domain_name,split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
    out = [[[],[]] for i in range(60)]
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            b = line.split('/')
            # if ' ' in b[2]:
            #     b[2] = f"'{b[2]}'"
            b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
            data_path = f"{'/'.join(b)}"
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            
            out[label][0].append(data_path)
            out[label][1].append(label)
    return out

def read_domainnet_all(dataset_path, domain_names, split="train"):
    data_paths = []
    data_labels = []
    for i,domain_name in enumerate(domain_names):
        split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                b = line.split('/')
                # if ' ' in b[2]:
                #     b[2] = f"'{b[2]}'"
                b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
                data_path = f"{'/'.join(b)}"
                data_path = path.join(dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels

def read_data_num(dataset_path, domain_name, split="train", maxnum = 10):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
    ind = [0]*60
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            b = line.split('/')
            # if ' ' in b[2]:
            #     b[2] = f"'{b[2]}'"
            b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
            data_path = f"{'/'.join(b)}"
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            if ind[label]==maxnum:
                continue
            else:
                data_paths.append(data_path)
                data_labels.append(label)
                ind[label]+=1
            if sum(ind)>(maxnum*60+5):break
            
    return data_paths, data_labels


class Nicopp(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name, weight_model=None):
        super(Nicopp, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name
        self.weight_model = weight_model

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index] 
        img = self.transforms(img)
        if self.weight_model!=None:
            img = self.weight_model(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)



def get_nico_dloader(base_path, domain_name, batch_size, num_workers=16):
    dataset_path = path.join(base_path)
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
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
    train_dataset = Nicopp(train_data_paths, train_data_labels, preprocess, domain_name)
    test_dataset = Nicopp(test_data_paths, test_data_labels, preprocess, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=256, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    train_loader_p = DataLoader(train_dataset, batch_size=256, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    return train_dloader, test_dloader, train_loader_p, train_dataset.__len__()



def get_nico_dataset(base_path, domain_name, batch_size,alpha=0.01,clients_for_eachdomain=5,num_workers=8):
    dataset_path = path.join(base_path)
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
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
    test_dataset = Nicopp(test_data_paths, test_data_labels, preprocess, domain_name)
    
    net_dataidx_map,traindata_cls_counts =partition(alpha, clients_for_eachdomain, train_data_labels,classes=60)
    if domain_name=='autumn': print(traindata_cls_counts)
    clients = [Nicopp(train_data_paths[net_dataidx_map[idxes]], train_data_labels[net_dataidx_map[idxes]], preprocess, domain_name) \
               for idxes in net_dataidx_map]
    clients_loader = [DataLoader(client, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True) \
                      for client in clients]
    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                            shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=256, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    # train_loader_p = DataLoader(train_dataset, batch_size=256, num_workers=num_workers, pin_memory=True,
    #                            shuffle=False)
    return clients_loader,test_dloader#,[DataLoader(client, batch_size=64, num_workers=num_workers, pin_memory=True,shuffle=True) for client in clients]


def get_nico_dloader_split(base_path,domain_name, batch_size):
    dataset_path = path.join(base_path,)
    out = read_domainnet_split(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(imgsize, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((imgsize,imgsize)),
        transforms.ToTensor()
    ])

    out = [Nicopp(client[0], client[1], transforms_train, domain_name) for client in tqdm(out)]
    test_dataset = Nicopp(test_data_paths, test_data_labels, transforms_test, domain_name)
    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                            shuffle=True)
    # test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                           shuffle=False)
    return out, test_dataset

def get_nico_dloader_all(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path)
    train_data_paths, train_data_labels = read_domainnet_all(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_all(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(imgsize, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((imgsize,imgsize)),
        transforms.ToTensor()
    ])

    train_dataset = Nicopp(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transforms_test, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    return train_dloader, test_dloader



def read_domainnet_split(dataset_path,domain_name):
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, "train"))
    out = [[[],[]] for i in range(60)]
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            
            out[label][0].append(data_path)
            out[label][1].append(label)
    return out

class imagenet_truncated(Dataset):

    def __init__(self, root, dataidxs=None,transform=None, target_transform=None, download=False, clss=None):

        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.clss = clss
        self.data = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        data = []

        if self.dataidxs==None:
            return self.root
        else:
            for i in self.dataidxs:
                data.append(self.root[i])
                # target = target[self.dataidxs]

            return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index][0],self.data[index][1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # if self.clss!=None: return img, target//self.clss
        return img, target

    def __len__(self):
        return len(self.data)
    
    
def partition(alphaa, n_netss, y_train,classes=60):
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