import torch
import random
import logging
from tqdm import tqdm
import numpy as np
import argparse
from datasets.DomainNet import get_domainnet_dataset
from datasets.NICOPP import get_nico_dataset
from utils import evaluation,evaluation_depthfl,lrcos
from client import Client
from server import Server



def parse_integer_list(input_string):
    try:
        return [int(x) for x in input_string[1:-1].split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("error ")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="/home/share/DomainNet")#/home/share/DomainNet /home/share/NICOpp
    parser.add_argument('--info', default='samelayer3',help='samelayerxx,inclusivefl,depthfl,ours')
    parser.add_argument('--alpha', default=100,type=float,help='degree of non-iid')
    parser.add_argument('--data', default='domainnet',help='domainnet or nico')
    parser.add_argument('--seed', default=0,type=int,)
    parser.add_argument('--batch_size', default=128, type=int,)
    parser.add_argument('--modeltype', default='ViT',help='ViT or mixer')
    parser.add_argument('--rounds', default=100, type=int,)
    parser.add_argument('--distribution', default='feature',help='feature or feature&label')
    parser.add_argument('--clientepoch', default=1,type=int,)
    parser.add_argument('--learningrate', default=0.01,type=float,)
    parser.add_argument('--domains', default=6,type=int,)
    parser.add_argument('--net_type', default=[12,10,8,6,4,3],type=parse_integer_list, help='[12,10,8,6,4,3],[3,3,3,3,3,3]...')
    parser.add_argument('--numclients_ineachround', default=6,type=int,)
    parser.add_argument('--clients_for_eachdomain', default=1,type=int,help='')
    return parser

########################################################################################################################
parser = get_parser()
args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

logging.basicConfig(
    filename=f'./finallogs/{args.modeltype}_{args.data}{args.net_type[0]}_{args.info}_{args.alpha}_{args.distribution}{args.clients_for_eachdomain}.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#======================= prepare dataset AND clients AND server==========================================

# For DepthFL, we treat every 3 layers as a block. 
depth_cls=3 if 'depthfl' in args.info else 0
        
if args.data  == 'domainnet': 
    num_classes = 100
    domainnums = 6
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    client_layers = []
    for layer_type in args.net_type:#
        client_layers.extend([layer_type]*args.clients_for_eachdomain)
    if 'samelayer' in args.info:
        client_layers = [int(args.info[9:])]*(domainnums*args.clients_for_eachdomain)

    clients,test_data  = [],[]
    A = np.zeros((domainnums*args.clients_for_eachdomain,12),dtype=int)
    for i in range(domainnums*args.clients_for_eachdomain):
        indices_to_one = np.random.choice(list(range(0,12)), client_layers[i], replace=False)
        A[i][indices_to_one] = 1
    if 'ours' not in args.info:
        A = [None]*(domainnums*args.clients_for_eachdomain)
    idx = 0
    for d,domain in enumerate(domains):
        trains,test = get_domainnet_dataset(args.base_path,domain, args.batch_size,args.alpha,clients_for_eachdomain=args.clients_for_eachdomain)
        test_data.append(test)
        for trainloader in trains:
            clients.append(Client(dataloader=trainloader, num_layers=client_layers[idx], num_classes=num_classes, depth_cls=depth_cls, modeltype = args.modeltype))
            idx+=1

elif args.data  == 'nico': 
    num_classes = 60
    domainnums = 6
    domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock','water']
    client_layers = []
    for layer_type in args.net_type:
        client_layers.extend([layer_type]*args.clients_for_eachdomain)
    if 'samelayer' in args.info:
        client_layers = [int(args.info[9:])]*(domainnums*args.clients_for_eachdomain)
    clients,test_data  = [],[]
    A = np.zeros((domainnums*args.clients_for_eachdomain,12),dtype=int)
    for i in range(domainnums*args.clients_for_eachdomain):
        indices_to_one = np.random.choice(list(range(0,12)), client_layers[i], replace=False)
        A[i][indices_to_one] = 1
    if 'ours' not in args.info:
        A = [None]*(domainnums*args.clients_for_eachdomain)
    idx = 0
    for d,domain in enumerate(domains):
        trains,test = get_nico_dataset(args.base_path,domain, args.batch_size,args.alpha,clients_for_eachdomain=args.clients_for_eachdomain)
        test_data.append(test)
        for trainloader in trains:
            clients.append(Client(dataloader=trainloader, num_layers=client_layers[idx], num_classes=num_classes,depth_cls=depth_cls, modeltype = args.modeltype))
            idx+=1
itersnum = [len(client.dataloader) for client in clients]
mmm = min(itersnum)

#=======================methods==========================================

if 'ours' in args.info:
    server = Server(A,num_layers=12,args=args,num_classes=num_classes,clayers = client_layers,depth_cls=depth_cls, modeltype = args.modeltype)
    for r in tqdm(range(args.rounds)):
        print(args.info)
        if args.distribution=='label':
            this_round_clients = list(range(args.clients_for_eachdomain))
        else:
            this_round_clients = sorted([np.random.choice(list(range(i*args.clients_for_eachdomain,i*args.clients_for_eachdomain+args.clients_for_eachdomain)), 1, replace=False)[0] for i in range(domainnums)])
        print(this_round_clients)
        lr = args.learningrate#lrcos(step=r,lr=args.learningrate,lr_min=0.001,T_max=args.rounds)
        
        tmpmodels = server.get_para_range(this_round_clients,r)
        for i in this_round_clients:
            clients[i].load_para(tmpmodels[i])  
            
        for i in this_round_clients:
            clients[i].train_baseline(lr, args.clientepoch,mmm,r,server.global_model)
        
        server.agg_range_fill([client.get_para() if i in this_round_clients else {} for i, client in enumerate(clients)],this_round_clients)
    
        print(f'round {r}')
        if r>args.rounds-5 or (r%20==0 and r>0):
            # torch.save(server.global_model.state_dict(),f'{args.modeltype}_{args.data}{args.net_type[0]}_{args.info}_{args.alpha}_{args.distribution}{args.clients_for_eachdomain}.pth')
            for i in range(len(test_data)):
                top1,_ = evaluation(server.global_model,test_data[i])
                logger.info('top1: %s' % str(top1))
                print(top1) 

if 'samelayer' in args.info:
    server = Server(A,num_layers=int(args.info[9:]),args=args,num_classes=num_classes,clayers = client_layers,depth_cls=depth_cls, modeltype = args.modeltype)
    for r in tqdm(range(args.rounds)):
        print(args.info)
        if args.distribution=='label':
            this_round_clients = list(range(args.clients_for_eachdomain))
        else:
            this_round_clients = sorted([np.random.choice(list(range(i*args.clients_for_eachdomain,i*args.clients_for_eachdomain+args.clients_for_eachdomain)), 1, replace=False)[0] for i in range(domainnums)])
        print(this_round_clients)
        lr = args.learningrate#lrcos(step=r,lr=args.learningrate,lr_min=0.001,T_max=args.rounds)
        
        for idx in this_round_clients:
            clients[idx].load_para(server.get_para_baseline())  
            
        for idx in this_round_clients:
            clients[idx].train_baseline(lr, args.clientepoch,mmm,r,server.global_model)
        
        server.agg_baseline([client.get_para() if i in this_round_clients else {} for i, client in enumerate(clients)])
        print(f'round {r}')
        if r>args.rounds-5 or (r%20==0 and r>0):
            for i in range(len(test_data)):
                top1,_ = evaluation(server.global_model,test_data[i])
                logger.info('top1: %s' % str(top1))
                print(top1)  
            
if 'inclusivefl' in args.info:
    server = Server(A,num_layers=12,args=args,num_classes=num_classes,clayers = client_layers,depth_cls=depth_cls, modeltype = args.modeltype)
    for r in tqdm(range(args.rounds)):
        print(args.info)
        if args.distribution=='label':
            this_round_clients = list(range(args.clients_for_eachdomain))
        else:
            this_round_clients = sorted([np.random.choice(list(range(i*args.clients_for_eachdomain,i*args.clients_for_eachdomain+args.clients_for_eachdomain)), 1, replace=False)[0] for i in range(domainnums)])
        print(this_round_clients)
        lr = args.learningrate#lrcos(step=r,lr=args.learningrate,lr_min=0.001,T_max=args.rounds)
        
        for idx in this_round_clients:
            clients[idx].load_para(server.get_para_baseline())  
            
        for idx in this_round_clients:
            clients[idx].train_baseline(lr, args.clientepoch,mmm,r,server.global_model)
        
        server.agg_baseline([client.get_para() if i in this_round_clients else {} for i, client in enumerate(clients)])
    
        print(f'round {r}')
        if r>args.rounds-5 or (r%20==0 and r>0):
            # torch.save(server.global_model.state_dict(),f'{args.modeltype}_{args.data}{args.net_type[0]}_{args.info}_{args.alpha}_{args.distribution}{args.clients_for_eachdomain}.pth')
            for i in range(len(test_data)):
                top1,_ = evaluation(server.global_model,test_data[i])
                logger.info('top1: %s' % str(top1))
                print(top1) 
            
if 'depthfl' in args.info:
    server = Server(A,num_layers=12,args=args,num_classes=num_classes,clayers = client_layers,depth_cls=depth_cls, modeltype = args.modeltype)
    for r in tqdm(range(args.rounds)):
        print(args.info)
        if args.distribution=='label':
            this_round_clients = list(range(args.clients_for_eachdomain))
        else:
            this_round_clients = sorted([np.random.choice(list(range(i*args.clients_for_eachdomain,i*args.clients_for_eachdomain+args.clients_for_eachdomain)), 1, replace=False)[0] for i in range(domainnums)])
        print(this_round_clients)
        lr = args.learningrate#lrcos(step=r,lr=args.learningrate,lr_min=0.001,T_max=args.rounds)
        for idx in this_round_clients:
            clients[idx].load_para(server.get_para_baseline())  
            
        for idx in this_round_clients:
            clients[idx].train_depthfl(lr, args.clientepoch, r,mmm,r,server.global_model)
        
        server.agg_baseline([client.get_para() if i in this_round_clients else {} for i, client in enumerate(clients)])
    
        print(f'round {r}')
        if r>args.rounds-5 or (r%20==0 and r>0):
            for i in range(len(test_data)):
                top1,_ = evaluation_depthfl(server.global_model,test_data[i])
                logger.info('top1: %s' % str(top1))
                print(top1)  
            

            
