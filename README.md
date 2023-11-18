# FedRA
This repository is the implementation of "FedRA: A Random Allocation Strategy for Federated Tuning to Unleash the Power of Heterogeneous Clients". 

## Requirements
### Dependencies
```
Python 3.8.18
and requirements.txt
```
### Datasets
A `datapath` should be defined (such as `datapath='/home/share/DomainNet/'`). The directory structure should be
```
/home/share/DomainNet/
│       
└───clipart/
│   │...
└───infograph/
│   │...

/home/share/NICOpp/
│       
└───NICO_DG
│   └───autumn/
│   └───dim/
│   ...
└───NICO_DG_official
│   └───autumn_train.txt
│   ...
```
Download and unzip the [DomainNet](http://ai.bu.edu/M3SDA/) and [NICO++](https://nicochallenge.com/dataset) dataset to datapath.

## Training

Feature-skew:
```
ALLlarge: 
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 100  --distribution 'feature' --info 'samelayer12' --clients_for_eachdomain 1 --modeltype 'ViT'

ALLsmall:
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 100  --distribution 'feature' --info 'samelayer3' --clients_for_eachdomain 1 --modeltype 'ViT'

InclusiveFL:
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 100  --distribution 'feature' --info 'inclusivefl' --clients_for_eachdomain 1 --modeltype 'ViT'

DepthFL:
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 100  --distribution 'feature' --info 'depthfl' --clients_for_eachdomain 1 --modeltype 'ViT'

FedRA:
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 100  --distribution 'feature' --info 'ours' --clients_for_eachdomain 1 --modeltype 'ViT'
```

Feature&label-skew:
```
ALLlarge: 
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 0.5  --distribution 'feature&label' --info 'samelayer12' --clients_for_eachdomain 5 --modeltype 'ViT' --rounds 150

ALLsmall: 
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 0.5  --distribution 'feature&label' --info 'samelayer3' --clients_for_eachdomain 5 --modeltype 'ViT' --rounds 150

InclusiveFL: 
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 0.5  --distribution 'feature&label' --info 'inclusivefl' --clients_for_eachdomain 5 --modeltype 'ViT' --rounds 150

DepthFL: 
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 0.5  --distribution 'feature&label' --info 'depthfl' --clients_for_eachdomain 5 --modeltype 'ViT' --rounds 150

FedRA: 
python main.py --base-path "/home/share/DomainNet" --data 'domainnet' --alpha 0.5  --distribution 'feature&label' --info 'ours' --clients_for_eachdomain 5 --modeltype 'ViT' --rounds 150
```


