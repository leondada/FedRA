import torch
import math
import torch.nn as nn
from tqdm import tqdm
from models.GetModel import build_promptmodel
from peft import inject_adapter_in_model, LoraConfig, get_peft_model,get_peft_model_state_dict


class foundationmodel(nn.Module):
    def __init__(self,layer = 12,num_classes=10,depth_cls=0,modeltype = 'ViT',lora_config = None):
        super(foundationmodel, self).__init__()
        self.back = build_promptmodel(num_classes=num_classes, edge_size=224, modeltype=modeltype, patch_size=16,Prompt_Token_num=0, depth = layer,depth_cls=depth_cls)
        self.back = get_peft_model(self.back, lora_config)

    def forward(self, x ):
        x = self.back(x)
        return x
    
def lrcos(step=0,lr=0.01,lr_min=0.0001,T_max=500):
    return 0.5*(1 + math.cos(math.pi * step / T_max)) *(lr - lr_min) + lr_min


def evaluation_depthfl(model, testdata):
    model.eval()
    top1s,topks = [],[]
    if type(testdata)==list:
        for test_data in tqdm(testdata):
            with torch.no_grad():
                total = 0
                top1 = 0
                topk = 0
                for (test_imgs, test_labels) in test_data:
                    test_labels = test_labels.cuda()
                    outs = model(test_imgs.cuda())
                    out = outs[-1]
                    # out = torch.stack(outs, dim=2)
                    # out = torch.sum(out, dim=2) / len(outs)
                    _,maxk = torch.topk(out,5,dim=-1)
                    total += test_labels.size(0)
                    test_labels = test_labels.view(-1,1) 
                    top1 += (test_labels == maxk[:,0:1]).sum().item()
                    topk += (test_labels == maxk).sum().item()
            top1s.append(100*top1/total)
            topks.append(100*topk/total)
        return top1s,topks
    # sum(top1s)/len(top1s),sum(topks)/len(topks)
    else:
        with torch.no_grad():
            total = 0
            top1 = 0
            topk = 0
            for (test_imgs, test_labels) in testdata:
                test_labels = test_labels.cuda()
                outs = model(test_imgs.cuda())
                out = outs[-1]
                # out = torch.stack(outs, dim=2)
                # out = torch.sum(out, dim=2) / len(outs)
                _,maxk = torch.topk(out,5,dim=-1)
                total += test_labels.size(0)
                test_labels = test_labels.view(-1,1) 
                top1 += (test_labels == maxk[:,0:1]).sum().item()
                topk += (test_labels == maxk).sum().item()
                
        return 100 * top1 / total,100*topk/total
        
def evaluation(model, testdata):
    model.eval()
    top1s,topks = [],[]
    if type(testdata)==list:
        for test_data in tqdm(testdata):
            with torch.no_grad():
                total = 0
                top1 = 0
                topk = 0
                for (test_imgs, test_labels) in test_data:
                    test_labels = test_labels.cuda()
                    out = model(test_imgs.cuda())
                    _,maxk = torch.topk(out,5,dim=-1)
                    total += test_labels.size(0)
                    test_labels = test_labels.view(-1,1) 
                    top1 += (test_labels == maxk[:,0:1]).sum().item()
                    topk += (test_labels == maxk).sum().item()
            top1s.append(100*top1/total)
            topks.append(100*topk/total)
        return top1s,topks
    # sum(top1s)/len(top1s),sum(topks)/len(topks)
    else:
        with torch.no_grad():
            total = 0
            top1 = 0
            topk = 0
            for (test_imgs, test_labels) in testdata:
                test_labels = test_labels.cuda()
                out = model(test_imgs.cuda())
                _,maxk = torch.topk(out,5,dim=-1)
                total += test_labels.size(0)
                test_labels = test_labels.view(-1,1) 
                top1 += (test_labels == maxk[:,0:1]).sum().item()
                topk += (test_labels == maxk).sum().item()
                
        return 100 * top1 / total,100*topk/total