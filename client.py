import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from peft import inject_adapter_in_model, LoraConfig, get_peft_model,get_peft_model_state_dict
import random
from utils import foundationmodel


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

class KLLoss(nn.Module):
    """KL divergence loss for self distillation."""

    def __init__(self):
        super().__init__()
        self.temperature = 1

    def forward(self, pred, label):
        """KL loss forward."""
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
            self.temperature
            * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss
        

        

class Client: # as a user
    def __init__(self, dataloader, num_layers=12, num_classes=10,aux = None,depth_cls=0,modeltype = 'ViT'):
        self.dataloader = dataloader
        if modeltype == 'ViT':
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=['proj','mlp.fc2'],#['mlp.fc2'],#['proj','mlp.fc2'],#["proj"],#mlp.0
                lora_dropout=0.1,
                bias="none",
            )
        elif modeltype == 'mixer':
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=['mlp_tokens.fc2','mlp_channels.fc2'],
                lora_dropout=0.1,
                bias="none",
            )
        self.local_model = foundationmodel(num_layers,num_classes,depth_cls,modeltype,lora_config).cuda()
        self.last_local_para = None

    def get_para_ori(self):
        return self.local_model.state_dict()
        
    def get_para(self):
        back = {}
        for k,v in self.local_model.named_parameters():
            if 'lora' in k or 'head' in k or 'Prompt' in k or 'norm' in k:#
                back[k] = v
        return back

    def load_para(self,para):
        self.local_model.load_state_dict(para,strict = False)
 

    def train_depthfl(self,lr,epochs,curr_round,mmm,round,globalmodel):
        for name, param in self.local_model.named_parameters():
            if 'head' in name or 'lora' in name or 'Prompt' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        fc_params = list(map(id, self.local_model.back.head.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params,
                             self.local_model.parameters())
        
        task_criterion = nn.CrossEntropyLoss().cuda()
        criterion_kl = KLLoss().cuda()
        optimizer =  torch.optim.SGD([{'params': base_params, 'lr': 1*lr},{'params': self.local_model.back.head.parameters(), 'lr': 1*lr},], lr=lr, momentum=0.9,weight_decay=1e-5)

        consistency_weight_constant = 500
        current = np.clip(curr_round, 0.0, consistency_weight_constant)
        phase = 1.0 - current / consistency_weight_constant
        consistency_weight = float(np.exp(-5.0 * phase * phase))
        
        for _ in (range(epochs)):
            for i, (image, label) in enumerate(tqdm(self.dataloader)):
                if i>mmm: break
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                outputs = self.local_model(image)
                loss = torch.zeros(1).cuda()
                if len(outputs)==1:
                    loss = task_criterion(outputs[0],label)
                else:
                    for o in range(len(outputs)):
                        loss += task_criterion(outputs[o], label)
                        for k in range(len(outputs)):
                            if o==k: continue
                            loss += (
                        consistency_weight
                        * criterion_kl(outputs[o], outputs[k].detach())
                        / (len(outputs) - 1)
                    )
                ## for feddyn
                # if self.last_local_para!=None: 
                #     reg_loss = 0.0
                #     cnt = 0.0
                #     for name, param in self.local_model.named_parameters():
                #         if param.requires_grad == True:
                #             term1 = (param * (
                #                 self.last_local_para[name].cuda() - globalmodel.state_dict()[name]
                #             )).sum()
                #             term2 = (param * param).sum()
    
                #             reg_loss += 0.1 * (term1 + term2)
                #             cnt += 1.0
                #     loss = loss + reg_loss / cnt
                loss.backward()
                nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()  
        self.last_local_para = self.local_model.state_dict()
                
    def train_baseline(self,lr,epochs,mmm,round,globalmodel):
        for name, param in self.local_model.named_parameters():
            if 'head' in name or 'lora' in name or 'Prompt' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        fc_params = list(map(id, self.local_model.back.head.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params,
                             self.local_model.parameters())
        
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer =  torch.optim.SGD([{'params': base_params, 'lr': 1*lr},{'params': self.local_model.back.head.parameters(), 'lr': 1*lr},], lr=lr, momentum=0.9,weight_decay=1e-5)

        for _ in (range(epochs)):
            for i, (image, label) in enumerate(tqdm(self.dataloader)):
                if i>mmm: break
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                output = self.local_model(image)
                loss = task_criterion(output,label)
                ## for feddyn
                # if self.last_local_para!=None: 
                #     reg_loss = 0.0
                #     cnt = 0.0
                #     for name, param in self.local_model.named_parameters():
                #         if param.requires_grad == True:
                #             term1 = (param * (
                #                 self.last_local_para[name].cuda() - globalmodel.state_dict()[name]
                #             )).sum()
                #             term2 = (param * param).sum()
    
                #             reg_loss += 0.1 * (term1 + term2)
                #             cnt += 1.0
                #     loss = loss + reg_loss / cnt
                loss.backward()
                nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
        self.last_local_para = self.local_model.state_dict()
                
