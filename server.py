import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from peft import inject_adapter_in_model, LoraConfig, get_peft_model,get_peft_model_state_dict
from utils import foundationmodel

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

class Server: 
    def __init__(self,A=None,num_layers=12,args=None,num_classes=10,clayers = None,depth_cls=0,modeltype = 'ViT'):
        
        if modeltype == 'ViT':
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=['proj','mlp.fc2'],#['mlp.fc2'],#["proj"],['mlp.fc2']#['proj','mlp.fc2']
                lora_dropout=0.1,
                bias="none",
            )
        elif modeltype == 'mixer':
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=['mlp_tokens.fc2','mlp_channels.fc2'],#mlp.0
                lora_dropout=0.1,
                bias="none",
            )
        self.global_model =foundationmodel(num_layers,num_classes,depth_cls,modeltype,lora_config).cuda()
        self.modeltype = modeltype

        self.num_layers=num_layers
        self.client_num = args.domains*args.clients_for_eachdomain
        self.cache_clients = [{k:v for k,v in self.global_model.state_dict().items() if 'lora' in k or 'head' in k or 'Prompt' in k} for i in range(self.client_num)]
        self.cache_clients_idx = [{k:False for k,v in self.global_model.state_dict().items() if 'lora' in k or 'head' in k or 'Prompt' in k} for _ in range(self.client_num)]
        self.client_layers = clayers
        self.exist = np.zeros((self.client_num,self.num_layers),dtype=int)
        self.mlast = {i:[{},0] for i in self.client_layers}
        self.lastA = A
        self.A = A
        
    def agg_ori(self,parameters):
        globalpara = self.get_para_ori()
        weights = [1/len(parameters)]*len(parameters)
        for key in parameters[0].keys():
            for c,para in enumerate(parameters):
                if c==0:
                    globalpara[key] = para[key]*weights[c]
                else:
                    globalpara[key] += para[key]*weights[c]
        self.global_model.load_state_dict(globalpara,strict= False)

        
    def agg_baseline(self,parameters):
        globalpara = {k:v*0 for k,v in self.global_model.state_dict().items() if 'lora' in k or 'head' in k or 'Prompt' in k or 'norm' in k}
        num = {k:0 for k in globalpara.keys()} 
        for key in globalpara.keys():
            for c,para in enumerate(parameters):
                tmp = para.get(key,None)
                if tmp != None:
                    globalpara[key] +=  tmp
                    num[key] += 1
        for key in globalpara.keys():
            if num[key]>0:
                globalpara[key] = globalpara[key]/num[key]
            else:
                globalpara[key] = self.global_model.state_dict()[key]
        self.global_model.load_state_dict(globalpara,strict= False)
        
    def get_para_baseline(self):
        back = {k:v for k,v in self.global_model.state_dict().items() if 'lora' in k or 'head' in k or 'Prompt' in k or 'norm' in k}
        return back

    def agg_range_fill(self,models,this_round_clients):
        indexs = copy.deepcopy(self.lastA)
        for i in range(1,len(self.lastA[0])):
            indexs[:,i]+=indexs[:,i-1]
        if self.modeltype=='ViT':
            average = {}
            average['back.base_model.model.Prompt_Tokens'] = sum([models[i]['back.base_model.model.Prompt_Tokens'].data for i in this_round_clients])/(len(this_round_clients))
            if 'back.base_model.model.patch_embed.proj.lora_A.default.weight' in models[0]:
                average['back.base_model.model.patch_embed.proj.lora_A.default.weight'] = sum([models[i]['back.base_model.model.patch_embed.proj.lora_A.default.weight'].data for i in this_round_clients])/(len(this_round_clients))
                average['back.base_model.model.patch_embed.proj.lora_B.default.weight'] = sum([models[i]['back.base_model.model.patch_embed.proj.lora_B.default.weight'].data for i in this_round_clients])/(len(this_round_clients))
    
            for i in range(12):
                ta,tb,tc,td,num = 0,0,0,0,0
                for j in this_round_clients:
                    if self.lastA[:,i][j]==1:
                        num += 1
                        ta+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.attn.proj.lora_A.default.weight'].data
                        tb+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.attn.proj.lora_B.default.weight'].data
                        tc+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.mlp.fc2.lora_A.default.weight'].data
                        td+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.mlp.fc2.lora_B.default.weight'].data
    
                if num>0:
                    average[f'back.base_model.model.blocks.{i}.attn.proj.lora_A.default.weight'] = ta/num
                    average[f'back.base_model.model.blocks.{i}.attn.proj.lora_B.default.weight'] = tb/num
                    average[f'back.base_model.model.blocks.{i}.mlp.fc2.lora_A.default.weight'] = tc/num
                    average[f'back.base_model.model.blocks.{i}.mlp.fc2.lora_B.default.weight'] = td/num
                else:
                    average[f'back.base_model.model.blocks.{i}.attn.proj.lora_A.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.attn.proj.lora_A.default.weight'].data
                    average[f'back.base_model.model.blocks.{i}.attn.proj.lora_B.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.attn.proj.lora_B.default.weight'].data
                    average[f'back.base_model.model.blocks.{i}.mlp.fc2.lora_A.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.mlp.fc2.lora_A.default.weight'].data
                    average[f'back.base_model.model.blocks.{i}.mlp.fc2.lora_A.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.mlp.fc2.lora_A.default.weight'].data
         
            average['back.base_model.model.norm.weight'] = sum([models[i]['back.base_model.model.norm.weight'].data for i in this_round_clients])/(len(this_round_clients))
            average['back.base_model.model.norm.bias'] = sum([models[i]['back.base_model.model.norm.bias'].data for i in this_round_clients])/(len(this_round_clients))
            average['back.base_model.model.head.weight'] = sum([models[i]['back.base_model.model.head.weight'].data for i in this_round_clients])/(len(this_round_clients))
            average['back.base_model.model.head.bias'] = sum([models[i]['back.base_model.model.head.bias'].data for i in this_round_clients])/(len(this_round_clients))
        
        elif self.modeltype=='mixer':
            average = {}
            for i in range(12):
                ta,tb,tc,td,num = 0,0,0,0,0
                for j in this_round_clients:
                    if self.lastA[:,i][j]==1:
                        num += 1   
                        ta+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.mlp_tokens.fc2.lora_A.default.weight'].data
                        tb+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.mlp_tokens.fc2.lora_B.default.weight'].data
                        tc+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.mlp_channels.fc2.lora_A.default.weight'].data
                        td+=models[j][f'back.base_model.model.blocks.{indexs[:,i][j]-1}.mlp_channels.fc2.lora_B.default.weight'].data
                if num>0:
                    average[f'back.base_model.model.blocks.{i}.mlp_tokens.fc2.lora_A.default.weight'] = ta/num
                    average[f'back.base_model.model.blocks.{i}.mlp_tokens.fc2.lora_B.default.weight'] = tb/num
                    average[f'back.base_model.model.blocks.{i}.mlp_channels.fc2.lora_A.default.weight'] = tc/num
                    average[f'back.base_model.model.blocks.{i}.mlp_channels.fc2.lora_B.default.weight'] = td/num
                else:
                    average[f'back.base_model.model.blocks.{i}.mlp_tokens.fc2.lora_A.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.mlp_tokens.fc2.lora_A.default.weight'].data
                    average[f'back.base_model.model.blocks.{i}.mlp_tokens.fc2.lora_B.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.mlp_tokens.fc2.lora_B.default.weight'].data
                    average[f'back.base_model.model.blocks.{i}.mlp_channels.fc2.lora_A.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.mlp_channels.fc2.lora_A.default.weight'].data
                    average[f'back.base_model.model.blocks.{i}.mlp_channels.fc2.lora_B.default.weight'] = self.global_model.state_dict()[f'back.base_model.model.blocks.{i}.mlp_channels.fc2.lora_B.default.weight'].data
         
            average['back.base_model.model.norm.weight'] = sum([models[i]['back.base_model.model.norm.weight'].data for i in this_round_clients])/(len(this_round_clients))
            average['back.base_model.model.norm.bias'] = sum([models[i]['back.base_model.model.norm.bias'].data for i in this_round_clients])/(len(this_round_clients))
            average['back.base_model.model.head.weight'] = sum([models[i]['back.base_model.model.head.weight'].data for i in this_round_clients])/(len(this_round_clients))
            average['back.base_model.model.head.bias'] = sum([models[i]['back.base_model.model.head.bias'].data for i in this_round_clients])/(len(this_round_clients))

        self.global_model.load_state_dict(copy.deepcopy(average),strict=False)
        
    def get_para_range(self,this_round_clients,r):
        self.lastA = np.zeros((self.client_num,self.num_layers),dtype=int)
        # if max(self.client_layers)<12:
        #     for i in range(self.client_num):
        #         indices_to_zero = np.argsort(self.exist[i])[:self.client_layers[i]]
        #         self.lastA[i][indices_to_zero] = 1
        #     self.lastA = np.zeros((self.client_num,self.num_layers),dtype=int)
        #     for i in this_round_clients:  
        #         row_indices = np.random.choice(list(range(12)), self.client_layers[i], replace=False)  
        #         self.lastA[i, row_indices] = 1  
        #     while np.any(np.sum(self.lastA,0)==0):
        #         now = np.sum(self.lastA,0)
        #         nll = np.where(now ==0)[0]
        #         i = np.random.choice(this_round_clients, 1, replace=False)[0]
        #         if len(nll)>self.client_layers[i]:
        #             self.lastA[i] = 0
        #             self.lastA[i, np.random.choice(nll, self.client_layers[i], replace=False)  ] = 1
        #         else:
        #             idx = list(nll) + list(np.random.choice(np.where(now >=1)[0], self.client_layers[i]-len(nll), replace=False)  )
        #             self.lastA[i] = 0
        #             self.lastA[i,idx ] = 1
        # else:
        for i in range(self.client_num):
            indices_to_zero = np.random.choice(list(range(0,12)), self.client_layers[i], replace=False)
            self.lastA[i][indices_to_zero] = 1
            assert sum(self.lastA[i]) == self.client_layers[i], 'error 172'
            
        print(self.lastA[this_round_clients])
        tp = [0 for _ in range(self.client_num)]
        modelgrads = [{} for _ in range(self.client_num)]
        if self.modeltype=='ViT':
            for i in this_round_clients:
                modelgrads[i]['back.base_model.model.Prompt_Tokens'] = self.global_model.state_dict()['back.base_model.model.Prompt_Tokens'].data
                modelgrads[i]['back.base_model.model.patch_embed.proj.lora_A.default.weight'] = self.global_model.state_dict()['back.base_model.model.patch_embed.proj.lora_A.default.weight'].data
                modelgrads[i]['back.base_model.model.patch_embed.proj.lora_B.default.weight'] = self.global_model.state_dict()['back.base_model.model.patch_embed.proj.lora_B.default.weight'].data
                for j in range(12):
                    if self.lastA[i][j]==1:
                        for k,v in self.global_model.back.base_model.model.blocks[j].state_dict().items():
                            modelgrads[i][f'back.base_model.model.blocks.{tp[i]}.'+k] = v.data
                        tp[i]+=1
                        
                modelgrads[i]['back.base_model.model.norm.weight'] = self.global_model.state_dict()['back.base_model.model.norm.weight'].data
                modelgrads[i]['back.base_model.model.norm.bias'] = self.global_model.state_dict()['back.base_model.model.norm.bias'].data
                modelgrads[i]['back.base_model.model.head.weight'] = self.global_model.state_dict()['back.base_model.model.head.weight'].data
                modelgrads[i]['back.base_model.model.head.bias'] = self.global_model.state_dict()['back.base_model.model.head.bias'].data
        elif self.modeltype=='mixer':
            for i in this_round_clients:
                for j in range(12):
                    if self.lastA[i][j]==1:
                        for k,v in self.global_model.back.base_model.model.blocks[j].state_dict().items():
                            modelgrads[i][f'back.base_model.model.blocks.{tp[i]}.'+k] = v.data
                        tp[i]+=1
                modelgrads[i]['back.base_model.model.norm.weight'] = self.global_model.state_dict()['back.base_model.model.norm.weight'].data
                modelgrads[i]['back.base_model.model.norm.bias'] = self.global_model.state_dict()['back.base_model.model.norm.bias'].data
                modelgrads[i]['back.base_model.model.head.weight'] = self.global_model.state_dict()['back.base_model.model.head.weight'].data
                modelgrads[i]['back.base_model.model.head.bias'] = self.global_model.state_dict()['back.base_model.model.head.bias'].data
        assert sum(tp) == sum([self.client_layers[i] for i in this_round_clients]), 'error 174'
        return copy.deepcopy(modelgrads)

    def agg_inclusiveFL(self,parameters):
        alltype = sorted(list(set(self.client_layers)))
        
        tmp = {}
        for i in range(len(parameters)):
            if not parameters[i]: continue
            for key in parameters[i]:
                parameters[i][key].data -= self.global_model.state_dict()[key].data
            if self.client_layers[i] not in tmp:
                tmp[self.client_layers[i]] = [copy.deepcopy(parameters[i]),1]
            else:
                tmp[self.client_layers[i]][1]+=1
                for key in parameters[i][0]:
                    tmp[self.client_layers[i]][0][key]+=parameters[i][key]
        for i in tmp:
            for key in tmp[i][0]:
                tmp[i][0][key] = tmp[i][0][key]/tmp[i][1]

        for i in range(len(alltype)-1):
            if self.mlast[alltype[0]][0] !={}:
                for key in tmp[alltype[i]][0]:
                    if f'back.base_model.model.blocks.{alltype[i]-1}' in key:
                        tmp[alltype[i]][0][key] *= 0.8
        for i in range(1,len(alltype)):
            for key in tmp[alltype[i]][0]:
                if 'back.base_model.model.blocks' in key and 'lora' in key and int(key.split('.')[4])>=alltype[i-1]:
                    if self.mlast[alltype[0]][0] !={}:
                        tmp[alltype[i-1]][0][f"back.base_model.model.blocks.{alltype[i-1]-1}.{'.'.join(key.split('.')[5:])}"] += 0.2*(self.mlast[alltype[i]][0][key]/(alltype[i]-alltype[i-1]))
                    
        for i in tmp:
            for key in tmp[i][0]:
                self.mlast[i][0][key] = tmp[i][0][key]

        
        globalpara = {k:v*0 for k,v in self.global_model.state_dict().items() if 'lora' in k or 'head' in k or 'Prompt' in k or 'norm' in k}
        num = {k:0 for k in globalpara.keys()} 
        for key in globalpara.keys():
            for i in tmp:
                ttt = tmp[i][0].get(key,None)
                if ttt != None:
                    globalpara[key] +=  ttt
                    num[key] += 1
        for key in globalpara.keys():
            globalpara[key] = globalpara[key]/num[key]+self.global_model.state_dict()[key].data

        self.global_model.load_state_dict(globalpara,strict= False)

       
     
