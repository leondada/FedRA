import torch
import torch.nn as nn

from timm.models.mlp_mixer import MlpMixer
class Mixer(MlpMixer):

    def __init__(
            self,
            num_classes=1000,
            num_blocks=8,
            embed_dim=768,
            depth_cls = 0,
    ):
        super().__init__(num_classes=num_classes,num_blocks=num_blocks,embed_dim=embed_dim)
        self.depth_cls = depth_cls
        self.num_classes = num_classes
        self.num_blocks = num_blocks


    def reset_classifier(self, num_classes, global_pool=None):

        if self.depth_cls>0:
            self.mid_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for i in range(int((self.num_blocks-1)//self.depth_cls))])
            
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        self.mid_out = []
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            if self.depth_cls > 0:
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x)
                    if i%self.depth_cls==0 and i>0:
                        y = self.norm(x)
                        y = y.mean(dim=1)
                        self.mid_out.append(self.mid_head[int((i-1)//self.depth_cls)](y))
            else:
                x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        if self.depth_cls>0:
            return self.mid_out+[x]
        else:
            return x