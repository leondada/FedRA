import timm
from .structure import *
from .structuremixer import *

def build_promptmodel(num_classes=2, edge_size=224, modeltype='ViT', patch_size=16,
                      Prompt_Token_num=10, VPT_type="Shallow", depth = 12,depth_cls=0):
    if modeltype == 'ViT':
        basic_model = timm.create_model('vit_base_patch16_224',pretrained=True,)

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type,depth=depth,depth_cls=depth_cls)
        model.load_state_dict(basic_model.state_dict(), strict = False)
        model.New_CLS_head(num_classes)
        model.Freeze()

    elif modeltype == 'mixer':
        basic_model = timm.create_model('mixer_b16_224',pretrained=True,)
        model = Mixer(embed_dim=768,num_blocks=depth,depth_cls=depth_cls)
        model.load_state_dict(basic_model.state_dict(), strict = False)
        model.reset_classifier(num_classes)
        
    else:
        print("The model is not difined now！！")
        return -1

    return model