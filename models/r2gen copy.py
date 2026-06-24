import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.encoder_decoder import EncoderDecoder
import math
from modules import lorentz

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, encoder_decoder=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.prompt_plip = nn.Parameter(torch.randn(1, 1, 512))
        self.fc = nn.Sequential(nn.LayerNorm(args.d_model),nn.Linear(args.d_model,args.d_model),nn.Linear(args.d_model,args.n_classes))
        if not encoder_decoder:
            print('use encoder_decoder: default')
            self.encoder_decoder = EncoderDecoder(args, tokenizer)
            
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')

        curv_init=20
        learn_curv=True
        self.patch_alpha = nn.Parameter(torch.tensor(args.d_vf**-0.5).log())
        # self.region_alpha = nn.Parameter(torch.tensor(args.d_vf**-0.5).log())
        # self.slide_alpha = nn.Parameter(torch.tensor(args.d_vf**-0.5).log())
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        
    def hyper_proj(self, x, alpha):
        x_hp = x * alpha.exp()
        with torch.autocast(x_hp.device.type, dtype=torch.float32):
            x_hp = lorentz.exp_map0(x_hp, self.curv.exp())
        return x_hp


    def cal_parameters(self):
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.parameters():

            mulValue = np.prod(param.size()) 
            Total_params += mulValue 
            if param.requires_grad:
                Trainable_params += mulValue 
            else:
                NonTrainable_params += mulValue

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_brca(self, images, targets=None, mode='train'):

        att_feats = images  # shape 1*N*384
    
        att_feats = torch.cat([torch.cat((self.prompt,self.prompt_plip),dim=2),att_feats],dim=1)
        fc_feats = torch.sum(att_feats[:,:,:-512],dim=1) #shape 1*384

        att_feats = self.hyper_proj(att_feats, self.patch_alpha)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, mode='encode')

            logits = self.fc(output[0,0,:]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError
        return output

