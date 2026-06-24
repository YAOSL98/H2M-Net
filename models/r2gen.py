import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.encoder_decoder import EncoderDecoder
import math
from modules import lorentz

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def build_regions_from_coords(coords, grid_size=2048, block=3):

    grid_x = coords[:, 0] // grid_size
    grid_y = coords[:, 1] // grid_size

    region_x = grid_x // block
    region_y = grid_y // block

    region_ids = region_x * 100000 + region_y
    return region_ids


def aggregate_patch_to_region(patch_feats, region_ids):
    """
    patch_feats: (N, D)
    region_ids: (N,)
    """
    unique_regions = torch.unique(region_ids)
    region_feats = []

    for rid in unique_regions:
        idx = region_ids == rid
        region_feat = patch_feats[idx].mean(dim=0)
        region_feats.append(region_feat)

    region_feats = torch.stack(region_feats, dim=0)  # (R, D)
    return region_feats


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, encoder_decoder=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.prompt_plip = nn.Parameter(torch.randn(1, 1, 512))
        self.prompt_region = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.prompt_plip_region = nn.Parameter(torch.randn(1, 1, 512))
        self.prompt_slide = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.prompt_plip_slide = nn.Parameter(torch.randn(1, 1, 512))
        self.fc = nn.Sequential(nn.LayerNorm(args.d_model),nn.Linear(args.d_model,args.d_model),nn.Linear(args.d_model,args.n_classes))
        if not encoder_decoder:
            print('use encoder_decoder: default')
            self.encoder_decoder = EncoderDecoder(args, tokenizer)
            self.encoder_decoder1 = EncoderDecoder(args, tokenizer)
            self.encoder_decoder2 = EncoderDecoder(args, tokenizer)
            
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')
        
        curv_init=20.0
        learn_curv=True
        self.patch_alpha = nn.Parameter(torch.tensor(args.d_vf**-0.5).log())
        self.region_alpha = nn.Parameter(torch.tensor(args.d_vf**-0.5).log())
        self.slide_alpha = nn.Parameter(torch.tensor(args.d_vf**-0.5).log())
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        self.curv1 = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        self.curv2 = nn.Parameter(
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

    def hyper_proj1(self, x, alpha):
        x_hp = x * alpha.exp()
        with torch.autocast(x_hp.device.type, dtype=torch.float32):
            x_hp = lorentz.exp_map0(x_hp, self.curv1.exp())
        return x_hp

    def hyper_proj2(self, x, alpha):
        x_hp = x * alpha.exp()
        with torch.autocast(x_hp.device.type, dtype=torch.float32):
            x_hp = lorentz.exp_map0(x_hp, self.curv2.exp())
        return x_hp
    
    def compute_entail_loss(self, x, y, curv, alpha=1.0):
        angle = lorentz.oxy_angle(x, y, curv)
        aperture = lorentz.half_aperture(x, curv)
        factor = torch.exp(torch.clamp(angle / aperture - 1, max=3))
        return factor * torch.clamp(angle - alpha * aperture, min=0)


    def compute_pos_loss(self, x, y, curv, margin=0.05):
        x_dis = lorentz.hyperbolic_distance_to_origin(x, curv)
        y_dis = lorentz.hyperbolic_distance_to_origin(y, curv)
        return torch.clamp(x_dis + margin - y_dis, min=0)


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

    def forward_brca(self, images, coords=None, targets=None, mode='train'):
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()


        att_feats = images  # shape 1*N*384
        att_feats = torch.cat([torch.cat((self.prompt,self.prompt_plip),dim=2),att_feats],dim=1)
        fc_feats = torch.sum(att_feats[:,:,:-512],dim=1) #shape 1*384
        att_feats = self.hyper_proj(att_feats, self.patch_alpha)


        patch_feats = images.squeeze(0)   # (N, D)
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.long)

        region_ids = build_regions_from_coords(coords, grid_size=2048, block=2)
        region_feats = aggregate_patch_to_region(patch_feats, region_ids)
        region_feats = region_feats.unsqueeze(0)  # (1, R, D)
        att_feats_region = torch.cat([torch.cat((self.prompt_region, self.prompt_plip_region), dim=2), region_feats], dim=1)  
        region_feats = torch.sum(att_feats_region[:,:,:-512],dim=1) #shape 1*384
        att_feats_region = self.hyper_proj1(att_feats_region, self.region_alpha)


        slide_ids = build_regions_from_coords(coords, grid_size=2048, block=4)
        slide_feats = aggregate_patch_to_region(patch_feats, slide_ids)
        slide_feats = slide_feats.unsqueeze(0)  # (1, R, D)
        att_feats_slide = torch.cat([torch.cat((self.prompt_slide, self.prompt_plip_slide), dim=2), slide_feats], dim=1)  
        slide_feats = torch.sum(att_feats_slide[:,:,:-512],dim=1) #shape 1*384
        att_feats_slide = self.hyper_proj2(att_feats_slide, self.slide_alpha)

        if mode == 'train':
            output = {}
            
            patch_region_ids = region_ids.to(att_feats_region.device)  # patch -> region
            patch_slide_ids = slide_ids.to(att_feats_region.device)    # patch -> slide

            # 先取唯一 region id
            unique_region_ids = torch.unique(patch_region_ids, sorted=True)
            unique_slide_ids = torch.unique(patch_slide_ids, sorted=True)

            slide_to_region_map = []

            for s_id in unique_slide_ids:
                patch_mask = patch_slide_ids == s_id 
                region_ids_in_slide = patch_region_ids[patch_mask]
                region_mask = torch.zeros(len(unique_region_ids), dtype=torch.bool, device=att_feats_region.device)


                for rid in torch.unique(region_ids_in_slide):
                    idx = (unique_region_ids == rid).nonzero(as_tuple=True)[0]
                    region_mask[idx] = True

                slide_to_region_map.append(region_mask)

            slide_to_region_map = torch.stack(slide_to_region_map, dim=0)  # (S, R)


            entail_sr_list = []
            for s_idx, region_mask in enumerate(slide_to_region_map):
                slide_feat = att_feats_slide[:, s_idx:s_idx+1, :]  # [1,1,D]
                region_feats_sel = att_feats_region[:, region_mask+1, :]  # [1,K,D], K <= R
                region_feat_mean = region_feats_sel.mean(dim=1, keepdim=True)
                entail_sr_list.append(self.compute_entail_loss(slide_feat, region_feat_mean, _curv))
            entail_sr = torch.stack(entail_sr_list).mean()


            pos_sr_list = []
            for s_idx, region_mask in enumerate(slide_to_region_map):
                slide_feat = att_feats_slide[:, s_idx:s_idx+1, :]  # [1,1,D]
                region_feats_sel = att_feats_region[:, region_mask+1, :]  # [1,K,D], K <= R
                region_feat_mean = region_feats_sel.mean(dim=1, keepdim=True)
                pos_sr_list.append(self.compute_pos_loss(slide_feat, region_feat_mean, _curv, margin=0.05))
            pos_sr = torch.stack(pos_sr_list).mean()

            unique_region_ids = torch.unique(patch_region_ids, sorted=True)
            region_to_patch_map = []

            for rid in unique_region_ids:
                patch_mask = patch_region_ids == rid
                region_to_patch_map.append(patch_mask)

            entail_rp_list = []
            for r_idx, patch_mask in enumerate(region_to_patch_map):
                region_feat = att_feats_region[:, r_idx:r_idx+1, :]  # [1,1,D]
                patch_feats_sel = att_feats[:, patch_mask+1, :]  # [1,K,D], K >= 1
                patch_feat_mean = patch_feats_sel.mean(dim=1, keepdim=True)  # [1,1,D]
                entail_rp_list.append(self.compute_entail_loss(region_feat, patch_feat_mean, _curv))

            entail_rp = torch.stack(entail_rp_list).mean()

            pos_rp_list = []
            for r_idx, patch_mask in enumerate(region_to_patch_map):
                region_feat = att_feats_region[:, r_idx:r_idx+1, :]
                patch_feats_sel = att_feats[:, patch_mask+1, :]
                patch_feat_mean = patch_feats_sel.mean(dim=1, keepdim=True)
                pos_rp_list.append(self.compute_pos_loss(region_feat, patch_feat_mean, _curv, margin=0.05))

            pos_rp = torch.stack(pos_rp_list).mean()

            global_embedding, output['global'] = self.encoder_decoder1(slide_feats, att_feats_slide, targets, None,  mode='forward')
            region_embedding, output['region'] = self.encoder_decoder2(region_feats, att_feats_region, targets, global_embedding, mode='forward')
            local_embedding, output['local'] = self.encoder_decoder(fc_feats, att_feats, targets, region_embedding, mode='forward')


            loss_dict = {}
            loss_dict['entail_vis'] = entail_sr + entail_rp
            loss_dict['pos_vis'] = pos_sr + pos_rp
            output['hyper_loss'] = loss_dict

            return output


        elif mode == 'sample':

            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output
        
        elif mode == 'encode':

            output = self.encoder_decoder(fc_feats, att_feats, mode='encode')
            logits = self.fc(output[0,0,:]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
            
        else:
            raise ValueError