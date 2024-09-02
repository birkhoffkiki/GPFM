"""
reference: https://github.com/hrzhang1123/DTFD-MIL/blob/main/Main_DTFD_MIL.py
Implementation of:
https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.pdf
Time: 2023.12.12
@author: MA JIABO
"""

import torch
import random
from .Attention import Attention_Gated as Attention
from .Attention import Attention_with_Classifier
from .network import Classifier_1fc, DimReduction
from ..mil import BaseMILModel
from ..loss import get_loss_function
import numpy as np


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


class DTFD_Model(BaseMILModel):
    def __init__(self, feature_dim, num_cls, droprate=0.0, droprate_2=0.0, 
                 mDim=512, distill='AFS', task='subtyping') -> None:
        """_summary_

        Args:
            feature_dim (_type_): _description_
            num_cls (_type_): _description_
            droprate (float, optional): _description_. Defaults to 0.0.
            droprate_2 (float, optional): _description_. Defaults to 0.0.
            mDim (int, optional): _description_. Defaults to 512.
            distill (str, optional): _description_. Defaults to 'AFS'.
        """
        super().__init__(task=task)
        self.task_type = task
        self.classifier = Classifier_1fc(mDim, num_cls, droprate)
        self.attention = Attention(mDim)
        self.dimReduction = DimReduction(feature_dim, mDim, numLayer_Res=0) # default value
        self.attCls = Attention_with_Classifier(L=mDim, num_cls=num_cls, droprate=droprate_2)
        self.loss_function = self.create_loss_function()        
        
        # default parameters defined by the authors
        self.numGroup_test = 4
        self.numGroup = 4
        self.total_instance = 4
        self.total_instance_test = 4
        self.distill = distill
        self.grad_clipping = 5
        self.num_MeanInference = 1
        
    def create_loss_function(self):
        if self.task_type == 'subtyping':
            return get_loss_function(self.task_type, 'CrossEntropy')
        elif self.task_type == 'survival':
            return get_loss_function(self.task_type, 'nll_surv')
        else:
            raise NotImplementedError    
        
    def set_up(self, lr, max_epochs, weight_decay, **kwargs):
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())

        self.optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=lr,  weight_decay=weight_decay)
        self.optimizer_adam1 = torch.optim.Adam(self.attCls.parameters(), lr=lr,  weight_decay=weight_decay)

        self.scheduler0 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_adam0, [max_epochs//2], gamma=0.2)
        self.scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_adam1, [max_epochs//2], gamma=0.2)

    
    def forward(self, wsi_feat):
        """give the predicted result of a wsi, 

        Args:
            wsi_feat (torch.Tensor): N*dim
            task_type: supported tasks
        """
        feat_index = list(range(wsi_feat.shape[0]))
        random.shuffle(feat_index)
        
        index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_pseudo_feat = []
        # this is logits
        slide_sub_preds = []
        
        instance_per_group = self.total_instance // self.numGroup
        
        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(wsi_feat, dim=0, index=torch.LongTensor(tindex).to(wsi_feat.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)


            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif self.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
        
        wsi_logits = self.attCls(slide_pseudo_feat)
        slide_sub_logits = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        
        # for survival
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(wsi_logits)
        sub_logits, sub_prob, sub_label = self.task_adapter(slide_sub_logits)
        
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
            'sub_logits': sub_logits,
            'sub_prob': sub_prob,
            'sub_label': sub_label,
        }
        return outputs
    

    def one_step(self, data, label, **kwargs):
        outputs = self.forward(data)

        ## optimization for the first tier
        sub_label = torch.cat([label for _ in range(self.numGroup)], dim=0)
        slide_sub_preds = outputs['sub_logits']
        slide_sub_prob = outputs['sub_prob']
        if self.task_type == 'subtyping':
            loss0 = self.loss_function(slide_sub_preds, sub_label)
        elif self.task_type == 'survival':
            slide_sub_c = torch.full(sub_label.shape, kwargs['c']).to(kwargs['c'].device)
            loss0 = self.loss_function(slide_sub_preds, slide_sub_prob, sub_label, slide_sub_c)
        self.optimizer_adam0.zero_grad()
        loss0.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clipping)
        self.optimizer_adam0.step()

        ## optimization for the second tier
        gSlidePred = outputs['wsi_logits']
        if self.task_type == 'subtyping':
            loss1 = self.loss_function(gSlidePred, label)
        elif self.task_type == 'survival':
            loss1 = self.loss_function(gSlidePred, outputs['wsi_prob'], label, kwargs['c'])
        self.optimizer_adam1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.attCls.parameters(), self.grad_clipping)
        self.optimizer_adam1.step()
        
        result = {
            'loss': loss0 + loss1,
            'tier_0_loss': loss0,
            'tier_1_loss': loss1,
            'call_scheduler': self._scheduler_callback,
            'wsi_logits': outputs['wsi_logits'],
            'wsi_prob': outputs['wsi_prob'],
            'wsi_label': outputs['wsi_label'],
        }
        return result
        
    def _scheduler_callback(self):
        self.scheduler0.step()
        self.scheduler1.step()
        print('Change lr0: {}'.format(self.scheduler0.get_lr()))
        print('Change lr1: {}'.format(self.scheduler1.get_lr()))
        

    def wsi_predict(self, data, **args):
        gPred_0 = torch.FloatTensor().to(data.device)
        gPred_1 = torch.FloatTensor().to(data.device)

        instance_per_group = self.total_instance // self.numGroup

        midFeat = self.dimReduction(data)
        AA = self.attention(midFeat, isNorm=False).squeeze(0)  ## N

        allSlide_pred_softmax = []
        for _ in range(self.num_MeanInference):
            feat_index = list(range(data.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]
            slide_d_feat = []
            slide_sub_preds = []
            for tindex in index_chunk_list:
                idx_tensor = torch.LongTensor(tindex).to(data.device)
                tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)
                tAA = AA.index_select(dim=0, index=idx_tensor)
                tAA = torch.softmax(tAA, dim=0)
        
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                if self.distill == 'MaxMinS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx_min = sort_idx[-instance_per_group:].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    slide_d_feat.append(d_inst_feat)
                elif self.distill == 'MaxS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx = topk_idx_max
                    d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    slide_d_feat.append(d_inst_feat)
                elif self.distill == 'AFS':
                    slide_d_feat.append(tattFeat_tensor)

            slide_d_feat = torch.cat(slide_d_feat, dim=0)
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)

            gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
            gSlidePred = self.attCls(slide_d_feat)
            
            allSlide_pred_softmax.append(gSlidePred)

        allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
        allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0)
        gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)

        gPred_1 = gPred_1[None, :]
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(gPred_1)
        sub_logits, sub_prob, sub_label = self.task_adapter(gPred_0)
        
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
            'sub_logits': sub_logits,
            'sub_prob': sub_prob,
            'sub_label': sub_label,
        }
        return outputs

