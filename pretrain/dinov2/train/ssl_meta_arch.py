# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn
import time
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model, pretrained_fsdp_wrapper

from dinov2.models.vision_transformer import BlockChunk
from dinov2.fms import CONCH, UNI, Phikon
from dinov2.kl_loss import get_kd_loss, FeatureLoss


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


def rip_backbone(ckpt):
    new_ckpt = {}
    for k, v in ckpt.items():
        
        k = '.'.join(k.split('.')[1:])
        new_ckpt[k] = v
    return new_ckpt


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()
        frozen_teachers_dict = dict()
        # easy to change code
        self.use_phikon_flag = True if cfg.distill.Phikon_loss_ratio > 0 else False
        self.use_uni_flag = True if cfg.distill.UNI_loss_ratio > 0 else False
        self.use_conch_flag = True if cfg.distill.CONCH_loss_ratio > 0 else False

        
        # frozen teachers
        if cfg.distill.Phikon_loss_ratio > 0:
            frozen_teachers_dict['phikon'] = Phikon()
        if cfg.distill.UNI_loss_ratio > 0:
            frozen_teachers_dict['uni'] = UNI()
        if cfg.distill.CONCH_loss_ratio > 0:
            frozen_teachers_dict['conch'] = CONCH()

        self.distill_loss_fn = get_kd_loss(cfg.distill.loss_name)
        self.distill_feature_loss_fn = FeatureLoss(dim=-1)
        

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            # chkpt = rip_backbone(chkpt['teacher'])
            # msg = student_backbone.load_state_dict(chkpt["teacher"], strict=False)
            msg = student_backbone.load_state_dict(chkpt, strict=False)
            logger.info(msg)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        # ---- distill heads-----
        distill_heads = {}
        logger.info("OPTIONS -- Distill heads")
        if self.use_phikon_flag:
            distill_heads['phikon'] = DINOHead( in_dim=embed_dim, out_dim=frozen_teachers_dict['phikon'].dim, 
                        hidden_dim=2048, bottleneck_dim=256, nlayers=1)
        if self.use_conch_flag:
            distill_heads['conch'] = DINOHead( in_dim=embed_dim, out_dim=frozen_teachers_dict['conch'].dim, 
                        hidden_dim=2048, bottleneck_dim=256, nlayers=1)
        if self.use_uni_flag:
            distill_heads['uni'] = DINOHead( in_dim=embed_dim, out_dim=frozen_teachers_dict['uni'].dim, 
                        hidden_dim=2048, bottleneck_dim=256, nlayers=1)
        self.distill_heads = nn.ModuleDict(distill_heads)
        
        #------------------------

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.distillation = nn.ModuleDict(frozen_teachers_dict)
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.distillation.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher, and FMs are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        _forward_start_time = time.time()
        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # fms teacher outpus
        @torch.no_grad()
        def get_distill_teacher_output():
            distill_output_dict = {}
            x = global_crops
            if self.use_uni_flag:
                uni_tokens = self.distillation.uni(x)
                distill_output_dict['uni'] = uni_tokens
            if self.use_phikon_flag:
                phikon_tokens = self.distillation.phikon(x)
                distill_output_dict['phikon'] = phikon_tokens
            if self.use_conch_flag:
                conch_tokens = self.distillation.conch(x)
                distill_output_dict['conch'] = conch_tokens

            return distill_output_dict

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        ## ---------Distillation process-------
        distill_teachers_output_dict = get_distill_teacher_output()
        # UNI
        # distillation process
        if self.cfg.distill.UNI_loss_ratio > 0:
            uni_cls_token = self.distill_heads.uni(student_global_cls_tokens.unsqueeze(1))
            distill_uni_loss = self.distill_loss_fn(uni_cls_token, distill_teachers_output_dict['uni']['cls_token'].detach().clone()) * self.cfg.distill.UNI_loss_ratio
            loss_dict['distill_uni_cls_loss'] = distill_uni_loss
            loss_accumulator += distill_uni_loss
        if self.cfg.distill.UNI_patch_ratio > 0:
            uni_patch_token = self.distill_heads.uni(student_global_backbone_output_dict['x_norm_patchtokens'])
            distill_uni_patch_loss = self.distill_feature_loss_fn(uni_patch_token, distill_teachers_output_dict['uni']['patch_token'].detach().clone()) * self.cfg.distill.UNI_patch_ratio
            loss_dict['distill_uni_patch_loss'] = distill_uni_patch_loss
            loss_accumulator += distill_uni_patch_loss

        # Phikon
        if self.cfg.distill.Phikon_loss_ratio > 0:
            phikon_cls_token = self.distill_heads.phikon(student_global_cls_tokens.unsqueeze(1)) 
            distill_phikon_loss = self.distill_loss_fn(phikon_cls_token, distill_teachers_output_dict['phikon']['cls_token'].detach().clone()) * self.cfg.distill.Phikon_loss_ratio
            loss_dict['distill_phikon_cls_loss'] = distill_phikon_loss
            loss_accumulator += distill_phikon_loss
        if self.cfg.distill.Phikon_patch_ratio > 0:
            phikon_patch_token = self.distill_heads.phikon(student_global_backbone_output_dict['x_norm_patchtokens'])
            distill_phikon_patch_loss = self.distill_feature_loss_fn(phikon_patch_token, distill_teachers_output_dict['phikon']['patch_token'].detach().clone()) * self.cfg.distill.Phikon_patch_ratio
            loss_dict['distill_phikon_patch_loss'] = distill_phikon_patch_loss
            loss_accumulator += distill_phikon_patch_loss
        
        # CONCH
        if self.cfg.distill.CONCH_loss_ratio > 0:
            conch_cls_token = self.distill_heads.conch(student_global_cls_tokens.unsqueeze(1)) 
            distill_conch_loss = self.distill_loss_fn(conch_cls_token, distill_teachers_output_dict['conch']['cls_token'].detach().clone()) * self.cfg.distill.CONCH_loss_ratio
            loss_dict['distill_conch_cls_loss'] = distill_conch_loss
            loss_accumulator += distill_conch_loss

        ## ---------------Distillation end---------


        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # logger.info('forward time:{}'.format(time.time() - _forward_start_time))
        # _start_time = time.time()
        self.backprop_loss(loss_accumulator)
        # logger.info('Backward time: {}'.format(time.time() - _start_time))
        self.fsdp_synchronize_streams()
        # logger.info('sync time:{}'.format(time.time() - _start_time))

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            # self.student.dino_head._streams = (
            #     self.teacher.dino_head._streams
            # ) = self.student.backbone._streams = self.teacher.backbone._streams
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.backbone, attr, stream)
                if self.use_phikon_flag:
                    setattr(self.distill_heads.phikon, attr, stream)
                if self.use_uni_flag:
                    setattr(self.distill_heads.uni, attr, stream)
                if self.use_conch_flag:
                    setattr(self.distill_heads.conch, attr, stream)
                
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()
        self.distillation.eval()
        

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
            
        # for distillation head
        for m in self.distill_heads.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        
        
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])

        # distillation heads
        for k, v in self.distill_heads.items():
            phikon_cfg = self.cfg.compute_precision.distill_heads[k]
            self.distill_heads[k] = get_fsdp_wrapper(phikon_cfg, modules_to_wrap=[BlockChunk])(self.distill_heads[k])
        
        # distillation techers
        for k, v in self.distillation.items():
            # self.distillation[k] = pretrained_fsdp_wrapper()(self.distillation[k])
            self.distillation[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.distillation[k])