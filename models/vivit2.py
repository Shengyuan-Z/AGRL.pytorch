# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv1d, Conv2d, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs



logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def resize_pos_embed(posemb, new_shape, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    
    gs_old = int(math.sqrt(posemb.shape[2]))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, new_shape, hight, width))
    posemb = posemb.reshape(posemb.shape[1], gs_old, gs_old, -1).permute(0,-1,-3,-2)
    posemb = F.interpolate(posemb, size=(hight, width), mode='bilinear')
    posemb = posemb.permute(0, -2, -1 , -3).reshape(1,posemb.shape[0], hight * width, -1)
    # posemb = posemb.reshape(1, hight * width, -1)
    return posemb

def np2th_vivit(weights, conv=False, times=0, dim=0, resize=False, new_shape=None, height=None, width=None):
    """Possibly convert HWIO to OIHW and add copied dimension"""
    tmp = weights if isinstance(weights, torch.Tensor) else torch.from_numpy(weights)
    if conv:
        tmp = tmp.permute([3, 2, 0, 1])
    if times:
        tmp = torch.stack([tmp for _ in range(times)], dim=dim)
    if resize: #only for position_embeddings
        # scale = tuple((torch.tensor(new_shape, dtype=float)/torch.tensor(tmp.shape, dtype=float)).tolist())
        # tmp = F.interpolate(tmp, size=scale, mode='bilinear')
        tmp = resize_pos_embed(tmp,new_shape,height,width)
    return tmp



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.001)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, seq_len=8, in_channels=3, method='sptial'):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        self.method = method

        if method == 'sptial':
            if config.patches.get("grid") is not None:
                grid_size = config.patches["grid"]
                patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
                n_patches = (img_size[0] // 16) * (img_size[1] // 16)
                self.hybrid = True
            else:
                patch_size = _pair(config.patches["size"])
                n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
                self.hybrid = False

            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=patch_size)
            
        elif method == 'temporal':
            patch_size = config.hidden_size
            n_patches = seq_len
            self.patch_embeddings = nn.Identity() # Conv1d
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.method == 'sptial':
            x = self.patch_embeddings(x)
            x = x.flatten(2)
            x = x.transpose(-1, -2)
            x = torch.cat((cls_tokens, x), dim=1)
            embeddings = x + self.position_embeddings
        elif self.method == 'temporal': # TODO: shape
            x = self.patch_embeddings(x)
            x = x.flatten(2)
            # x = x.transpose(-1, -2)
            x = torch.cat((cls_tokens, x), dim=1)
            embeddings = x + self.position_embeddings

        return self.dropout(embeddings)


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis=False, method='spatial'):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        num_layers = config.transformer["num_layers"] if method=='spatial' else config.transformer["tmp_num_layers"]
        for _ in range(num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Spatial_Transformer(nn.Module):
    def __init__(self, config, img_size, seq_len, vis=False):
        super(Spatial_Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size,seq_len=seq_len)
        self.encoder = Encoder(config, vis)

    def forward(self, x): # [bs, seq_len, c, h, w]
        x = self.embeddings(x)
        x, attn_weights = self.encoder(x)
        return x, attn_weights

class Temporal_Transformer(nn.Module):
    def __init__(self, config, img_size, seq_len, vis=False):
        super(Temporal_Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size,seq_len=seq_len,method='temporal')
        self.encoder = Encoder(config, vis, method='temporal')

    def forward(self, x):
        x = self.embeddings(x)
        x, attn_weights = self.encoder(x)
        return x, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size, seq_len, num_classes, zero_head=False, vis=False):
        # vis for if weights are returned
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.spa_transformer = Spatial_Transformer(config, img_size, seq_len, vis)
        self.tmp_transformer = Temporal_Transformer(config, img_size, seq_len, vis)
        
        self.bottleneck = nn.BatchNorm1d(config.hidden_size)
        self.bottleneck.bias.requires_grad_(False)
        self.head = Linear(config.hidden_size, num_classes,bias=False)

        self.t = config.t # t frames per tublet
        self.img_size = img_size
        self.config = config
        self.seq_len = seq_len

    def forward(self, x, labels=None): 
        bs, seq, c, h, w = x.shape

        x = x.reshape(bs*seq, c, h, w)
        x, _ = self.spa_transformer(x) #[bs*seq, #C, feature_size]
        x = x[:,0,:] #[bs*seq, 1, feature_size]
        x = x.reshape(bs,seq,-1) #[bs, seq, feature_size]

        x, _ = self.tmp_transformer(x) 
        x = x[:,0,:]
        x = x.reshape(bs,-1)
        features = self.bottleneck(x)

        if self.training:
            cls = self.head(features) 
            return cls, features
        else:
            return features

    def load_from(self, weights):
        with torch.no_grad():
            weights_init_classifier(self.head)
            weights_init_kaiming(self.bottleneck)
            # if self.zero_head: # needed
            #     nn.init.zeros_(self.head.weight)
            #     # nn.init.zeros_(self.head.bias)
            # else:
            #     self.head.weight.copy_(np2th(weights["head/kernel"]).t())
            #     # self.head.bias.copy_(np2th(weights["head/bias"]).t())c

            ### spatial
            self.spa_transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.spa_transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.spa_transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.spa_transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.spa_transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.spa_transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.spa_transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.spa_transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.spa_transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname) 

            ### temporal
            # patch_embeddings is identity, so no weights
            self.tmp_transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.tmp_transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.tmp_transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # reshape by linear method
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.tmp_transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.tmp_transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[:, 1:] # [1, 1, 768], [1,196, 768]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[:]

                posemb_grid = posemb_grid.transpose(-1,-2)
                posemb_grid = F.interpolate(posemb_grid, size=(ntok_new), mode='linear')
                posemb_grid = posemb_grid.transpose(-1,-2)

                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.tmp_transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            

            for bname, block in self.tmp_transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    # call class Block's load_from
                    # the names of components don't matter
                    unit.load_from(weights, n_block=uname) 



CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}


if __name__=='__main__':
    import configs as configs
    from .modeling_resnet import ResNetV2 
    
    samples = torch.rand((5,8,3,224,224))
    model = VisionTransformer(CONFIGS['ViT-B_16'], (224,224), seq_len=8, zero_head=True, num_classes=1000)
    model.load_from(np.load('/home/mygit/AGRL.pytorch/vit_checkpoint/ViT-B_16.npz'))
    cls, features = model(samples)
    print(f"cls.shape: {cls.shape} \n feat.shape: {features.shape}")