import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .indexed_latticeformer_encoder import IndexedLatticeformerEncoder, CrystalformerEncoderCUDA
from . import pooling
from .latticeformer_params import LatticeformerParams

import copy


class GradientScaler(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, onehot, sizes, scale):
    ctx.save_for_backward(onehot, sizes)
    ctx.scale = scale
    return x
  @staticmethod
  def backward(ctx, g):
    (onehot, sizes) = ctx.saved_tensors
    avr = pooling.avr_pool(onehot, sizes).mean(axis=0)
    w = onehot @ avr
    m = w>0
    w = w[m]
    w = (1/w)
    w /= w.mean()
    w.clamp_(max=ctx.scale)
    g[m] *= w[:,None]
    return g, None, None, None

class Latticeformer(torch.nn.Module):
    """
    Latticeformer: str
    
    """
    def __init__(self, params):
        super().__init__()
        embedding_dim = copy.deepcopy(params.embedding_dim)
        self.params = params
        num_layers = getattr(params, 'num_layers', 4)
        model_dim = getattr(params, 'model_dim', 128)
        ff_dim = getattr(params, 'ff_dim', 512)
        t_fixup_init = getattr(params, 't_fixup_init', True)
        #lattice_range = getattr(params, 'lattice_range', 2)
        #adaptive_cutoff_sigma = getattr(params, 'adaptive_cutoff_sigma', 0.0)
        #scale_crystal = getattr(params, 'scale_crystal', [2.5])
        #scale_crystal_reci = getattr(params, 'scale_crystal_reci', [0.0])
        #domain = getattr(params, 'domain', 'real')
        #gauss_lb_real = getattr(params, 'gauss_lb_real', 0)
        #gauss_lb_reci = getattr(params, 'gauss_lb_reci', 0)
        # value_pe_dist_real = getattr(params, 'value_pe_dist_real', False)
        # value_pe_dist_reci = getattr(params, 'value_pe_dist_reci', False)
        # value_pe_wave_real = getattr(params, 'value_pe_wave_real', False)
        # value_pe_wave_reci = getattr(params, 'value_pe_wave_reci', False)
        exclude_self = getattr(params, 'exclude_self', False)
        self.pooling = getattr(params, 'pooling', "max")
        pre_pooling_op = getattr(params, 'pre_pooling_op', "w+bn+relu")
        dropout = getattr(params, 'dropout', 0.1)
        #normalize_gauss = getattr(params, 'normalize_gauss', False)
        #positive_func = getattr(params, 'positive_func', 'abs')
        #value_pe_headed = getattr(params, 'value_pe_headed', True)
        head_num = getattr(params, 'head_num', 8)
        v_dim = getattr(params, 'v_dim', 0)
        k_dim = getattr(params, 'k_dim', 0)
        #norm_func_mode = getattr(params, 'norm_func_mode', 0)
        norm_type = getattr(params, 'norm_type', "bn")
        #value_pe_dist_max = getattr(params, 'value_pe_dist_max', 10.0)
        #value_pe_width_scale = getattr(params, 'value_pe_width_scale', 1.0)
        #gauss_state = getattr(params, 'gauss_state', "q")
        #value_pe_condproj = getattr(params, 'value_pe_condproj', "no")
        scale_grad = getattr(params, 'scale_grad', 0.0)
        use_cuda_code = getattr(params, 'use_cuda_code', True)
        print('Latticeformer params----')
        print('dropout:', dropout)
        # print('domain:', domain)
        print('head_num:', head_num)
        print('v_dim:', v_dim)
        print('k_dim:', k_dim)
        # print('lattice_range:', lattice_range)
        # print('adaptive_cutoff_sigma:', adaptive_cutoff_sigma)
        # print('scale_crystal:', scale_crystal)
        # print('scale_crystal_reci:', scale_crystal_reci)
        # print('gauss_lb_real:', gauss_lb_real)
        # print('gauss_lb_reci:', gauss_lb_reci)
        # print('value_pe_dist_real:', value_pe_dist_real)
        # print('value_pe_dist_reci:', value_pe_dist_reci)
        # print('value_pe_wave_real:', value_pe_wave_real)
        # print('value_pe_wave_reci:', value_pe_wave_reci)
        # print('value_pe_condproj:', value_pe_condproj)
        # print('value_pe_dist_max:', value_pe_width_scale)
        # print('value_pe_width_scale:', value_pe_width_scale)
        print('exclude_self:', exclude_self)
        print('pooling:', self.pooling)
        print('pre_pooling_op:', pre_pooling_op)
        # print('normalize_gauss:', normalize_gauss)
        # print('positive_func:', positive_func)
        # print('value_pe_headed:', value_pe_headed)
        # print('norm_func_mode:', norm_func_mode)
        # print('gauss_state:', gauss_state)
        print('scale_grad:', scale_grad)
        print('use_cuda_code:', use_cuda_code)
        self.scale_grad = scale_grad
        t_activation = getattr(params, 't_activation', 'relu')
        print(t_activation)

        lattice_params = LatticeformerParams()
        lattice_params.parseFromArgs(params)

        #     domain=domain,
        #     lattice_range=lattice_range,
        #     adaptive_cutoff_sigma=adaptive_cutoff_sigma,
        #     gauss_lb_real=gauss_lb_real,
        #     gauss_lb_reci=gauss_lb_reci,
        #     scale_real=scale_crystal,
        #     scale_reci=scale_crystal_reci,
        #     normalize_gauss=normalize_gauss,
        #     value_pe_dist_real = value_pe_dist_real,
        #     value_pe_wave_real = value_pe_wave_real,
        #     value_pe_dist_reci = value_pe_dist_reci,
        #     value_pe_wave_reci = value_pe_wave_reci,
        #     value_pe_headed=value_pe_headed,
        #     value_pe_condproj=value_pe_condproj,
        #     positive_func=positive_func,
        #     exclude_self=exclude_self,
        #     norm_func_mode=norm_func_mode,
        #     value_pe_dist_max=value_pe_dist_max,
        #     value_pe_width_scale=value_pe_width_scale,
        #     gauss_state=gauss_state,
        # )

        self.ATOM_FEAT_DIM = 98
        self.input_embeddings = nn.Linear(self.ATOM_FEAT_DIM, model_dim, bias=False)
        emb_scale = model_dim**(-0.5)
        if t_fixup_init:
            emb_scale *= (9*num_layers)**(-1/4)
        nn.init.normal_(self.input_embeddings.weight, mean=0, std=emb_scale)

        from .indexed_lattice_multi_head_attention_cuda import CUPY_AVAILABLE
        
        if use_cuda_code and not CUPY_AVAILABLE:
            print("Please install cupy and pytorch-pfn-extras to use the CUDA implementation.")
        Encoder = CrystalformerEncoderCUDA if use_cuda_code and CUPY_AVAILABLE else IndexedLatticeformerEncoder
        self.encoder = Encoder(
            model_dim, 
            head_num,
            activation=t_activation,
            num_encoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            t_fixup_init=t_fixup_init,
            no_layer_norm=t_fixup_init,
            lattice_params=lattice_params,
            k_dim=k_dim,
            v_dim=v_dim)

        dim_pooled = model_dim
        if norm_type == "bn":
            norm_type = nn.BatchNorm1d
        elif norm_type == "ln":
            norm_type = nn.LayerNorm
        elif norm_type == "in":
            norm_type = nn.InstanceNorm1d
        elif norm_type in ["id", "no"]:
            norm_type = nn.Identity
        else:
            raise NotImplementedError(f"norm_type: {norm_type}")

        self.proj_before_pooling = lambda x: x
        if pre_pooling_op == "w+bn+relu":
            dim_pooled = embedding_dim.pop(0)
            self.proj_before_pooling = nn.Sequential(
                nn.Linear(model_dim, dim_pooled),
                norm_type(dim_pooled),
                nn.ReLU(True)
            )
        elif pre_pooling_op == "w+relu":
            dim_pooled = embedding_dim.pop(0)
            self.proj_before_pooling = nn.Sequential(
                nn.Linear(model_dim, dim_pooled),
                nn.ReLU(True)
            )
        elif pre_pooling_op == "relu":
            self.proj_before_pooling = nn.ReLU(True)
        elif pre_pooling_op == "no":
            pass
        else:
            raise NotImplementedError(f"pre_pooling_op: {pre_pooling_op}")

        if self.pooling == "max":
            self.pooling_layer = pooling.max_pool
        elif self.pooling == "avr":
            self.pooling_layer = pooling.avr_pool
        else:
            raise NotImplementedError(f"pooling: {pooling}")

        # regression mode when targets = "hoge" or ["hoge", "foo"]
        final_dim = 1 if isinstance(params.targets, str) \
            else len(params.targets)
        
        in_dim = [dim_pooled] + embedding_dim[:-1]
        out_dim = embedding_dim
        layers = []
        for di, do in zip(in_dim, out_dim):
            layers.append(nn.Linear(di, do))
            layers.append(norm_type(do))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(out_dim[-1], final_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        trans = data.trans_vec
        sizes = data.sizes
        onehot_x = x

        if self.params.use_cgcnn_feat:
            if x.device != self.atom_feat.device:
                self.atom_feat = self.atom_feat.to(x.device)
            # Matrix multiplication: (N, 98) x (98, C) = (N, C)
            x = x @ self.atom_feat
        x = self.input_embeddings(x)
        if self.scale_grad > 0:
            x = GradientScaler().apply(x, onehot_x, sizes, self.scale_grad)

        batch_size = sizes.shape[0]
        device_count = torch.cuda.device_count()
        if not getattr(self.params, 'ddp', False) and device_count > 1 and x.is_cuda and batch_size > 1:
            x0 = None
            if self.encoder.layers[0].self_attn.gauss_scale[0] == 0:
                # forward once to initialize the internal parameters
                with torch.no_grad():
                    x0 = self.encoder(x, pos, batch, trans, sizes, onehot_x)
            
            n = min(device_count, batch_size)
            device_ids = [f'cuda:{id}' for id in range(n)]
            replicas = nn.parallel.replicate(self.encoder, device_ids)

            # split the batched data so that each split has
            # approximately the same sum-of-squared-system-sizes.
            if True:
                size2 = (sizes*sizes).cpu().numpy()
                sort_inds = np.argsort(-size2)
                total_sizes = np.zeros(n, np.int64)
                total_sizes[0] = 1
                item_lists = [[] for _ in range(n)]
                for item in sort_inds:
                    dev_id = np.argmin(total_sizes)
                    total_sizes[dev_id] += size2[item]
                    item_lists[dev_id].append(item)
                
                split_sizes = sizes.tolist()
                x_ = torch.split_with_sizes(x, split_sizes)
                p_ = torch.split_with_sizes(pos, split_sizes)
                h_ = torch.split_with_sizes(onehot_x, split_sizes)
                
                inputs = []
                for dev in range(n):
                    sz = sizes[item_lists[dev]].to(device_ids[dev])
                    bt = sum([[i]*s for i,s in enumerate(sz.tolist())], [])
                    inputs.append((
                        torch.cat([x_[i] for i in item_lists[dev]]).to(device_ids[dev]),
                        torch.cat([p_[i] for i in item_lists[dev]]).to(device_ids[dev]),
                        torch.tensor(bt, dtype=torch.long, device=device_ids[dev]),
                        trans[item_lists[dev]].to(device_ids[dev]),
                        sz,
                        torch.cat([h_[i] for i in item_lists[dev]]).to(device_ids[dev]),
                    ))
                    
                del x_, p_
                master_device = data.x.device
                x = nn.parallel.parallel_apply(replicas, inputs)
                sizes_ = [inputs[dev][-2]for dev in range(n)]
                sizes_ = nn.parallel.gather(sizes_, master_device)
                del inputs

                x = nn.parallel.gather(x, master_device)
                x = torch.split_with_sizes(x, sizes_.tolist())
                t = [None]*len(x)
                for src, des in enumerate(sum(item_lists, [])):
                    t[des] = x[src]
                x = torch.cat(t)
                if x.device != master_device:
                    x = x.to(master_device)

            if x0 is not None:
                print(f"parallel mismatch: {abs(x0-x).detach().max().item()}")
        else:
            x = self.encoder(x, pos, batch, trans, sizes)
        # x: (total_point_num, d_model)

        x = self.proj_before_pooling(x)
        if self.pooling.startswith("pma"):
            x = self.pooling_layer(x, batch, sizes.shape[0])
        else:
            x = self.pooling_layer(x, batch, sizes)

        output_cry = self.mlp(x)
        
        return output_cry

