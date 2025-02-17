import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# from .embedder import get_embedder

class Embedder:
    """
    borrow from
    https://github.com/zju3dv/animatable_nerf/blob/master/lib/networks/embedder.py
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj


XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(data):
    # data in format of (N, 4, 3):
    mins = torch.as_tensor(
        XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
    divider = torch.as_tensor(
        XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
    data = (data - mins) / divider
    return data


class ContinuousLiDARWithTextEmbedding(nn.Module):
    """
    Use continuous bbox corrdicate and text embedding with CLIP encoder
    """

    def __init__(
        self,
        lidar_per_instance=100,
        embedder_num_freq=4,
        proj_dims=[1280, 1024, 1024, 768],
        minmax_normalize=True,
        use_text_encoder_init=True,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        input_dims = 3
        output_num = lidar_per_instance  # 8 points

        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0], proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))


    def prepare(self, cfg, **kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(
                kwargs['tokenizer'], kwargs['text_encoder'],
                cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")


    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def forward_feature(self, pos_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)
        emb = self.second_linear(emb)
        return emb

    def forward(self, lidars: torch.Tensor,
                masks=None, **kwargs):

        (B, N) = masks.shape[:2]
        lidars = rearrange(lidars, 'b n ... -> (b n) ...')
        if masks is None:
            masks = torch.ones(len(lidars))
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)

        # box
        if self.minmax_normalize:
            lidars = normalizer(lidars)
        
        lidar_emb = self.fourier_embedder(lidars)
        lidar_emb = lidar_emb.reshape(
            lidar_emb.shape[0], -1).type_as(self.null_pos_feature)

        lidar_emb = lidar_emb * masks + self.null_pos_feature[None] * (1 - masks)
        
        # combine
        emb = self.forward_feature(lidar_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        return emb


if __name__=="__main__":
    lidar_embedder_kwargs = dict()
    lidar_embedder_kwargs['lidars']= torch.randn(18, 45, 100, 3)
    lidar_embedder_kwargs['masks'] = torch.randn(18, 45)
    lidar_embedder_kwargs['masks'] = (lidar_embedder_kwargs['masks']>0).float()
    

    lidar_embedder_op =ContinuousLiDARWithTextEmbedding(lidar_per_instance=100,
                                    embedder_num_freq=4,
                                    proj_dims=[1280, 1024, 1024, 768],
                                    minmax_normalize=True,
                                    use_text_encoder_init=True)
    
    emb=lidar_embedder_op(**lidar_embedder_kwargs)
    print(emb.shape)