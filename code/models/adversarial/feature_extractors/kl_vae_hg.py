from .base_encoder import BaseEncoder
from diffusers.models import AutoencoderKL


class VAEEmbedding(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_name)

    def forward(self, images):
        images = 2.0 * images - 1.0
        output = self.model.encode(images)
        z = output.latent_dist.mode()
        return z


# from .base_encoder import BaseEncoder
# from diffusers.models import AutoencoderKL
# import torch

# class VAEEmbedding(BaseEncoder):
#     def __init__(self, model_name):
#         super().__init__()
#         self.model = AutoencoderKL.from_pretrained(model_name)

#     def forward(self, images: torch.Tensor) -> torch.Tensor:
#         # scale [0,1] → [–1,1]
#         images = 2.0 * images - 1.0

#         # === RAW ENCODER CALLS (no no_grad) ===
#         # 1) conv encoder
#         h = self.model.encoder(images)
#         # 2) quantization conv (splits to mean/logvar)
#         h = self.model.quant_conv(h)
#         mean, logvar = h.chunk(2, dim=1)
#         z = mean
#         # 3) post-quant conv
#         z = self.model.post_quant_conv(z)
#         # 4) scale by the learned factor
#         z = z * self.model.config.scaling_factor

#         return z



