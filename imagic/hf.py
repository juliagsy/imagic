from transformers import PretrainedConfig, PreTrainedModel


class ImagicConfig(PretrainedConfig):
    model_type = "imagic"

    def __init__(
        self,
        embed_dims=768,
        seq_len=64,
        **kwargs,
    ):
      self.embed_dims = embed_dims
      self.seq_len = seq_len
      super().__init__(**kwargs)


class ImagicModel(PreTrainedModel):
    config_class = ImagicConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Im2Mu(
            embed_dims=config.embed_dims,
            seq_len=config.seq_len,
        )

    def forward(self, img, wav):
        return self.model.forward(img, wav)

    def generate(self, img, wav=None, guidance_scale=3, device="cpu"):
        return self.model.generate(img, wav=wav, guidance_scale=guidance_scale, device=device)