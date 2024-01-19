import numpy as np
import torch
import torch.nn as nn
from transformers import MusicgenForConditionalGeneration, AutoModel


class Im2Mu(nn.Module):
  def __init__(self, embed_dims=768, seq_len=64):
    super(Im2Mu, self).__init__()

    self.musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    self.muvis = AutoModel.from_pretrained("juliagsy/muvis", trust_remote_code=True).model.vit

    self.loss_ce = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
    self.img_lin = nn.Linear(197, 256)


  def shift_right(self, input_ids):
    shifted_input_ids = torch.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = 0
    return shifted_input_ids


  def forward(self, img, wav):
    img_e = self.muvis(**img)["last_hidden_state"]
    img_embeds = self.musicgen.get_encoder()(
        inputs_embeds=img_e
    )["last_hidden_state"]
    img_embeds = img_embeds.permute(0, 2, 1)
    img_embeds = self.img_lin(img_embeds)
    img_embeds = img_embeds.permute(0, 2, 1)

    wav_tokens = self.musicgen.get_audio_encoder().encode(
        **wav,
    )["audio_codes"]
    wav_size = wav_tokens.size()
    wav_tokens = wav_tokens.view((wav_size[1] * wav_size[2], wav_size[-1]))
    wav_tokens = self.shift_right(wav_tokens)

    ret = self.musicgen(
        decoder_input_ids=wav_tokens,
        encoder_outputs=(img_embeds,),
    )
    loss = self.loss_ce(ret.logits.view(-1, self.musicgen.config.audio_encoder.codebook_size), wav_tokens.view(-1))
    return loss


  def generate(self, img, wav=None, guidance_scale=3, max_new_tokens=256, device="cpu"):
    img_embeds = self.muvis(**img)["last_hidden_state"]
    img_embeds = img_embeds.permute(0, 2, 1)
    img_embeds = self.img_lin(img_embeds)
    img_embeds = img_embeds.permute(0, 2, 1)

    img_embeds = self.musicgen.get_encoder()(
        inputs_embeds=img_embeds
    )["last_hidden_state"]

    if wav is not None:
      input_ids = self.musicgen.get_audio_encoder().encode(
        **wav,
      )["audio_codes"]
      wav_size = input_ids.size()
      input_ids = input_ids.view((wav_size[1] * wav_size[2], wav_size[-1]))
      input_ids = self.shift_right(input_ids)
      ret = self.musicgen.generate(
          decoder_input_ids=input_ids,
          encoder_outputs=(img_embeds,),
          do_sample=True,
          guidance_scale=guidance_scale,
          max_new_tokens=256,
      )
    else:
      input_ids = torch.zeros((4, 1)).long().to(device)
      decoder_attention_mask = torch.ones((img_embeds.size(0), 1)).long().to(device)
      ret = self.musicgen.generate(
          decoder_input_ids=input_ids,
          decoder_attention_mask=decoder_attention_mask,
          encoder_outputs=(img_embeds,),
          do_sample=True,
          guidance_scale=guidance_scale,
          max_new_tokens=max_new_tokens,
      )
    return ret