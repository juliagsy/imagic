# Imagic

## Introduction

## Usage

```python
from transformers import AutoImageProcessor
from imagic.hf import ImagicModel

vit_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", low_cpu_mem_usage=True)

i2m = ImagicModel.from_pretrained("juliagsy/imagic")
```

## Image-conditioned Music Generation

### Generation

```python
from IPython.display import Audio
from IPython.core.display import display

input_img = "<your-image>"

img = vit_proc(input_img, do_rescale=False, return_tensors="pt")
img = img.to("cuda")

gen_wav = i2m.generate(img, guidance_scale=3)
sampling_rate = i2m.model.musicgen.config.audio_encoder.sampling_rate
display(Audio(gen_wav[0].cpu().numpy(), rate=sampling_rate))
```

### Example

Example image:

![example image 1](examples/img_5.png)

[Output wav](examples/wav_5.wav)

