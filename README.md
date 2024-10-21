# Allegro
<p align="center">
<img src="https://github.com/rhymes-ai/Allegro/blob/main/allegro/assets/Allegro_banner.gif"/>
</p>


<p align="center">
 <a href="https://rhymes.ai/" target="_blank"> Gallery</a> · <a href="https://huggingface.co/rhymes-ai/Allegro" target="_blank">HuggingFace</a> · <a href="https://www.rhymes.ai/blog-details/" target="_blank">Blog</a> · <a href="https://arxiv.org/pdf/2410.05993" target="_blank">Paper</a> · <a href="https://discord" target="_blank">Discord</a> 
</p> 
Allegro is capable of producing high-quality, 6-second videos at 30 frames per second and 720p resolution from simple text prompts.

# Model Info
<table>
  <tr>
    <th>Model</th>
    <td>Allegro</td>
  </tr>
  <tr>
    <th>Description</th>
    <td>Text-to-Video Generation Model</td>
  </tr>
 <tr>
    <th>Download</th>
    <td><a href="https://huggingface.co/rhymes-ai/Allegro">Hugging Face</a></td>
</tr>
  <tr>
    <th rowspan="2">Parameter</th>
    <td>VAE: 175M</td>
  </tr>
  <tr>
    <td>DiT: 2.8B</td>
  </tr>
  <tr>
    <th rowspan="2">Inference Precision</th>
    <td>VAE: FP32/TF32/BF16/FP16 (best in FP32/TF32)</td>
  </tr>
  <tr>
    <td>DiT/T5: BF16/FP32/TF32</td>
  </tr>
  <tr>
    <th>Context Length</th>
    <td>79.2k</td>
  </tr>
  <tr>
    <th>Resolution</th>
    <td>720 x 1280</td>
  </tr>
  <tr>
    <th>Frames</th>
    <td>88</td>
  </tr>
  <tr>
    <th>Video Length</th>
    <td>6 seconds @ 15 fps</td>
  </tr>
  <tr>
    <th>Single GPU Memory Usage</th>
    <td>9.3G BF16 (with cpu_offload)</td>
  </tr>
</table>

# Requirement
- Download the weight in Hugging Face: [rhymes-ai/Allegro ](https://huggingface.co/rhymes-ai/Allegro)
- Prerequisites: Python >= 3.10, PyTorch >= 2.4, CUDA >= 12.4.
- Tip: It is recommended to use Anaconda to create a new environment (Python >= 3.10) to run the following example.
```python 
  git clone https://github.com/rhythms-ai/allegro
  conda create -n allegro python=3.10 -y
  conda activate allegro
  
  pip install requirements
```

# Inference
Tip: It is highly recommended to use a video frame interpolation model (such as [EMA-VFI](https://github.com/mcg-nju/ema-vfi)) to enhance the result to 30 FPS.
```python
  python single_inference.py \
  --user_prompt 'A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats.' \
  --vae your/path/to/vae \
  --dit your/path/to/transformer \
  --text_encoder your/path/to/text_encoder \
  --tokenizer your/path/to/tokenizer \
  --guidance_scale 7.5 \
  --num_sampling_steps 100 \
  --seed 42
```
# Limitation
- The model cannot render celebrities, legible text, specific locations, streets or buildings.

# Future Plan
- Multiple GPU inference and further speed up (PAB)
- Text & Image-To-Video (TI2V) video generation
- Motion-controlled video generation
- Visual quality enhancement

# Support
If you encounter any problems or have any suggestions, feel free to open an [issue](https://github.com/rhymes-ai/Allegro/issue) or send an email to huanyang@rhymes.ai. 

# Citation

# License
This repo is released under the Apache 2.0 License.

# Disclaimer

The Allegro model is provided on an "AS IS" basis, and we disclaim any liability for consequences or damages arising from your use. Users are kindly advised to ensure compliance with all applicable laws and regulations. This includes, but is not limited to, prohibitions against illegal activities and the generation of content that is violent, pornographic, obscene, or otherwise deemed non-safe, inappropriate, or illegal. By using these models, you agree that we shall not be held accountable for any consequences resulting from your use.

# Acknowledgment
We extend our heartfelt appreciation for the great contribution to the open-source community, especially Open-Sora-Plan, as we build our diffusion transformer (DiT) based on Open-Sora-Plan v1.2.
- [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan): A project aims to create a simple and scalable repo, to reproduce Sora.
- [Open-Sora](https://github.com/hpcaitech/Open-Sora): An initiative dedicated to efficiently producing high-quality video.
- [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization system.
- [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): An open-source project that provides a user-friendly and high-performance infrastructure for video generation. 
- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
- [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
- [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
- [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
- [Playground](https://playground.com/blog/playground-v2-5): A state-of-the-art open-source model in text-to-image generation.
