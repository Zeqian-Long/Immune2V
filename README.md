<div align="center">
  
# 💉 Immune2V: Image Immunization Against Dual-Stream Image-to-Video Generation

[Zeqian Long](https://zeqian-long.github.io/)<sup>1*</sup>, [Ozgur Kara](https://karaozgur.com/)<sup>1*</sup>, [Haotian Xue](https://xavihart.github.io/)<sup>2*</sup>, [Yongxin Chen](https://yongxin.ae.gatech.edu/)<sup>2</sup>, [James M. Rehg](https://rehg.org/)<sup>1</sup>

<p><sup>*</sup> Equal contribution</p>

<sup>1</sup> University of Illinois Urbana-Champaign,  <sup>2</sup> Georgia Institute of Technology



<a href='https://immune2v.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![arXiv](https://img.shields.io/badge/arXiv-2604.10837-b31b1b.svg)](https://arxiv.org/abs/2604.10837)
[![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-orange.svg)](https://github.com/Zeqian-Long/Immune2V) 
[![GitHub Stars](https://img.shields.io/github/stars/Zeqian-Long/Immune2V)](https://github.com/Zeqian-Long/Immune2V)

</div>




<p>
We propose <b>Immune2V</b>, an image immunization framework designed to prevent the unauthorized animation of protected images by I2V models.
</p>

<p align="center">
<img src="resources/teaser.png" width="1080px"/>
</p>


# 🔥 News

- [2026.4.15] Paper released!


# 🛠️ Code Setup
We adopt our source code from DiffSynth-Studio (Licensed under Apache License 2.0.). You can refer to their [official repo](https://github.com/modelscope/DiffSynth-Studio), or running the following command to construct the environment.
```
pip install -r requirements.txt
```


We recommend running the experiments on a GPU with at least 80GB VRAM (e.g., **A100-80GB**). 
The system should also provide at least **100GB** of available disk space to download and store the model weights.


# 🤗 Model Download
Download models using huggingface-cli:

```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan-AI/Wan2.1-I2V-14B-480P 
```
Or modify the local-dir as needed.


# 🪄 Immunize your own image

## Command Line
You can run the following scripts in the terminal to attack your own image. 
```
python -m run_attack.preprocess_data
python -m run_attack.Immune-attack
```

Please modify the hyperparameters in config.yaml accordingly.


Then test using
```
python -m run_attack.Immune-test
```

Some examples shown in the paper are provided in the folder `/attacked`. Due to stochastic sampling and random seed variations, reproduced results may exhibit slight differences.

To run the whole dataset, run
```
chmod +x run_batch_attack.sh
./run_batch_attack.sh
```

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{long2026immune2v,
  title={Immune2V: Image Immunization Against Dual-Stream Image-to-Video Generation},
  author={Long, Zeqian and Kara, Ozgur and Xue, Haotian and Chen, Yongxin and James M., Rehg},
  year={2026}
}

```
