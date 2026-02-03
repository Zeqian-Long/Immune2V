<div align="center">
  
# Preliminary Implementation on Wan2.1 I2V Attack


</div>




# 🛠️ Code Setup
We adopt our source code from DiffSynth-Studio. You can refer to their [official repo](https://github.com/modelscope/DiffSynth-Studio), or running the following command to construct the environment.
```
pip install -r requirements.txt
```


We recommend you to run the experiment on a single A100 GPU (80G).


# 🤗 Model Download
Download models using huggingface-cli:

```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan-AI/Wan2.1-I2V-14B-480P 
```
Or modify the local-dir as needed.


# 🪄 Attack your own image




## Command Line
You can run the following scripts in the terminal to attack your own image. 
```
python ./run_attack/preprocess_data.py
python ./run_attack/Wan2.1-I2V-attack.py
```

Please modify the hyperparameters in cofig.yaml accordingly.


Then test using
```
python ./run_attack/Wan2.1-I2V-test.py
```



<!-- # 🖋️ Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support! -->

<!-- ```
@article{wang2024taming,
  title={Taming Rectified Flow for Inversion and Editing},
  author={Wang, Jiangshan and Pu, Junfu and Qi, Zhongang and Guo, Jiayi and Ma, Yue and Huang, Nisha and Chen, Yuxin and Li, Xiu and Shan, Ying},
  journal={arXiv preprint arXiv:2411.04746},
  year={2024}
}
``` -->

