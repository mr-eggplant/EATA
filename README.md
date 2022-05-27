# 🌄 Efficient Test-Time Model Adaptation without Forgetting

This is the official project repository for [Efficient Test-Time Model Adaptation without Forgetting 🔗](https://arxiv.org/abs/2204.02610) by
Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao and Mingkui Tan **(ICML 2022)**.

🌄 EATA conducts model learning at test time to adapt a pre-trained model to test data that has distributional shifts ☀️ 🌧 ❄️, such as corruptions, simulation-to-real discrepancies, and other differences between training and testing data.
- 1️⃣: EATA conducts selective/sample-adaptative optimization, i.e., only perform backward propagation for reliable and non-redundant test samples, which improves adaptaion efficiency and performance simtaneously.

- 2️⃣: EATA regularizes the model parameters during testing to prevent the  forgetting on in-distribution/clean test samples 😋. 



**Installation**:

EATA depends on

- Python 3
- [PyTorch](https://pytorch.org/) >= 1.0


**Data preparation**:

This repository contains code for evaluation on ImageNet and [ImageNet-C 🔗](https://arxiv.org/abs/1903.12261) with ResNet models. But feel free to use your own data and models!

- Step 1: Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc). 

- Step 2: Put IamgeNet-C at "--data_corruption" and put ImageNet **test/val set**  at  "--data".



**Usage**:

```
import eata

model = TODO_model()

model = eata.configure_model(model)
params, param_names = eata.collect_params(model)
optimizer = TODO_optimizer(params, lr=2.5e-4)
adapt_model = eata.EATA(model, optimizer, fishers, e_margin, d_margin) 

outputs = adapt_model(inputs)  # now it infers and adapts!

```
Notes: 
- fishers are pre-calculated (see main.py) for preventing forgetting, 
- e_margin and d_margin are two parameters for selective (sample-adaptive) optimization.

## Example: Adapting a pre-trained ResNet-50 model on ImageNet-C (Corruption).

**Usage**:

```
python3 main.py --data /path/to/imagenet --data_corruption /path/to/imagenet-c --exp_type 'continual' or 'each_shift_reset' --algorithm 'eata' or 'eta' or 'tent' --output /output/dir
```

'--exp_type' is choosen from:
- 'continual'  means the model parameters will never be reset, also called online adaptation; 

- 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.


'--algorithm' is chosen from:

- 'tent' (baseline);
- 'eta' (ours eata w/o regularization);
- 'eata' (ours)

**Results**:

Here, we report the results on ImageNet-C, severity level = 5, with ResNet-50.

- **Corruption accuracy** (ETA/EATA achieves higher corruption accuracy but use fewer backward passes (compared to TTT, Tent), thereby more efficient):

| Method             | Gauss. | Shot | Impul. | Defoc. | Glass | Motion | Zoom | Snow | Frost | Fog  | Brit. | Contr. | Elastic | Pixel | JPEG | Avg. #Forwards | Avg. #Backwards |
|--------------------|--------|------|--------|--------|-------|--------|------|------|-------|------|-------|--------|---------|-------|------|----------------|-----------------|
| R-50 (GN)+JT       | 94.9   | 95.1 | 94.2   | 88.9   | 91.7  | 86.7   | 81.6 | 82.5 | 81.8  | 80.6 | 49.2  | 87.4   | 76.9    | 79.2  | 68.5 | 50,000         | 0               |
|   +[TTT 🔗](http://proceedings.mlr.press/v119/sun20b.html)             | 69.0   | 66.4 | 66.6   | 71.9   | 92.2  | 66.8   | 63.2 | 59.1 | 81.0  | 49.0 | 38.2  | 61.1   | 50.6    | 48.3  | 52.0 | 50,000✖️21     | 50,000✖️20      |
| R-50 (BN)          | 97.8   | 97.1 | 98.2   | 82.1   | 90.2  | 85.2   | 77.5 | 83.1 | 76.7  | 75.6 | 41.1  | 94.6   | 83.1    | 79.4  | 68.4 | 50,000         | 0               |
|   +[Tent 🔗](https://arxiv.org/abs/2006.10726)            | 71.6   | 69.8 | 69.9   | 71.8   | 72.7  | 58.6   | 50.5 | 52.9 | 58.7  | 42.5 | 32.6  | 74.9   | 45.2    | 41.5  | 47.7 | 50,000         | 50,000          |
|   +ETA (ours)      | 64.9   | 62.1 | 63.4   | 66.1   | 67.1  | 52.2   | 47.4 | 48.1 | 54.2  | 39.9 | 32.1  | 55.0   | 42.1    | 39.1  | 45.1 | 50,000         | 26,031          |
|   +EATA (ours)     | 65.0   | 63.1 | 64.3   | 66.3   | 66.6  | 52.9   | 47.2 | 48.6 | 54.3  | 40.1 | 32.0  | 55.7   | 42.4    | 39.3  | 45.0 | 50,000         | 25,150          |
|   +EATA (lifelong) | 65.0   | 61.9 | 63.2   | 66.2   | 65.8  | 52.7   | 46.8 | 48.9 | 54.4  | 40.3 | 32.0  | 55.8   | 42.8    | 39.6  | 45.3 | 50,000         | 28,243          |
<!-- |   +TTA             | 95.9   | 95.1 | 95.5   | 87.5   | 91.8  | 87.1   | 74.2 | 86.0 | 80.9  | 78.7 | 47.0  | 87.6   | 85.4    | 75.4  | 66.4 | 50,000✖️64     | 0               | -->
<!-- |   +BN              | 84.5   | 83.9 | 83.7   | 80.0   | 80.0  | 71.5   | 60.0 | 65.2 | 65.0  | 51.5 | 34.1  | 75.9   | 54.2    | 49.3  | 58.9 | 50,000         | 0               | -->
<!-- |   +MEMO            | 92.5   | 91.3 | 91.0   | 80.3   | 87.0  | 79.3   | 72.4 | 74.7 | 71.2  | 67.9 | 39.0  | 89.0   | 76.2    | 67.0  | 62.5 | 50,000✖️65     | 50,000✖️64      | -->

- **Clean accuracy** (testing the model's source accuracy on clean/original imagenet test set). EATA improves model's corruption acc. and maintains the source acc., while Tent can not.  

<p align="center">
<img src="figures/forgetting_results.png" alt="forgetting_results" width="100%" align=center />
</p>



Please see our [PAPER 🔗](https://arxiv.org/abs/2204.02610) for detailed results.



## Correspondence

Please contact Shuaicheng Niu by niushuaicheng [at] gmail.com 📬.


## Citation
If the EATA method or fully test-time adaptation without forgetting are helpful in your research, please consider citing our paper:
```
@InProceedings{niu2022efficient,
  title={Efficient Test-Time Model Adaptation without Forgetting},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Chen, Yaofo and Zheng, Shijian and Zhao, Peilin and Tan, Mingkui},
  booktitle = {The Internetional Conference on Machine Learning},
  year = {2022}
}
```

## Acknowledgment
The code is greatly inspired by (heavily from) the [Tent 🔗](https://github.com/DequanWang/tent) and [TTT 🔗](https://github.com/yueatsprograms/ttt_imagenet_release).
