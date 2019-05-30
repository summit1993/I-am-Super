# I-am-Super
Video Super Resolution Framework with Pytorch

## Environment
* Python 3.6 [+]
* Pytorch 1.0
* Ubuntu 16.04
* CUDA 8.0 [+]

## Models 
### VSR
1. Fast Spatio-Temporal Residual Network for Video Super-Resolution, CVPR, 2019. (FSTRN)
2. Recurrent Back-Projection Network for Video Super-Resolution, CVPR, 2019. (RBPN)
### SISR
1. 	Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee: Enhanced Deep Residual Networks for Single Image Super-Resolution. CVPR Workshops 2017: 1132-1140.
2. Jiahui Yu, Yuchen Fan, Jianchao Yang, Ning Xu, Zhaowen Wang, Xinchao Wang, Thomas S. Huang: Wide Activation for Efficient and Accurate Image Super-Resolution. CoRR abs/1808.08718.

## Run
* Run the main python file in Model folders, e.g., 

&ensp;&ensp;&ensp;&ensp; CUDA_VISIBLE_DEVICES=6,2,4 python Poker_main.py

* It is very convenient to train, evaluate or test the model, where you just need to edit the main file, e.g.,

&ensp;&ensp;&ensp;&ensp; if you just want to train the model, edit the code in the main file like
```python
# remove the val and test process
configs.dataset_configs.pop('val')
configs.dataset_configs.pop('test')
```

&ensp;&ensp;&ensp;&ensp; or if you just want to use the model to predict, edit the code in the main file like
```python
# set your trained model path
configs.model_configs['pre_model'] = "your trained model path"
# remove the train and val process
configs.dataset_configs.pop('train')
configs.dataset_configs.pop('val')
```


## Material
* VSR Framworks ([LoSealL/VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution))
* VSR papers ([flyywh/Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution))
* Open Video Restoration ([link](https://xinntao.github.io/open-videorestoration/rst_src/overview.html))

## Papers
#### Conference
1. Fast Spatio-Temporal Residual Network for Video Super-Resolution, CVPR, 2019. (FSTRN)
2. Recurrent Back-Projection Network for Video Super-Resolution, CVPR, 2019. (RBPN) ([code](https://github.com/alterzero/RBPN-Pytorch))
3. Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation, CVPR, 2018. (VSR-DUF) ([code1](https://github.com/yhjo09/VSR-DUF)) ([code2](https://github.com/HymEric/VSR-DUF-Reimplement))
4. Mehdi S. M. Sajjadi et al., Frame-Recurrent Video Super-Resolution, CVPR, 2018. (FRVSR)
5. Xin Tao et al., Detail-Revealing Deep Video Super-Resolution, ICCV, 2017. ([code](https://github.com/jiangsutx/SPMC_VideoSR))
6. Ding Liu et al., Robust Video Super-Resolution With Learned Temporal Dynamics, ICCV, 2017.
7. Deep Super Resolution for Recovering Physiological Information from Videos, CVPRW, 2018.
8. Video Super Resolution Based on Deep Convolution Neural Network With Two-Stage Motion Compensation, ICMEW, 2018.

#### Journal
1. Generative Adversarial Networks and Perceptual Losses for Video Super-Resolution, TIP, 2019.
2. Multi-Memory Convolutional Neural Network for Video Super-Resolution, TIP, 2019.
3. Learning Temporal Dynamics for Video Super-Resolution: A Deep Learning Approach, TIP, 2018.
4. Video Super-Resolution via Bidirectional Recurrent Convolutional Networks, TPAMI, 2018.

#### arXiv
1. EDVR: Video Restoration with Enhanced Deformable Convolutional Networks, arXiv, 2019, 05. (NTIRE 2019, 1st), ([code](https://github.com/xinntao/EDVR))
2. Two-Stream Oriented Video Super-Resolution for Action Recognition, arXiv, 2019, 03.
3. Yapeng Tian, Yulun Zhang, Yun Fu, and Chenliang Xu. TDAN: Temporally Deformable Alignment Network for Video Super-Resolution, arXiv, 2018, 12. (TDAN)
4. Temporally Coherent GANs for Video Super-Resolution (TecoGAN), arXiv, 2018, 11. ([code](https://github.com/thunil/TecoGAN))
5. 3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks, arXiv, 2018, 12.
6. Photorealistic Video Super Resolution, arXiv, 2018, 07.
7. Adapting Image Super-Resolution State-of-the-arts and Learning Multi-model Ensemble for Video Super-Resolution, arXiv, 2019, 5. (NTIRE 2019, 2nd)

## Models to be Reproduced
1. VSR-DUF
  (Tip or Que: replace the 3D Conv with FSTRN Model, and discard the BN Layers)
2. EDVR 
2. FRVSR
4. TDAN
