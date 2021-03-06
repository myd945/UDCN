# UDCN
## Unsupervised Decomposition and Correction Network for Low-light Image Enhancement

The Repo is the open source for the paper. 

## Description
Normally illuminated images with high visibility are the prerequisites of many computer vision systems and algorithms. Therefore, low-light image enhancement (LIE) is of paramount importance. While deep learning-based methods have achieved substantial success during the past few years, most of them require paired training data, which is difficult to be collected in the field of LIE. With this in mind, this paper advocates a novel Unsupervised Decomposition and Correction Network (UDCN) for LIE without depending on paired data for training. Inspired by the Retinex model, our method first decomposes images into illumination and reflectance components with an image decomposition network (IDN). Then, the decomposed illumination is processed by an illumination correction network (ICN) and fused with the reflectance to generate a primary enhanced result. In contrast with fully supervised learning approaches, UDCN is an unsupervised one which is trained only with low-light images and corresponding histogram equalized (HE) counterparts (can be derived from the low-light image itself) as input. Both the decomposition and correction networks are optimized under the guidance of hybrid no-reference quality-aware losses and inter-consistency constraints between the low-light image and its HE counterpart. In addition, we also utilize an unsupervised noise removal network (NRN) to remove the noise previously hidden in the darkness for further improving the primary result. Qualitative and quantitative comparison results are reported to demonstrate the efficacy of UDCN and its superiority over several representative alternatives in the literature.
## Overview
![fig_framework](https://github.com/myd945/UDCN/blob/main/fig_framework_00.png) 
Framework of the proposed UDCN which contains three sub-networks, i.e., image decomposition network (IDN), illumination correction network (ICN), and noise removal network (NRN)
## Results
![fig_re](https://github.com/myd945/UDCN/blob/main/fig_re_00.jpg)
![fig_re](https://github.com/myd945/UDCN/blob/main/HE_01.png)

## RLL Dataset

### Examples of the low-light images in our collected RLL dataset. First and second rows correspond to the outdoor and indoor scenes, respectively. 
![fig_re](https://github.com/myd945/UDCN/blob/main/fig_RLLdataset_00.jpg)
We collect 800 real-world low-light images from various exsisted datasets, containing rich indoor and outdoor scenes. The collected Real-world Low-Light dataset is called RLL in short.
#### Dataset Download
[BaiduPan](https://pan.baidu.com/s/1fjhdsmOs6eoo7cg_HQEzhw) ?????????:RLLD       [Google Drive](https://drive.google.com/file/d/1D5WiL3EGCgBEONoNPXirjkOvwq03GY3T/view?usp=sharing)

## Other test Dataset
VV Dataset[BaiduPan](https://pan.baidu.com/s/1TNDPSIkN3PZd3YxlEvBX_w) ?????????:z5t5       [Google Drive](https://drive.google.com/file/d/10IJ83MvpThSjeOFT7G54iXfLyerPwK46/view?usp=sharing)   
DICM Dataset[BaiduPan](https://pan.baidu.com/s/1E6xYoNcItMJfcZ0kOR-vVQ) ?????????:p2jd       [Google Drive](https://drive.google.com/file/d/1n8cZTejRPAsNRAr5hTPpvE7Du-UEhvi-/view?usp=sharing)  
NPE Dataset[BaiduPan](https://pan.baidu.com/s/1MFZ7f0On_q9dJyyeZJ5sxQ) ?????????:7nu9       [Google Drive](https://drive.google.com/file/d/1OJgiGg4LAOovmjlkNWjUHT6wSSi3-TFz/view?usp=sharing)   
LIME Dataset[BaiduPan](https://pan.baidu.com/s/1LB4Bq7ZxMcJI_M2frl83HA) ?????????:ndyj       [Google Drive](https://drive.google.com/file/d/1mDRUHlCa4eMSXvOZ4jO9tkVMBfL_Q6Jn/view?usp=sharing)   
MEF Dataset[BaiduPan](https://pan.baidu.com/s/1dZ4_4TSgVwD-5KEdlymmcg) ?????????:bblf       [Google Drive](https://drive.google.com/file/d/1xYjcmszjQKR48X9_AJ3_sMhZSaN6jL08/view?usp=sharing)    
[ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

## Code

The source code will be available after the paper is published.

## Cite
    @ARTICLE{9757816,
    author={Jiang, Qiuping and Mao, Yudong and Cong, Runmin and Ren, Wenqi and Huang, Chao and Shao, Feng},
    journal={IEEE Transactions on Intelligent Transportation Systems}, 
    title={Unsupervised Decomposition and Correction Network for Low-Light Image Enhancement}, 
    year={2022},
    volume={},
    number={},
    pages={1-16},
    doi={10.1109/TITS.2022.3165176}}
