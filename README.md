# SOoD: Self-Supervised Out-of-Distribution Detection Under Domain Shift for Multi-Class Colorectal Cancer Tissue Types

This repository contains the Pytorch implementation of the method proposed the _SOoD: Self-Supervised Out-of-Distribution Detection Under Domain Shift for Multi-Class Colorectal Cancer Tissue Types_ , which recently  has been accepted at ICCV 2021 Workshop (CVMAD).

## A) Dependencies

Install dependencies via executing the `setup.sh` script using the conda environment. We use Python 3.9 and Pytorch 1.9.

## B) Training the SOoD Model 

To visualize the training process, setup the visdom server by running:

```
visdom -p 1076 -env_path path_to_repo/src/Visdom
```

To train our model, run:

```
python main.py SOoD Kather expname --aug_type E/H --n_prototypes 16 --ssl_mode sinkhorn
```

Run unsupervised fine-tuning with:

```
python main.py SOoDftUns Kather expname2 --ssl_ckpt Experiments/expname/Checkpoints/SOoD_best_loss_source.pth --aug_type E/H --n_prototypes 16 --ssl_mode sinkhorn
```

Run supervised fine-tuning with:

```
python main.py SOoDftClassifier Kather expname3 --ssl_ckpt Experiments/expname/Checkpoints/SOoD_best_loss_source.pth --aug_type E/H --n_prototypes 16 --ssl_mode sinkhorn --pct_data percentage_data
```

## C) Set the parameters to run baselines and ablations:

* 3heavy+1style: `--lmd_h 3 --lmd_ood 1`
* 1heavy+3style: `--lmd_h 1 --lmd_ood 3`
* 0heavy+1style: `--lmd_h 0 --lmd_ood 1`
* 1heavy+0style: `--lmd_h 1 --lmd_ood 0`
* 8 prototypes: `--n_prototypes 8`
* 24 prototypes: `--n_prototypes 24`
* colorization: `--aug_type E/H/S`

* Swav Baseline: `--aug_type normal --color_transformation --ssl_mode sinkhorn`
* DINO Baseline: `--aug_type normal --color_transformation --ssl_mode S/T`
* Classifier: `--aug_type normal --ssl_mode no`

## D) Testing

To test any model with name `model_name`, simply run:

```
python main.py model_name Kather expname --checkpoint path/to/checkpoint --test 
```

## E) Our results and checkpoints

#### Unsupervised methods

| Method | AUROC | AUPRC | Checkpoint |
|--------|--------|------------|------------|
| **SOoD-pretrain**  |  **88.38**      | **80.43**  | [ckpt](https://drive.google.com/file/d/1vNTPAM_u6EbSP5H9PehNswp4KZNZK5dE/view?usp=sharing) |
| **SOoD-finetune Unsupervised**|  **92.77 +/- 0.48**  | **90.90 +/- 1.00**       |[ckpt](https://drive.google.com/drive/folders/1HfWHXJhzu8dIxwEvCaLe8LHyLHs0zvX9?usp=sharing) |
| SOoD-pretrain 0heavy | 84.63       | 75.76         |[ckpt](https://drive.google.com/file/d/1Nm0QTMax0WhBE653t3BpZRUzI5I0R2iq/view?usp=sharing) |
| SOoD-finetune 0heavy | 91.60 +/- 0.59       | 90.62 +/- 0.54         |[ckpt](https://drive.google.com/drive/folders/19x-DXwiVp_3khK0nOvImkiaHqC3kZ1Wn?usp=sharing) |
| SOoD-pretrain 3heavy | 85.18       | 76.41         |[ckpt](https://drive.google.com/file/d/1IKdFhRtUUUbBjlvier0aGzZgzXbNroID/view?usp=sharing) |
| SOoD-finetune 3heavy | 90.77 +/- 0.47       | 87.95 +/- 0.33         |[ckpt](https://drive.google.com/drive/folders/12q562nHUfMPEZVtSvc24-HZDlMSIcZQw?usp=sharing) |
| SOoD-pretrain 0style | 83.95       | 75.63         |[ckpt](https://drive.google.com/file/d/1j8HBUpQjZ2DTcVequ6tlOv1djJ0bp_hx/view?usp=sharing) |
| SOoD-finetune 0style | 85.87 +/- 3.25       | 84.56 +/- 2.26         |[ckpt](https://drive.google.com/drive/folders/1_R9d2q3xR-1uwha-nLzU5fO5oO3MQuUQ?usp=sharing) |
| SOoD-pretrain 3style | 82.76       | 71.80         |[ckpt](https://drive.google.com/file/d/1kwFHiFKGK1aOERy338Sir7Xk6l-1GvBd/view?usp=sharing) |
| SOoD-finetune 3style | 90.99 +/- 0.28       | 88.64 +/- 0.39         |[ckpt](https://drive.google.com/drive/folders/1Zef1svtukrCpgJk3O0SpWeWMFz47G2Ou?usp=sharing) |
| SOoD-pretrain 8prots|  82.52      |  71.44        |[ckpt](https://drive.google.com/file/d/19aSVQcTe0rrbmMf4PIOrX3NAIAp080Hn/view?usp=sharing) |
| SOoD-finetune 8prots|  88.85 +/- 0.57      | 85.30 +/- 0.22         |[ckpt](https://drive.google.com/file/d/19aSVQcTe0rrbmMf4PIOrX3NAIAp080Hn/view?usp=sharing) |
| SOoD-pretrain 24prots| 86.70       | 79.60         |[ckpt](https://drive.google.com/file/d/1Uoj-xm4F-iVWltKqE14kniPb70l_cLm3/view?usp=sharing) |
| SOoD-finetune 24prots| 89.24 +/- 0.40       | 84.71 +/- 0.43         |[ckpt](https://drive.google.com/drive/folders/1pigoNjch66yKBBcu_DDTb3Vuom9V5dxh?usp=sharing) |
| SOoD-pretrain ColorStyle| 82.44       | 75.23         |[ckpt](https://drive.google.com/file/d/1j1kAXCqbAyZisKTXJ9u8HuGLEsYGttlG/view?usp=sharing) |
| SOoD-finetune ColorStyle| 88.58 +/- 0.73       | 86.54 +/- 0.57         |[ckpt](https://drive.google.com/drive/folders/1stXy_s0QaavjvuQfBLPW3iybbTPS4ljN?usp=sharing) |

#### Linear classification baselines on frozen features 
| Method | Linear | KNN | Checkpoint |
|--------|--------|------------|------------|
| **SOoD-finetune Supervised 100%** | **73.24 +/- 0.39**       |  **83.45**        |[ckpt](https://drive.google.com/drive/folders/1WFd3TVyGVEazZSQnwO5Xpn7UYaXVMX_7?usp=sharing) |
| SOoD-finetune Supervised 20% | 73.39 +/- 0.69       |    -      |[ckpt](https://drive.google.com/drive/folders/1WUGzAw8Shr2sBq2KapTMXgdTPr7U_LHq?usp=sharing) |
| SOoD-finetune Supervised 10% | 73.24 +/- 0.82       |    -      |[ckpt](https://drive.google.com/drive/folders/1d7eKgoXWqc9gTWiHGlbU6-WTh-iVhOtE?usp=sharing) |
| SOoD-finetune Supervised 1% |  62.59 +/- 1.42      |     -     |[ckpt](https://drive.google.com/drive/folders/1MFrewBUNJjC3ESTOShE_mayzAnXUNZfy?usp=sharing) |
| SwAV 100% |  48.83 +/- 1.83      |  41.72        |[ckpt](https://drive.google.com/drive/folders/1rD0hkyBJQzUaDn-i0lb6qS6ucrDztAWu?usp=sharing) |
| DINO 100% |  42.03 +/- 8.51      |  32.20        |[ckpt](https://drive.google.com/drive/folders/1emb3LgNtpngdfyq2ZgUp5XJq7JYdoTLt?usp=sharing) |
| Supervised Source 100% |  65.13 +/- 3.57      |   41.50 +/- 3.84       |[ckpt](https://drive.google.com/drive/folders/1BSKxDNIUSgE8Tb9kEc6A3FOAbSOS1e9y?usp=sharing) |
| Supervised Translated 100% | 78.31 +/- 5.98       | 77.48 +/- 1.61      |[ckpt](https://drive.google.com/drive/folders/1z7i_rJxJooKkb41nX1khxGMlS15N2Ewi?usp=sharing) |

----
### Licence

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
