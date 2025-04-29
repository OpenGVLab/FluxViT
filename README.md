# Make Your Training Flexible: Towards Deployment-Efficient Video Models

This repo is the official implementation of "[Make Your Training Flexible: Towards Deployment-Efficient Video Models](https://arxiv.org/abs/2503.14237)". By Chenting Wang, Kunchang Li, Tianxiang Jiang, XiangyuZeng, Yi Wang, and Limin Wang.

<img src="./figs/teaser.png" alt="teaser" width="50%"/>
 
## Update

- **2025/04/29**: Release of the single modality evaluation script and part of model weights.
- **2025/03/18**: We build the repo and release the [paper](https://arxiv.org/abs/2503.14237).

## Introduction

Popular video training methods mainly operate on a fixed number of tokens sampled from a predetermined spatiotemporal grid, resulting in sub-optimal accuracy-computation trade-offs due to inherent video redundancy. They also lack adaptability to varying computational budgets for downstream tasks, hindering applications of the most competitive model in real-world scenes. We thus propose a new test setting, Token Optimization, for maximized input information across budgets, which optimizes the size-limited set of input tokens through token selection from more suitably sampled videos. To this end, we propose a novel augmentation tool termed Flux. By making the sampling grid flexible and leveraging token selection, it is easily adopted in most popular video training frameworks, boosting model robustness with nearly no additional cost. We integrate Flux in large-scale video pre-training, and the resulting FluxViT establishes new state-of-the-art results across extensive tasks at standard costs. Notably, with 1/4 tokens only, it can still match the performance of previous state-of-the-art models with Token Optimization, yielding nearly 90\% savings.

<img src="./figs/main.png" alt="Sampling Strategy" width="40%"/>

## Performance

### Single Modality Action Recognition

#### K400

- Base Scale

| **Model**        | **GFLOPs**       | **Top-1** | **Top-1 + TO** |
|------------------|------------------|-----------|----------------|
| InternVideo2-B   | 440×12           | 87.4      |      -         |
| FluxViT-B        | 440×12           | 89.6      |      90.0      |
| FluxViT-B        | 255×12           | 89.3      |      89.7      |
| FluxViT-B        | 108×12           | 87.3      |      88.9      |
| FluxViT-B        | 49×12            | 84.7      |      87.4      |

- Small Scale

| **Model**        | **GFLOPs**       | **Top-1** | **Top-1 + TO** |
|------------------|------------------|-----------|----------------|
| InternVideo2-S   | 154×12           | 85.8      |       -        |
| FluxViT-S        | 154×12           | 87.7      |      88.0      |
| FluxViT-S        | 83×12            | 87.3      |      87.7      |
| FluxViT-S        | 32×12            | 84.7      |      86.6      |
| FluxViT-S        | 13×12            | 80.1      |      84.7      |

#### SSv2

- Base Scale

| **Model**        | **GFLOPs**      | **Top-1** | **Top-1 + TO** |
|------------------|-----------------|-----------|----------------|
| InternVideo2-B   | 253×6           | 73.5      |      -         |
| FluxViT-B        | 440×6           | 75.3      |      75.6      |
| FluxViT-B        | 255×6           | 75.1      |      75.7      |
| FluxViT-B        | 108×6           | 72.0      |      75.1      |
| FluxViT-B        | 49×6            | 56.8      |      73.9      |

- Small Scale

| **Model**        | **GFLOPs**      | **Top-1** | **Top-1 + TO** |
|------------------|-----------------|-----------|----------------|
| InternVideo2-S   | 154×6           | 71.5      |       -        |
| FluxViT-S        | 154×6           | 73.4      |      73.8      |
| FluxViT-S        | 83×6            | 72.9      |      73.4      |
| FluxViT-S        | 32×6            | 70.0      |      72.5      |
| FluxViT-S        | 13×6            | 55.3      |      70.9      |

### Multi Modality Video-Text Retrieval

<img src="./figs/zs_vt_retrieval.png" alt="zs_vt_retrieval" width="40%"/>

### Multi Modality VideoChat Model

Coming Soon

## Acknowledgement

This repository is built based on [UniFormer](https://github.com/Sense-X/UniFormer), [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [VINDLU](https://github.com/klauscc/VindLU) and [Unmasked Teacher](https://github.com/OpenGVLab/unmasked_teacher/) repository. 
