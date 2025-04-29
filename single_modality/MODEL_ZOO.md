# Model Zoo

## Note

- For all the pretraining and finetuning, we adopt spaese/uniform sampling.
- `#Frame` $=$ `#input_frame` $\times$ `#crop` $\times$ `#clip`
- `#input_frame` means how many frames are input for model per inference
- `#crop` means spatial crops (e.g., 3 for left/right/center)
- `#clip` means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

## Pretraining

TBD

## Distillation

TBD

## Finetuning

### K710

TBD


### K400

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 91.3 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-K400/blob/main/1B_ft_k710_ft_k400_f8.pth) | TBD |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 91.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-K400/blob/main/1B_ft_k710_ft_k400_f16.pth) | TBD |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 91.9 | TBD | TBD |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 92.1 | TBD | TBD |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash PT + K710 FT  | 8x3x4    | 85.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_ft_k710_ft_k400_f8/pytorch_model.bin) | TBD |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash PT + K710 FT  | 8x3x4    | 87.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_ft_k710_ft_k400_f8/pytorch_model.bin) | TBD |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash PT + K710 FT  | 8x3x4    | 90.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin) | TBD |
| $\text{FluxViT}_{dist}$-S/14 | K-Mash PT + K710 FT  | 8x3x4    | 87.3 | [Link](https://drive.google.com/file/d/1OTjTsAnZGaq7AufDaw8IYLeSgmLYZjds/view?usp=sharing) | [run.sh](./exp/small/eval/k400_eval.sh) |
| $\text{FluxViT}_{dist}$-B/14 | K-Mash PT + K710 FT  | 8x3x4    | 89.3 | TBD | TBD |



### SthSth V2

TBD
