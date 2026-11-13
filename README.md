# Breaking Feature Collapse in Self-Supervised Time Series Encoders

Official code and pre-trained weights of OTISv2, a general-purpose time series encoder. OTISv2 combines masked data modelling with self-distillation and a KoLeo regulariser, which act directly on the feature space of the encoder.

## Environment Setup
Run the following commands from the root directory of this project to set up the environment. This is needed only once.
```
conda env create --file envs/otisv2.yaml
conda activate otisv2
```

## Pre-Processing & Data Formatting
OTISv2 encodes variate and time positions with axial rotary position embeddings (RoPE), so the pre-trained weights apply to any number of variates and time points.

### Downstream Data
Fine-tuning, linear probing, forecasting and the frozen evaluation read each data split from one `.pt` file. The file holds one tensor of the shape (N, V, T) or (N, C, V, T), where N, C, V, and T represent the number of samples, the number of channels (similar to RGB channels in images; default: 1), the number of variates, and the number of time points, respectively. A Python list with one tensor per sample, of the shape (V, T) or (C, V, T), is also valid. The list entries can be tuples (_name_: str, _sample_: torch.Tensor), which is the format of OTIS; OTISv2 does not use _name_.

Use `util/preprocess.py` to normalise and clamp raw data of the shape (N, V, T), stored as `.pt`, `.npy` or `.npz`.
```
python3 util/preprocess.py --input $raw_data --output $data_path
```

The labels (`--labels_path`) are one tensor in a `.pt` file: one-hot rows of the shape (N, number of classes) for classification, and target values of the shape (N, number of targets) for regression.

### Pre-Training Data
Pre-training reads a memory-mapped corpus, which is a folder with two files:
- `data.bin`: the float32 values of all samples, one sample after the other. Each sample has the shape (V, T) and is stored row by row.
- `index.npy`: an int64 array of the shape (N, 3). Row i holds (offset, V, T) of sample i, where offset is the position of its first value in `data.bin`, counted in float32 values.

## Model Weights
Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/16kCh4raI3Uvgz9wt3gbvKVC3-QB-OCL7?usp=sharing).

| File  | Parameters | `--model` (classification, regression) | `--model` (forecasting) |
|---|---|---|---|
| `otisv2_7.1M.pth` | 7.1M | `otis_base` | `otis_base_dec160d4b` |

Each file holds the pre-trained encoder and the decoder (1.7M parameters).

Pass a file to the training commands below with `--finetune $checkpoint`. Without `--finetune`, the model trains from a random initialisation.

Use these settings to match pre-training:
- The `--model` name from the table.
- `--patch_height 1 --patch_width 24`. A different patch size re-initialises the patch embedding.
- `--rope_theta 200000 --num_register_tokens 4 --attention_sink`. Do not pass `--qk_norm`. The training scripts also read these four settings from the file.

To load the weights in your own code, use `torch.load(checkpoint, map_location="cpu", weights_only=True)`. The weights are under the key `model`.

### Scaling Study
The encoders of the scaling study are available in the same folder. They are trained for one epoch on the full pre-training corpus (221M training tokens) and serve to reproduce the study, not to replace the weights above.

| File | Parameters | `--model` (classification, regression) | `--model` (forecasting) |
|---|---|---|---|
| `otisv2_scaling_7.1M.pth` | 7.1M | `otis_base` | `otis_base_dec160d4b` |
| `otisv2_scaling_14.8M.pth` | 14.8M | `otis_small` | `otis_small_dec160d4b` |
| `otisv2_scaling_40M.pth` | 40M | `otis_large` | `otis_large_dec160d4b` |
| `otisv2_scaling_116M.pth` | 116M | `otis_huge` | `otis_huge_dec160d4b` |

## Pre-Training
The pre-training data is a memory-mapped corpus, see [Pre-Training Data](#pre-training-data). Run the following command for OTISv2-7.1M on two GPUs (effective batch size 384, peak learning rate 1.2e-3). `--time_steps` must be a multiple of `--patch_width`.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node 2 --nnodes 1 --node_rank 0 main_pretrain.py --world_size 2 --dist_eval --num_workers $num_workers --seed $sd --model otis_base_dec160d4b --input_channels 1 --time_steps 1008 --patch_height 1 --patch_width 24 --batch_size 192 --epochs 50 --blr 1e-4 --weight_decay 0.025 --clip_grad 3.0 --layer_decay 0.975 --patch_embed_lr_mult 0.2 --warmup_fraction 0.1 --decay_fraction 0.1 --drop_path 0.1 --drop_path_decoder 0.1 --mask_ratio 0.7 --probabilistic_masking --include_forecasting --forecasting_probability 0.33 --forecasting_mask_ratio 0.5 --crop_lower_bnd 0.5 --crop_upper_bnd 1.0 --jitter_sigma 0.25 --rescaling_sigma 0.5 --ft_surr_phase_noise 0.1 --ncc_weight 0.1 --sd_weight 0.1 --ibot_teacher_temp_start 0.04 --ibot_teacher_temp 0.055 --ibot_teacher_temp_warmup_fraction 0.04 --ibot_ema_decay 0.996 --ibot_ema_cosine --ibot_freeze_last_layer_fraction 0.02 --ibot_teacher_mask_ratio 0.5 --koleo_weight 0.025 --rope_theta 200000 --num_register_tokens 4 --data_path $data_path --output_dir $output_dir
```

The command needs about 48 GB of memory per GPU. With less memory, lower `--batch_size` and raise `--accum_iter` so that the effective batch size stays the same, e.g. `--batch_size 96 --accum_iter 2`. On a single GPU, set `--nproc_per_node 1 --world_size 1` and keep the effective batch size, e.g. `--batch_size 64 --accum_iter 6`.

The self-distillation teacher has three options:
- `--ibot_teacher_real_superset`: the teacher view always contains more real patches than the student view.
- `--no-ibot_teacher_drop_path`: the teacher runs without drop path.
- `--no-ibot_freeze_cls_head_only`: the freeze at the start of training also covers the last layer of the patch head.

## Training
### Classification
Run the following command.
```
python3 main_finetune.py --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --epochs $epochs --blr $blr --warmup_fraction $warmup_fraction --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

For several GPUs, or inside a Slurm job (also with one GPU), run the following command.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune.py --world_size $world_size --dist_eval --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --blr $blr --epochs $epochs --warmup_fraction $warmup_fraction --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

### Regression
Run the following command for a multi-output regression with N variables.
```
python3 main_finetune.py --num_workers $num_workers --seed $sd --downstream_task regression --nb_classes N --lower_bnd 0 --upper_bnd N --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --blr $blr --epochs $epochs --warmup_fraction $warmup_fraction --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

For several GPUs, or inside a Slurm job (also with one GPU), run the following command.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_finetune.py --world_size $world_size --dist_eval --num_workers $num_workers --seed $sd --downstream_task regression --nb_classes N --lower_bnd 0 --upper_bnd N --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --blr $blr --epochs $epochs --warmup_fraction $warmup_fraction --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

### Linear Probing
Run the following command. It trains a linear head, the [CLS] token and the register tokens; all other encoder weights stay frozen.
```
python3 main_linprobe.py --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --epochs $epochs --blr $blr --warmup_fraction $warmup_fraction --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

### Forecasting
Run the following command.
```
python3 main_forecast.py --num_workers $num_workers --seed $sd --downstream_task forecasting --mask_ratio $mr --input_channels $input_channels --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --blr $blr --epochs $epochs --warmup_fraction $warmup_fraction --data_path $data_path --val_data_path $val_data_path --output_dir $output_dir
```

For several GPUs, or inside a Slurm job (also with one GPU), run the following command.
```
torchrun --rdzv-endpoint=localhost:$port --nproc_per_node $world_size --nnodes $nodes --node_rank 0 main_forecast.py --world_size $world_size --dist_eval --num_workers $num_workers --seed $sd --downstream_task forecasting --mask_ratio $mr --input_channels $input_channels --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --model $model --finetune $checkpoint --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --blr $blr --epochs $epochs --warmup_fraction $warmup_fraction --data_path $data_path --val_data_path $val_data_path --output_dir $output_dir
```

## Evaluation
### Frozen Evaluation
`main_frozen_eval.py` evaluates the pre-trained weights for classification without any training of the encoder. It embeds each data set once and scores three classifiers on the frozen embeddings: k-NN (k = 5), 5-shot prototypes (mean over three random draws) and a linear classifier. It reports the balanced accuracy for the [CLS] token (`cls`) and for the average of the patch tokens (`gap`). The script reads the model settings from the weights file.

For your own data, pass a labelled training split and the split to evaluate.
```
python3 main_frozen_eval.py --finetune $checkpoint --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

For the UEA and UCR benchmarks, run the following command.
```
python3 main_frozen_eval.py --finetune $checkpoint --uea_data_base $uea_data_base --ucr_data_base $ucr_data_base --output_dir $output_dir
```
`$uea_data_base` and `$ucr_data_base` hold one folder per data set, named as in the archives (see `util/benchmarks.py`). Each folder holds `train.pt`, `train_labels.pt`, `test.pt` and `test_labels.pt` in the format of [Downstream Data](#downstream-data).

Build these files from the archive files with `util/prepare_uea_ucr.py`. The script reads `<name>_TRAIN.ts` and `<name>_TEST.ts` in each folder. Phoneme in the UEA list is the uni-variate version of the UCR archive.
```
python3 util/prepare_uea_ucr.py --archive uea --data_base $uea_data_base
python3 util/prepare_uea_ucr.py --archive ucr --data_base $ucr_data_base
```

### Fine-Tuned Models and Zero-Shot Forecasting
Use the `--eval` flag and pass the checkpoint with `--resume`. The evaluation runs on the data of `--val_data_path`.

#### Classification and Regression
Pass a fine-tuned checkpoint, i.e. a file that the training wrote to `$output_dir`. The pre-trained weights hold no task head, so train one first (see [Training](#training)). For a checkpoint from linear probing, use `main_linprobe.py` instead of `main_finetune.py`. For classification tasks, e.g. run the following command.
```
python3 main_finetune.py --eval --resume $finetuned_checkpoint --num_workers $num_workers --seed $sd --downstream_task classification --nb_classes $nb_classes --input_channels $input_channels --input_variates $input_variates --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --model $model --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --epochs $epochs --blr $blr --warmup_fraction $warmup_fraction --data_path $data_path --labels_path $labels_path --val_data_path $val_data_path --val_labels_path $val_labels_path --output_dir $output_dir
```

#### Forecasting
Pass a fine-tuned checkpoint, or the pre-trained weights for zero-shot forecasting. `--mask_ratio` is the share of the time series, at its end, that the model forecasts.
```
python3 main_forecast.py --eval --resume $checkpoint --num_workers $num_workers --seed $sd --downstream_task forecasting --mask_ratio $mr --input_channels $input_channels --time_steps $time_steps --patch_height $patch_height --patch_width $patch_width --ncc_weight $ncc --model $model --rope_theta 200000 --num_register_tokens 4 --attention_sink --batch_size $bs --blr $blr --epochs $epochs --warmup_fraction $warmup_fraction --data_path $data_path --val_data_path $val_data_path --output_dir $output_dir
```

## Citation
Please cite the following work:
```
@inproceedings{Turgut2026,
  title={Breaking Feature Collapse in Self-Supervised Time Series Encoders},
  author={Turgut, {\"O}zg{\"u}n and Rueckert, Daniel},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Notice
The code and the pre-trained weights are released under the MIT License. This project includes third-party software components that are subject to their respective licenses. Detailed information including component names, licenses, and copyright holders is provided in the respective files. Please review the [LICENSE](LICENSE) file before using or distributing this software.
