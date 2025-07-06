# PIC_Dense
Counting by Points: Density-Guided Weakly-Supervised Nuclei Segmentation in Histopathological Images

### Using the code

### Requirement
```
conda env create -f environment.yml
```

### Data preparation

#### Generate Ground Truth Density Maps
Please refer to [./datamodel/gen_den_map.py] for the density_map of the datasets.

#### Generate Pseudo Instance Labels
Please refer to [./datamodel/gen_cluster_labels.py] for the pse_inst_map of the datasets.

####  Combine Ground Truth Instance Labels, Pseudo Instance Labels, Ground Truth Density Maps, and Point Annotations
Please refer to [./datamodel/conbine_to_mask.py] for the mask of the datasets used in training.

### Training
1. Configure the training settings according to your requirements and dataset. You can refer to the sample configuration file at [./configs/examples/nuclei_segmentation/PanNuke_Train_example.yaml].

2. Run the following command to train the model:
```
python ./nuclei_segmentation/run_Dense.py --config ./configs/examples/nuclei_segmentation/PanNuke_Train_example.yaml --gpu 0
```

### Inference
Run the following command to inference :
```
python ./cell_segmentation/inference/inference_Dense_experiment_pannuke.py --run_dir ./log_path_checkpoints_saved --gpu 0
```
