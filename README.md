# Enhanced nnUNet

nnUNet was developed by Isensee et al. and further information on the original framework may be found by reading the following original paper:


    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
    for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
    
The nnUNet is a fully automated and generalisable framework which automatically configures the full training pipeline for any 3D medical imaging  segmentation task it is applied on, taking into account dataset properties and hardware constraints.  

The nnUNet utilises a standard UNet type architecture which is self-configuring in terms of both depth and hyperparameters. 
We extend the original [nnUNet code](https://github.com/MIC-DKFZ/nnUNet) by integrating features found in more advanced UNet variations, namely residual blocks.

# Usage

Below a brief guide to using the modified nnUNet framework is presented which is based on the original nnUNet guide by Isensee et al. For a more detailed/insightful explanation please refer to the [original nnUNet repository](https://github.com/MIC-DKFZ/nnUNet).

Note: The code in this repository is derived from the original nnUNet repository and is identical except for the modification/addition of the following files:  `experiment_planner_residual3DUNet_v21.py`, `conv_blocks.py`, `generic_modular_custom_UNet.py`, `generic_modular_UNet.py`,`nnUNetTrainerV2_ResidualUNet.py`.


The following instructions are specific to the running of the nnUNet integrated Residual variations of the nnUNet. 

### Installation

To install, clone the git page and use pip install. Make sure latest version of PyTorch is installed. 


          git clone https://github.com/ndipenoch/nnUnet_RASPP.git
          cd nnUnet_RASPP
          pip install -e .
        

### A. Experiment Planning and Preprocessing

Ensure data is in correct format compatible with nnUNet - refer to [original nnUNet page](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for details. Furthermore paths and relevant folders must be correctly set up as shown [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

To commence experiment planning perform following steps:

##### 1) Run basic planning: 

```bash
nnUNet_plan_and_preprocess -t TASK_ID 
```

##### 2) Run planning for custom model: 

##### Residual UNet:

```bash
nnUNet_plan_and_preprocess -t TASK_ID -pl3d ExperimentPlanner3DResidualUNet_v21
```

### B. Network Training

We here concentrate on training demonstrations using the 3D full-resolution configuration for the UNet architecture variant. 

Run the following depending on which architecture one wishes to experiment training with:

##### Residual UNet:
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres CustomTrainerASPPResidual TASK_NAME_OR_ID FOLD --npz
```

Note: as discussed in the [original nnUNet repository](https://github.com/MIC-DKFZ/nnUNet), one does not have to run training on all folds for inference to run (running full training on one fold only is sufficient).


### C. Running Inference

We here concentrate on inference demonstrations using the 3D full-resolution configuration for the UNet architecture variant.

Run the following depending on which architecture one wishes to experiment inference with:

##### ASPP and Residual UNet:

```bash
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m 3d_fullres -f [0, 1, 2, 3, 4] --trainer CustomTrainerASPPResidual
```

Note: For information on network ensembling refer to [original nnUNet repository](https://github.com/MIC-DKFZ/nnUNet). 
