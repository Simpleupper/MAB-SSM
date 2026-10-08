# MAB-SSM

### Installation
Install via pip (Recommended)

```bash
#1. Create and activate the environment
conda create -n mab-ssm python=3.9 -y
conda activate mab-ssm

#2. Clone the MAB-SSM repository
git clone https://github.com/Simpleupper/MAB-SSM.git
cd MAB-SSM

#3. Install PyTorch for the CUDA version available on your system
See: https://pytorch.org/get-started/locally/

#4. Install the principal dependencies
pip install numpy nnunetv2 dynamic-network-architectures mamba-ssm

#5. Install the 3D wavelet-transform dependency
git clone https://github.com/KeKsBoTer/torch-dwt.git
pip install -e ./torch-dwt
```

For reproducibility, it is recommended to record the exact PyTorch, CUDA, mamba-ssm, nnunetv2, and dynamic-network-architectures versions used in the local environment.

### Network Construction with nnU-Net v2
The recommended entry point is get_umamba_bot_3d_from_plans, which builds the model from an nnU-Net v2 configuration

```python
from MAB_SSM_net import get_umamba_bot_3d_from_plans

network = get_umamba_bot_3d_from_plans(
    plans_manager=plans_manager,
    dataset_json=dataset_json,
    configuration_manager=configuration_manager,
    num_input_channels=num_input_channels,
    deep_supervision=True,
)
```

### Standalone Model Initialization
The network can also be instantiated directly for development or shape verification

```python
import torch
from torch import nn
from MAB_SSM_net import MABSSM

model = MABSSM(
    input_channels=1,
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 320],
    conv_op=nn.Conv3d,
    kernel_sizes=[(3, 3, 3)] * 5,
    strides=[
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    ],
    n_conv_per_stage=[2, 2, 2, 2, 2],
    num_classes=2,
    n_conv_per_stage_decoder=[2, 2, 2, 2],
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-5, "affine": True},
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
    deep_supervision=False,
)

model.eval()
x = torch.randn(1, 1, 128, 128, 128)

with torch.no_grad():
    y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
```

### Training and Validation
MAB_SSM_net.py defines the network architecture and the nnU-Net model-construction interface. Dataset conversion, preprocessing, planning, training, validation, and inference should be performed through an nnU-Net v2 trainer that imports this network builder

```bash
# Dataset fingerprint extraction and experiment planning
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train the registered MAB-SSM trainer
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr TRAINER_NAME

# Predict a test set
nnUNetv2_predict \
  -i INPUT_FOLDER \
  -o OUTPUT_FOLDER \
  -d DATASET_ID \
  -c 3d_fullres \
  -f FOLD \
  -tr TRAINER_NAME
```
⭐ **If you find this work useful, please star the repository!**
