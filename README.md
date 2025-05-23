# dlc-localcall
Prevent repeated access to Hugging Face for model downloads during each model analysis.

# Install Miniforge and Configure Environment Variables
[conda-forge/miniforge: A conda-forge distribution.](https://github.com/conda-forge/miniforge)
*Recommended to use WSL2*

# Install CUDA and Configure Environment Variables
[CUDA Toolkit 12.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
*For WSL2, it is recommended to use the runfile installer. After installation, follow the instructions to configure environment variables in your `.bashrc` or `.zshrc`.*

# Install Pupil Detection Toolkit
## Create the dlc Environment
```bash
mamba create -n dlc python=3.10
```

## Install dlc TensorFlow Version
```bash
mamba activate dlc
mamba install deeplabcut[tf,modelzoo] # or mamba install "deeplabcut[tf,modelzoo]"
```
