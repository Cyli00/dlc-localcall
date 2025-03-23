# dlc-localcall
防止每次分析模型时重复访问huggingface下载模型。

# 安装miniforge并且配置环境变量
[conda-forge/miniforge: A conda-forge distribution.](https://github.com/conda-forge/miniforge)
*推荐使用 wsl2*

# 安装cuda并且配置环境变量
[CUDA Toolkit 12.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
*wsl2 推荐使用 runfile，安装完毕后根据指令在 .bashrc 或者 .zshrc 配置环境变量*

# 安装瞳孔识别工具包
## 创建 dlc 环境
`mamba create -n dlc python=3.10`

## 安装 dlc tensorflow version
```undefined
mamba activate dlc
mamba install deeplabcut[tf,modelzoo] # 或者 mamba install "deeplabcut[tf,modelzoo]
```
