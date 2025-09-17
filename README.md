### Installation

```
conda create --name rku_tal python=3.8 -y
conda activate rku_tal
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch  # **This** command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
mim install mmdet
pip install einops
pip install numpy==1.23.5
```

