
## A platform for comparing segmentation results of different architectures.

### Change log


### Setup
python version: 3.7.11

Creating conda environment:
```shell
conda create --name [env_name] python=3.7.11
```

CUDA version: 11.1
PyTorch version: 1.8.0

PyTorch installation command:
```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Installing dependencies:
```shell
pip install -r requirements.txt
```

### Directory structure

```shell
.
├── smain.py
├── recon.py
├── README.md
├── requirements.txt
├── utils.py
├── models
└── data
    └── segzeiss
        └── train
            ├── img0.jpg
            ├── img1.jpg
            └── ...
        └── val
            ├── img0.jpg
            ├── img1.jpg
            └── ...
        └── test
            ├── img0.jpg
            ├── img1.jpg
            └── ...            
└── figures
    ├── result0.png
    ├── result1.png
    └── ...
```





