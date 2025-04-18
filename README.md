## Introduction
This repository contains the main source code of our benign and malignant model (deepFan).
### Prerequisites
- Ubuntu 16.04.4 LTS
- Python 3.6.13
- Pytorch 1.10.0+cu113
- NVIDIA GPU + CUDA_10.1 CuDNN_8.2
This repository has been tested on NVIDIA TITANXP. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

### Installation
Install dependencies:

```
pip install -r requirements.txt
```

#### Usage
This article mainly introduces the training and reasoning of the main framework; other individual modules can be adjusted according to the parameters of the paper.
- deepFan model：All models are saved in the model folder. Fusion integrates the ViT module, Fine_Grained module and GCN module. The specific training and reasoning are as follows.
    ```
    # train 
    python main_dfl_2scale.py config.py
    ```
    During inference, you need to modify the ‘inference_mode’ in config.py to ‘Ture’ and "resume" is set to the file path where you saved the training model:
    ```
    # inference 
    python main_dfl_2scale.py config.py
    ```
    #### The main parameters are as following:
    - --config: the path to the configuration file
    #### configuration file:
    - train task: config.py
      - inference_mode = False
      - model_name = 'fusion'
      - train_set_dir: Save the image name and label csv file path of the training data
      - val_set_dirs: Save the image name and label csv file path of the val data
      - mode_save_base_dir: Model output address
    - infer task: config.py
      - inference_mode = Ture
      - model_name = fusion
      - resume = Save the trained model path
      - save_csv: Result output
      
 data-driven approach in lung cancer prevention..
### Prerequisites
- Ubuntu 16.04.4 LTS
- Python 3.6.13
- Pytorch 1.10.0+cu113
- NVIDIA GPU + CUDA_10.1 CuDNN_8.2
This repository has been tested on NVIDIA TITANXP. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

### Installation
Install dependencies:

```
pip install -r requirements.txt
```

#### Usage
This article mainly introduces the training and reasoning of the main framework; other individual model can be adjusted according to the parameters of the paper.
![](/Users/deepwise/Library/Containers/com.tencent.xinWeChat/Data/Library/Caches/com.tencent.xinWeChat/2.0b4.0.9/03bb467d72339b4a93f3c44b074fd6c8/dragImgTmp/WeChatdae6db8f0bab7ec665f808fa0dbfe678.png)
- ScreenLungNet model：All models are saved in the model folder. The Multi-Nodule Model is implemented in the mlp.py file, and the Global Lung Model is implemented in the vit.py file.
    ```
    # train 
    python main_dfl_2scale.py config.py
    ```
    During inference, you need to modify the ‘inference_mode’ in config.py to ‘Ture’ and "resume" is set to the file path where you saved the training model:
    ```
    # inference 
    python main_dfl_2scale.py config.py
    ```
    #### The main parameters are as following:
    - --config: the path to the configuration file
    #### configuration file:
    - train task: config.py
      - inference_mode = False
      - model_name = 'fusion'
      - train_set_dir: Save the image name and label csv file path of the training data
      - val_set_dirs: Save the image name and label csv file path of the val data
      - mode_save_base_dir: Model output address
    - infer task: config.py
      - inference_mode = Ture
      - model_name = fusion
      - resume = Save the trained model path
      - save_csv: Result output
      
