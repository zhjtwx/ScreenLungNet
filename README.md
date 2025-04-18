# ScreenLungNet
A model for predicting lung cancer risk in 3 years based on CT images
## Introduction
Lung cancer remains one of the leading causes of cancer-related mortality worldwide, largely due to its late detection and rapid progression. Early identification of individuals at high risk is critical for timely intervention and improved patient outcomes. In recent years, low-dose computed tomography (LDCT) has become a widely used screening tool for early lung cancer detection, demonstrating significant potential in reducing mortality. However, traditional radiological assessments primarily focus on identifying apparent lung nodules, often overlooking subtle patterns and features that may signal future malignancy.
With the advancement of artificial intelligence (AI) and deep learning, there has been a paradigm shift toward more comprehensive and predictive approaches in medical imaging. This study aims to develop a deep learning model that can predict an individual's risk of developing lung cancer within a three-year window, solely based on baseline CT images. By leveraging hybrid architectures, the model can capture both local and global features from chest CT scans, offering a more nuanced risk stratification than traditional methods.
The proposed model is trained and validated using large-scale, annotated datasets, incorporating both imaging biomarkers and potentially auxiliary clinical information. This predictive framework holds promise for enhancing current lung cancer screening strategies by enabling personalized surveillance plans and more targeted interventions. Ultimately, it aims to bridge the gap between early detection and risk prediction, contributing to a more proactive and data-driven approach in lung cancer prevention..
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
      
