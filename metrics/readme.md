# GIPHY Celebrity Detector Installation Guide

For the evaluation of the celebrity eraser, we utilized the GIPHY Celebrity Detector [official installation instructions](https://github.com/Giphy/celeb-detection-oss/tree/master/examples). However, since the latest version of the GIPHY detector is somewhat outdated, certain issues may arise. If you encounter any difficulties or fail to obtain results consistent with those reported in our paper, please follow the instructions provided on this page. 
We thank the following contributors that our code is based on: [MACE](https://github.com/Shilin-LU/MACE?tab=readme-ov-file).


## Installation Steps

1. Clone the repository:

```
cd metrics
git clone https://github.com/Giphy/celeb-detection-oss.git
mv celeb-detection-oss giphy
cd giphy
conda create -n giphy python=3.6
conda activate giphy
```

2. Modify the `setup.py` file and requirements_cpu.txt:
- replace `x.req` with `x.requirement` (line 37) in the `metrics/giphy/setup.py` file
- Comment out numpy `numpy==1.15.1` (line 8) and `torch==0.4.1` (last line) in the `requirements_cpu.txt` file


3. install python libraries:

```
pip install -r requirements_gpu.txt
pip install -e .
cd examples
cp .env.example .env
pip install imageio==2.4.1 pandas
pip install --upgrade scikit-image
```

4. Download the  latest archive with Giphy pretrained models (resources):

```
# cd examples
mv resources backup_resources
python download_model.py
```

5. Modify the face detection network configuration
 - Open `metrics/giphy/model_training/preprocessors/face_detection/network.py`
 - Modify to include `allow_pickle=True` (line 88)
 
   ```
   data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()
   ```
