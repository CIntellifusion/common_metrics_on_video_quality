## Environment Setup

```bash
conda create -n common_metrics python=3.10 && conda activate common_metrics

# install necessary packages
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install --index-url https://download.pytorch.org/whl/cu121 xformers 
conda install suitesparse -c conda-forge
pip install open3d tensorboard scipy opencv-python tqdm matplotlib pyyaml opencv-python decord imageio
pip install evo --upgrade --no-binary evo
pip install gdown

# install droid-backend
git submodule update --init reprojection_error/DROID-SLAM --recursive 
cd reprojection_error/Droid-SLAM
python setup.py install
cd ../..
```

## Usage

```bash
python calculate_reprojection_error.py
```