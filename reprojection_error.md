## Environment Setup

```bash
conda create -n common_metrics python=3.10 && conda activate common_metrics

# install necessary packages
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install --index-url https://download.pytorch.org/whl/cu121 xformers 
conda install suitesparse -c conda-forge
pip install open3d tensorboard scipy opencv-python tqdm matplotlib pyyaml opencv-python decord imageio pathlib
pip install evo --upgrade --no-binary evo
pip install gdown

# install droid-backend
git submodule update --init --recursive 
cd reprojection_error/DROID-SLAM
python setup.py install
cd ../..
```

## Usage

```bash
python calculate_reprojection_error.py
```

## Visualization Droid-SLAM Reconstruction

```bash
 python reprojection_error/view_recon.py reconstructions/000000_recon.pth
```

Note: You need GUI display environment to visualize the reconstruction. If you are using a remote server, you can use mobaXterm or X11 forwarding to enable GUI display.