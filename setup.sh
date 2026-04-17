conda create -n geo-aware python=3.9 pip=22.3.1
conda activate geo-aware
conda install gxx=11.4.0 -c conda-forge
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
pip install -e .
pip install jupyter xformers==0.0.16 git+https://github.com/facebookresearch/segment-anything.git