# Benchmark installation



<br>

## 1. Setup Conda

```
# Conda installation

# For Linux
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For OSX
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

chmod +x ~/miniconda.sh    
./miniconda.sh  

source ~/.bashrc          # For Linux
source ~/.bash_profile    # For OSX
```


<br>

## 2. Setup Python environment for CPU

```
# Clone GitHub repo
conda install git
git clone https://github.com/ChaofanTao/litegt.git
cd litegt

# Install python environment
conda env create -f environment_cpu.yml   

# Activate environment
conda activate graph_transformer
```

X 

<br>

## 3. Setup Python environment for GPU

NOTE: If you have downloaded gpu-version pytorch before, you just need to install some small libraries.
```
pip install dgl tqdm numpy scikit-learn scikit-image networkx tensorboard tensorboardx
```
Make sure that your CUDA, [dgl](https://www.dgl.ai/pages/start.html), [pytorch](https://pytorch.org/) use compatible versions. Use ```cat /usr/local/cuda/version.txt``` to check CUDA version. We use ```CUDA=11.0, dgl=0.6.0.post1, pytorch=1.7.0+cu110``` and run on NVIDIA-3090 cards.



The following shows that how to build env from scratch.
DGL 0.5.x requires CUDA **10.2**.

For Ubuntu **18.04**

```
# Setup CUDA 10.2 on Ubuntu 18.04
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y cuda-10-2
sudo reboot
cat /usr/local/cuda/version.txt # Check CUDA version is 10.2

# Clone GitHub repo
conda install git
git clone https://github.com/ChaofanTao/litegt.git
cd litegt

# Install python environment
conda env create -f environment_gpu.yml 

# Activate environment
conda activate graph_transformer
```






<br><br><br>
