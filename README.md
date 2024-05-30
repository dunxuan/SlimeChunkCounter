# Slime Chunk Counter


## Usage

1. Clone repositories

```shell
git clone https://github.com/dunxuan/SlimeChunkCounter.git
cd SlimeChunkCounter
```

2. Create Conda environment

```shell
conda create -n .conda
```

3. Install packages

```shell
conda install cudatoolkit=11.8 cudnn pytorch torchvision torchaudio pytorch-cuda=11.8  -c pytorch -c nvidia
pip install 'git+https://github.com/MostAwesomeDude/java-random.git'
```

4. Run Code

```shell
conda activate .conda
python3 run.py
```