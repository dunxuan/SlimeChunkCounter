# Slime Chunk Counter

Count the number of Slime blocks in a given area of Minecraft (find the largest slime farm possible)

## Usage

1. Clone repositories

```shell
git clone https://github.com/dunxuan/SlimeChunkCounter.git
cd SlimeChunkCounter
```

2. Create Conda environment

```shell
conda env create -f environment.yml --prefix "./.conda"
```

3. Run Code

```shell
conda activate ./.conda
python run.py
```

## Draw a map of a seed

```shell
python get_chunk_map.py
```