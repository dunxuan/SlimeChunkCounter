# Slime Chunk Counter

Count the number of Slime blocks in a given area of Minecraft (find the largest slime farm possible)

## Preview

Slime chunk number: 50, seed: 2772304346537327921, afk chunk position: (360, 436)

[Chunkbase.com:](https://www.chunkbase.com/apps/slime-finder#seed=2772304346537327921&platform=java&x=5760&z=6976&zoom=1.2)

![Chunkbase](doc/Chunkbase.png)

get_chunk_map.py:

![alt text](doc/get_chunk_map.png)

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
