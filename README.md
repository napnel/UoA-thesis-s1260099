# UoA Thesis s1260099

## Installation using docker

```
git clone https://github.com/napnel/UoA-thesis-s1260099.git
cd UoA-thesis-s1260099
docker build .
```

## Usage

```
python main.py -h
```

```
python main.py --local_dir ./ray_results --algo DQN
```

```
python summarize_performance.py --local_dir ./ray_results
```
