# UoA Thesis s1260099

This repository is the program code for the thesis of an undergraduate student at the University of Aizu.

The paper is [An Evaluation on Deep Reinforcement Learning Algorithms for Stock Trading](paper/s1260099.pdf).

## Installation using docker (Recommended)

```
git clone https://github.com/napnel/UoA-thesis-s1260099.git && cd UoA-thesis-s1260099
docker build .
```

## Other installation

Please use Ubuntu as the OS; Rllib does not officially support windows.

```
git clone https://github.com/napnel/UoA-thesis-s1260099.git && cd UoA-thesis-s1260099
conda crate --name <env_name> python==3.8
codna activate <env_name>
pip install -r requirements.txt
```

## Usage

```
python main.py -h
```

## Running example

```
python main.py --local_dir ./ray_results --algo DQN --ticker ^N225
```

This example shows a Deep Q-Network learning on the Nikkei Stock Average.

The execution result is . /ray_results, and you can see the agent's behavior visualized in backtest.html.

If you want to reduce the computation time, increase the --num_workers or decrease the --num_samples.

```
python summarize_performance.py --local_dir ./ray_results
```

This run summarizes the computation time, learning curve, and investment performance of all the agents stored in ray_results.
