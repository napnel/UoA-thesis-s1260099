@echo off
setlocal
python main.py --logdir ./ray_results/compare-algo-11 --stop-timesteps 500000 --algo DQN
python main.py --logdir ./ray_results/compare-algo-11 --stop-timesteps 500000 --algo A2C
python main.py --logdir ./ray_results/compare-algo-11 --stop-timesteps 500000 --algo PPO
python main.py --logdir ./ray_results/compare-algo-11 --stop-timesteps 500000 --algo SAC