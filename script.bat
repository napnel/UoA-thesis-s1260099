@echo off
setlocal
python expt_cv_class_api.py --logdir ./ray_results/compare-algo-5 --max_timesteps 50000 --algo DQN
python expt_cv_class_api.py --logdir ./ray_results/compare-algo-5 --max_timesteps 50000 --algo PPO
python expt_cv_class_api.py --logdir ./ray_results/compare-algo-4 --max_timesteps 50000 --algo A2C