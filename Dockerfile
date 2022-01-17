FROM rayproject/ray:1.9.2-py38-cpu

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    curl \
    && apt-get clean

RUN pip install --no-cache-dir -U pip \
    autorom[accept-rom-license] \
    gym[atari] \
    scikit-image \
    tensorflow \
    lz4 \
    pytest-timeout \
    smart_open \
    tensorflow_probability \
    dm_tree \
    h5py   # Mutes FutureWarnings \
    bayesian-optimization \
    hyperopt \
    ConfigSpace==0.4.10 \
    torch \
    torchvision \
    tensorboardX \
    dragonfly-opt \
    zoopt \
    tabulate \ 
    mlflow \
    pytest-remotedata>=0.3.1 \
    matplotlib \
    jupyter \
    pandas \
    yfinance \
    empyrical \
    scikit-learn \
    ta \
    backtesting \
    minepy==1.0.0 \
    SciencePlots \
    dill

COPY . /home/code
WORKDIR /home/code