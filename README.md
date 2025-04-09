# A simulation-heuristics dual-process model for intuitive physics

<p align="left">
    <a href='https://drive.google.com/file/d/1R6x_3L0DQFJ3HU0yYvjkhPGKA94tzNok/view?usp=sharing'>
    <img src='https://img.shields.io/badge/Data-GoogleDrive-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Data'>
    </a>
    <a href='https://vimeo.com/1073941780'>
      <img src='https://img.shields.io/badge/Demo-Vimeo-red?style=plastic&logo=Vimeo&logoColor=red' alt='Demo'>
    </a>
</p>

## Project structure
```
Dual_model
├── model
│   ├── simulation
│   │   ├── results           # IPE results
│   │   └── IPE.py            # Running intuitive physics engine
│   └── model_compare.py      # Model comparision script
├── pouring
│   ├── generate_diverse.py   # Data simulation script
│   ├── make_videos.py        # Video making tools
│   ├── simulator.py          # Pouring simulator
│   └── tool.py               # Other tools
├── stimuli
│   ├── round1-1
│   ├── round1-2
│   └── round1-3
├── config_diverse.yml        # Data simulation configs
├── data.csv                  # Human data
└── README.md
```
## Getting started

```
conda create -n dual python=3.9
pip install pymunk pygame
pip install opencv-python
pip install matplotlib pandas numpy
pip instal statsmodels
```

## Dowload pouring dataset
Download the pouring-marble dataset from this [link](https://drive.google.com/file/d/1R6x_3L0DQFJ3HU0yYvjkhPGKA94tzNok/view?usp=sharing). The human results can be found in `data.csv`.

## Generate pouring dataset

```
python ./pouring/generate_diverse.py
```

## Run IPE

We implement a multi-process IPE to perform simulation under noise.

```
python ./model/simulation/IPE.py
```

## Compare models
Compare models including Deterministic physics, IPE, Heuristic model, and our SHM.

```
python ./model/model_compare.py
```