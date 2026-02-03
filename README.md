# Bit Flip Attack on Privacy-Sensitive Data

This repository contains implementations of bit flip attacks targeting models trained on privacy-sensitive data, such as medical imaging and facial recognition datasets.

## Overview

Bit flip attacks are a form of fault injection attack that directly manipulates model parameters at the binary level. By flipping just a few bits in model weights, these attacks can cause models to:

1. Leak private information
2. Reduce accuracy on specific tasks
3. Create backdoors in model behavior
4. Bypass privacy protections

* This project was ran on a NVIDIA RTX4090 GPU ,recommended for best results connecting via ssh and running on the GPU  (eg. runpod)

## Quick Start

```bash
## clone and cd into the repo 

cd bfa_cleaned


## install uv package manager (faster than pip) create venv, istall dependencies

pip install uv

uv venv 


source venv/bin/activate


uv pip install -r requirements.txt


```

2. Run chosen  attacl:
```bash
python celeba_face_identification_attack.py

python lfw_face_identification_attack_V3.py

python lfw_face_attack_V1.py

python   medical_imaging_attack.py
```


## Create advanced plots after saving results to ´resuls/´ dir


```bash

    python create_publication_plots.py  /results/INSERT_PATH/

```




## License


MIT 


