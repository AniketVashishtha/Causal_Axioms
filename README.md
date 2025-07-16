# Teaching Transformers Causal Reasoning Through Axiomatic Training

Teaching transformers to apply causal axioms, by training them on synthetic data constructed using causal rules and axioms like Transitivity and D-Separation! 

Check out the paper: https://openreview.net/pdf?id=AhebPqDOMI 

![image](https://github.com/user-attachments/assets/e1d503a1-9794-4f72-9959-5d49dc58be9c)


This repository contains the training and evaluation data for transitivity and d-separation rules of causality.

## Axiomatic Pre-Training

### Setup

Tested on Python 3.11.12 and Pytorch 2.7.0

We recomend using a virtual environment to install the dependencies. E.g. you can use a `conda` environment.

```bash
conda create -n causal_axioms python=3.11
conda activate causal_axioms
conda install pip
```

**Install our fork of the Transformers Library**

```bash
cd transformers
pip install -e .
cd ..
```

**Install rest of the dependencies**
```bash
pip install -r requirements.txt
```

### Run training and evaluation

**For Transitivity**

```bash
bash axiomatic_training/run_transitivity.sh
```

**For D-Separation**

```bash
bash axiomatic_training/run_dsep.sh
```

Note: Run both the scripts from the root directory of the repository.
