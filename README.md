# On Uncertainty Calibration for Equivariant Functions

This repository contains all code for `On The Uncertainty Calibration of Equivariant Functions` by [Edward Berman](https://ebrmn.space/) and [Jacob Ginesin](https://jakegines.in/), advised by [Robin Walters](https://www.robinwalters.com/). This work is a product of our Mathematics Research Capstone course (Math 4020) at Northeastern University. 

![image](assets/tictactoe.png)

# Running Experiments

First, clone the repo (and the submodules!) via

`git clone --recurse-submodules https://github.com/EdwardBerman/EquivariantUQBounds.git`

## Moons

For the moons experiment / example, first make the dataset with 

`python make_dataset.py`

There are optional keywords 

```py
degrees: int = 90
n_samples: int = 1000
noise: float = 0.1
```

Then run `train_MLP.py` and `train_equivariant.py` to train the models

## Swiss Roll

Instructions are in the `ext_theory` Git submodule on how to run the "spiral" experiment

## Galaxy Morphology

Instructions are in the `SIDDA` Git submodule on how to run, including a script for producing the noised version of the dataset and a zenodo link to the default .npy files

## Chemical Properties

Run the `run.py` script in `/uq_bleading/`
