# On Uncertainty Calibration for Equivariant Functions

This repository contains all code for `On Uncertainty Calibration for Equivariant Functions` by [Edward Berman](https://ebrmn.space/) and [Jacob Ginesin](https://jakegines.in/), advised by [Robin Walters](https://www.robinwalters.com/). This work is a product of our Mathematics Research Capstone course (Math 4020) at Northeastern University. 

![image](assets/tictactoe.png)

# Running Experiments

First, clone the repo (and the submodules!) via

`git clone --recurse-submodules https://github.com/EdwardBerman/EquivariantUQBounds.git`

## Vector Field Regression

```
# Generate a dataset (or use an existing one)
python make_vector_field_dataset.py --dataset_type spiral --n_samples 2000 --noise 0.1 --visualize True

# Run the full experiment
python train_vector_field.py --dataset_type spiral --n_samples 2000 --noise 0.1 --hidden_dim 32 --maximum_epochs 100
```


## Swiss Roll

Instructions are in the `ext_theory` Git submodule on how to run the "spiral" experiment

## Galaxy Morphology

Instructions are in the `SIDDA` Git submodule on how to run, including a script for producing the noised version of the dataset and a zenodo link to the default .npy files

## Chemical Properties

Run the `run.py` script in `/uq_bleading/`
