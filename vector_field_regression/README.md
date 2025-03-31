# Equivariant Vector Field Regression with Uncertainty Estimation
Investigates how equivariance properties affect the quality and calibration of uncertainty estimates in vector field regression tasks. 

# Datasets

Features several types of available datasets to generate:
- `grid` generates points in a square
- `spiral` generates vectors in a spiral
- `spiral-fixed` generates vectors tangent to a circle with a fixed radius of 1
- `rotational` generates vectors pointing towards (0,0) with a length dependent on their distance from (0,0)
- `sine` generates vectors pointing towards (0,0) with a length dependent on the sine of the norm of the length from (0,0)
- `divergent` generates vectors pointing away from (0,0) with a length dependent on their distance from (0,0)

```bash
# generate an example dataset
python make_vector_field_dataset.py --dataset_type [type] --n_samples 2000 --noise 0.1 --visualize True
```
(save-dir is modifiable in the file itself)

# Training models
to train models, we can do the following:
```bash
python train_vector_field.py --dataset_type [type] --n_samples 2000 --noise 0.1 --hidden_dim 32 --maximum_epochs 100
```
(save-dir is modifiable in the file itself)

If we want to skip training and use the existing results files to output graphs, we can append `--skip-training`. 

