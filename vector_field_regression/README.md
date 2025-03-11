# Equivariant Vector Field Regression with Uncertainty Estimation

investigates how equivariance properties affect the quality and calibration of uncertainty estimates in vector field regression tasks

## Running the Experiment

```bash
# Generate a dataset (or use an existing one)
python make_vector_field_dataset.py --dataset_type spiral --n_samples 2000 --noise 0.1 --visualize True

# Run the full experiment
python train_vector_field.py --dataset_type spiral --n_samples 2000 --noise 0.1 --hidden_dim 32 --maximum_epochs 100
```
