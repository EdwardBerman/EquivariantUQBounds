from sklearn import datasets
import numpy as np
import os
from dataclasses import dataclass
from simple_parsing import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    @dataclass
    class Options:
        degrees: int = 90
        n_samples: int = 1000
        noise: float = 0.1

    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()

    n_samples = args.options.n_samples
    degrees = args.options.degrees
    noise = args.options.noise

    input_locations, _ = datasets.make_moons(n_samples=n_samples, noise=noise)

    rotation_theta_degrees = 90
    rotation_matrix = np.array([[np.cos(rotation_theta_degrees), -np.sin(rotation_theta_degrees)],
                                [np.sin(rotation_theta_degrees), np.cos(rotation_theta_degrees)]])

    output_locations = np.array([np.dot(rotation_matrix, input_location) for input_location in input_locations])

    if not os.path.exists('../data'):
        os.makedirs('../data')

    np.save(f'../data/input_locations_{degrees}_degree_offset_{n_samples}_samples_{noise}_noise.npy', input_locations)
    np.save(f'../data/output_locations_{degrees}_degree_offset_{n_samples}_samples_{noise}_noise.npy', output_locations)
