#!/usr/bin/env python

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Build the navigation map for the given datasets"
    )
    parser.add_argument(
        "datasets", nargs="+", help="Datasets path. Can be passed one or more datasets"
    )
    parser.add_argument("imageWidth", type=int, help="Image width")
    parser.add_argument("imageHeight", type=int, help="Image height")
    parser.add_argument("navigationWidth", type=int, help="Navigation width")
    parser.add_argument("navigationHeight", type=int, help="Navigation height")
    parser.add_argument("neighborhood", type=int, help="Neighborhood")
    parser.add_argument(
        "destination", type=str, help="Filename to where to save the navigation map"
    )
    args = parser.parse_args()

    image_size = [args.imageHeight, args.imageWidth]
    navigation_size = [args.navigationHeight, args.navigationWidth]

    navigation_map = make_navigation_map(
        args.datasets, image_size, navigation_size, args.neighborhood
    )

    np.save(args.destination, navigation_map)


def make_navigation_map(datasets, image_size, navigation_size, neighborhood):
    """Build the navigation map for the given dataset.

    Args:
      datasets: list. Datasets.
      image_size: list. Height and width of the image.
      navigation_size: list. Height and width of the navigation map.
      neighborhood: int. Neighborhood size

    Returns:
      ndarray containing the navigation map.

    """
    navigation_map = np.zeros(navigation_size[0] * navigation_size[1])

    for dataset_path in datasets:
        dataset = np.loadtxt(dataset_path, delimiter="\t")

        top_left = [
            np.floor(min(dataset[:, 2]) - neighborhood / 2),
            np.ceil(max(dataset[:, 3]) + neighborhood / 2),
        ]
        cell_x = np.floor(
            ((dataset[:, 2] - top_left[0]) / image_size[1]) * navigation_size[1]
        )
        cell_y = np.floor(
            ((top_left[1] - dataset[:, 3]) / image_size[0]) * navigation_size[0]
        )
        grid_pos = cell_x + cell_y * navigation_size[1]
        # For each cell, counts the pedestrian  only once
        grid_pos = np.stack([dataset[:, 1], grid_pos], axis=1)
        grid_pos = np.unique(grid_pos, axis=0)
        grid_pos = grid_pos[:, 1].astype(int)
        np.add.at(navigation_map, grid_pos, 1)

    # Normalize in [0,1]
    max_norm = max(navigation_map)
    navigation_map = navigation_map / max_norm
    navigation_map = np.reshape(
        navigation_map, [navigation_size[0], navigation_size[1]]
    )

    return navigation_map


if __name__ == "__main__":
    main()
