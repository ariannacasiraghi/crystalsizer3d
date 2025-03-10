import hashlib
import json
import os
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision.transforms.functional import center_crop, to_tensor

from crystalsizer3d import DATA_PATH, LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.sequence.utils import get_image_paths, load_manual_measurements
from crystalsizer3d.util.image_scale import get_pixels_to_mm_scale_factor
from crystalsizer3d.util.utils import init_tensor

model_dirs = DATA_PATH / 'predictor_training_results'

models = {
    'RegNetZ': {
        'id': '9c4a370fc35a3a2960f5c20c7d85824b',
        'n_params': 21.7,
    },
    'RegNetX': {
        'id': '109442a24724b918d4968e7cc59e8122',
        'n_params': 6.63,
    },
    'RegNetY': {
        'id': '164a4c1f2189c22b3a6f109872b7c6a1',
        'n_params': 80.76,
    },
    'EfficientNet-B3': {
        'id': '8ea7c2b2ef328dd8a6715fa9125d8162',
        'n_params': 10.79,
    },
    'EfficientNet-B4': {
        'id': 'a49f4616a685d110debbc933980d2263',
        'n_params': 17.66,
    },
    'ECA-ResNet-D': {
        'id': '61695949ed261c8f57f3d88564f2c8a5',
        'n_params': 100.18,
    },
    'ViT-Small': {
        'id': 'ea0ab9ec62ef9b092521898efc4a5555',
        'n_params': 21.86,
    },
    'ViT-Base': {
        'id': '7d05459a51a4795e3c8a2135f30c2a83',
        'n_params': 86.18,
    },
    'ViT-RelPos': {
        'id': '0d7f8d907ca42d1b3d3259265e9fe5aa',
        'n_params': 118.58,
    },
    'MobileNet-V4-Small': {
        'id': 'f8d33efad1be8f366c0da9017ad9e162',
        'n_params': 2.58,
    },
    'MobileNet-V4-Large': {
        'id': 'e76b2101914c9effe1786e60b9e18cec',
        'n_params': 31.39,
    },
}


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to compare one-shot prediction models.')

    parser.add_argument('--images-dir', type=Path,
                        help='Path to the images directory.')
    parser.add_argument('--measurements-dir', type=Path,
                        help='Path to the manual measurements.')
    parser.add_argument('--pixel-to-mm', type=float, default=None,
                        help='Pixels to millimeters conversion factor. If not provided, it will be calculated dynamically.')
    parser.add_argument('--initial-scene', type=Path,
                        help='Path to an initial scene file.')
    args = parser.parse_args()

    return args


def calculate_directory_checksum(dir_path):
    sha1 = hashlib.sha1()

    for root, dirs, files in os.walk(dir_path):
        for names in sorted(files):
            filepath = os.path.join(root, names)
            if filepath[-5:] == '.json':
                continue

            # Gathering file attributes
            file_stats = os.stat(filepath)
            sha1.update(names.encode('utf-8'))
            sha1.update(str(file_stats.st_size).encode('utf-8'))
            sha1.update(str(file_stats.st_mtime).encode('utf-8'))

    return sha1.hexdigest()


def read_tensorboard_data(logdir):
    scalars_combined = defaultdict(list)
    events_dir = logdir / 'events'
    checksum = calculate_directory_checksum(events_dir)
    cache_file = logdir / (checksum + '.json')

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Traverse through each run directory
    for dirpath, dirnames, filenames in os.walk(events_dir):
        for filename in filenames:
            if filename.startswith('events.out.tfevents'):
                event_file = os.path.join(dirpath, filename)
                logger.info(f'Processing event file: {event_file}')
                event_acc = EventAccumulator(event_file)
                event_acc.Reload()
                scalar_tags = event_acc.Tags().get('scalars', [])
                for tag in scalar_tags:
                    scalar_events = event_acc.Scalars(tag)
                    for event in scalar_events:
                        scalars_combined[tag].append((event.step, event.wall_time, event.value))

    # Sort the lists by the step value for each tag
    for tag in scalars_combined:
        scalars_combined[tag].sort(key=lambda x: x[0])

    # Write cache
    with open(cache_file, 'w') as f:
        json.dump(scalars_combined, f)

    return scalars_combined


def plot_model_comparison():
    """
    Make a plot of different models and their losses.
    """
    save_dir = LOGS_PATH / 'comparisons'
    save_dir.mkdir(parents=True, exist_ok=True)
    nth_epochs = 5
    loss_keys = [
        ('epoch/test/losses/distances', 'Distances loss'),
        ('epoch/test/losses/3d', '3D loss'),
        ('epoch/test/losses/material', 'Material loss'),
        ('epoch/test/losses/light', 'Light loss')]

    # Load the tensorboard data
    for model_name, model_spec in models.items():
        model_spec['scalars'] = read_tensorboard_data(model_dirs / model_spec['id'])

    def plot_losses(ax, loss_key, title, x_axis='epoch'):
        ax.grid(True, which='both', ls='--')

        for model_name, model_spec in models.items():
            scalars = model_spec['scalars']
            steps, wall_times, losses = zip(*scalars[loss_key])

            # Convert wall times from seconds to hours elapsed
            start_time = min(wall_times)
            elapsed_hours = [(wt - start_time) / 3600 for wt in wall_times]

            # Select every nth epoch
            selected_points = list(range(nth_epochs - 1, len(steps), nth_epochs))

            # Plot every point, with markers for the selected epochs
            x_vals = np.array(elapsed_hours if x_axis == 'time' else steps)[selected_points]
            y_vals = np.array(losses)[selected_points]
            ax.plot(x_vals, y_vals,
                    marker='o',
                    markersize=model_spec['n_params'] / 10,
                    label=f'{model_name} - ({model_spec["n_params"]:.2f}M)',
                    alpha=0.7)

        ax.set_yscale('log')
        if x_axis == 'time':
            ax.set_xlabel('Elapsed Time (hours)')
        else:
            ax.set_xlabel('Epoch')
        # ax.set_ylabel(loss_name)
        if title:
            ax.set_title(f'{title}')

    def plot_loss_against_size(ax, loss_key, loss_name):
        xs, ys = [], []
        ax.grid(True, which='both', ls='--')

        for model_name, model_spec in models.items():
            scalars = model_spec['scalars']
            steps, wall_times, losses = zip(*scalars[loss_key])
            x, y = model_spec['n_params'], losses[-1]
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y,
                       s=model_spec['n_params'] * 10,
                       label=f'{model_name} ({model_spec["n_params"]:.2f}M)',
                       alpha=0.7)

        xs = np.array(xs)
        ys = np.array(ys)
        x_mid = np.ptp(xs) / 2 + np.min(xs)
        y_mid = np.ptp(ys) / 2 + np.min(ys)
        for i, (model_name, model_spec) in enumerate(models.items()):
            x, y = xs[i], ys[i]
            va = 'bottom' if y < y_mid else 'top'
            ha = 'right' if x > x_mid else 'left'
            plt.text(x, y, f'{model_name} ({model_spec["n_params"]:.2f}M)', fontsize=9,
                     verticalalignment=va, horizontalalignment=ha)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of parameters (M)')
        ax.set_ylabel('Final test loss')
        ax.set_title(loss_name)

    # Set font sizes
    plt.rc('axes', titlesize=11, titlepad=6)  # fontsize of the title
    plt.rc('axes', labelsize=9, labelpad=4)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=7)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=7)  # fontsize of the legend
    plt.rc('xtick.major', pad=1, size=1)
    plt.rc('ytick.major', pad=1, size=1)

    gs = GridSpec(
        nrows=4,
        ncols=4,
        wspace=0.4,
        hspace=0.5,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[0.8, 0.8, 1, 1],
        top=0.97,
        bottom=0.06,
        left=0.08,
        right=0.99
    )
    fig = plt.figure(figsize=(7.5, 7.8))

    ax = fig.add_subplot(gs[0:2, 0:2])
    plot_losses(ax, 'epoch/test/total', 'Total loss', x_axis='epoch')
    ax.xaxis.set_tick_params(pad=2, size=2)
    ax.yaxis.set_tick_params(pad=2, size=2)

    ax = fig.add_subplot(gs[0:2, 2:])
    plot_loss_against_size(ax, 'epoch/test/total', 'Total loss')
    ax.xaxis.set_tick_params(pad=2, size=2)
    ax.yaxis.set_tick_params(pad=2, size=2)

    for col_idx, (loss_key, loss_name) in enumerate(loss_keys):
        ax = fig.add_subplot(gs[2, col_idx])
        plot_losses(ax, loss_key, loss_name, x_axis='epoch')
        ax = fig.add_subplot(gs[3, col_idx])
        plot_losses(ax, loss_key, loss_name, x_axis='time')

    # Save the figure
    plt.savefig(save_dir / f'{START_TIMESTAMP}_model_comparison.svg', transparent=True)
    plt.show()

    # Save the loss data for showing in a table in the paper
    target_epoch = 40
    data = {}
    for model_name, model_spec in models.items():
        scalars = model_spec['scalars']
        steps, wall_times, losses = zip(*scalars['epoch/test/total'])
        idx = np.argmin(np.abs(np.array(steps) - target_epoch))
        elapsed_hours = [(wt - min(wall_times)) / 3600 for wt in wall_times][idx]
        data[model_name] = {'epoch': steps[idx], 'elapsed_hours': elapsed_hours, 'losses': {'total': losses[idx]}}
        for loss_key, loss_name in loss_keys:
            _, _, losses = zip(*scalars[loss_key])
            data[model_name]['losses'][loss_key] = f'{losses[idx]:.4f}'
    with open(save_dir / f'{START_TIMESTAMP}_model_comparison.json', 'w') as f:
        json.dump(data, f, indent=2)


def calculate_measurement_errors():
    """
    Calculate the measurement errors for different predictors.
    """
    args = get_args()
    save_dir = LOGS_PATH / 'measurement_errors'
    save_dir.mkdir(parents=True, exist_ok=True)
    oversize_input = False
    resize_args = dict(mode='bilinear', align_corners=False)
    img_size = 512
    manager = None

    def get_model_path(model_key):
        model_id = models[model_key]['id']
        files = sorted(model_dirs.glob(f'{model_id}_*.json'), reverse=True)
        if not files:
            raise FileNotFoundError(f'No model files found for key={model_key} and id={model_id} in {model_dirs}')
        return files[0]

    def load_manager(model_key):
        nonlocal manager, img_size
        manager = Manager.load(
            model_path=get_model_path(model_key),
            args_changes={
                'runtime_args': {
                    'use_gpu': USE_CUDA,
                    'batch_size': 1,
                },
            },
            save_dir=save_dir,
            path_substitutions=[
                ('/home/home02/scsti', '/mnt/arc4'),
            ]
        )

        # Set up the resizing
        manager.predictor.resize_input = False
        img_size = 512 if oversize_input else manager.image_shape[-1]

    # Load the measurements
    load_manager('RegNetZ')
    measurements = load_manual_measurements(
        measurements_dir=args.measurements_dir,
        manager=manager
    )
    measurement_idxs = measurements['idx']

    # Load the image paths
    sf_args = SequenceFitterArgs(images_dir=args.images_dir)
    image_paths = get_image_paths(sf_args, load_all=True)
    image_paths = [p[1] for p in image_paths if p[0] in measurement_idxs]

    # Get the pixel-to-mm scale factor
    dist_to_mm = get_pixels_to_mm_scale_factor(args.initial_scene, image_paths[0], args.pixel_to_mm)

    # Clone the base crystal
    scene0 = Scene.from_yml(args.initial_scene)
    crystal = scene0.crystal.clone()
    crystal.to('cpu')

    # Cache the target vertices
    vertices_targets = {}
    for i in range(len(image_paths)):
        crystal.build_mesh(
            scale=init_tensor(measurements['scale'][i]),
            distances=init_tensor(measurements['distances'][i]),
            origin=init_tensor(measurements['origin'][i]),
            rotation=init_tensor(measurements['rotation'][i]),
        )
        vertices_targets[i] = crystal.vertices.clone().detach()

    # Set up loss outputs
    errors_d = {k: np.zeros(len(image_paths)) for k in models.keys()}
    errors_v = {k: np.zeros(len(image_paths)) for k in models.keys()}

    def calculate_prediction_errors(model_key, X_path, idx_m):
        # Resize the input if needed
        X = to_tensor(Image.open(X_path)).to(manager.device)[None, ...]
        if X.shape[-2] != X.shape[-1]:
            min_dim = min(X.shape[-2:])
            X = center_crop(X, [min_dim, min_dim])
        if oversize_input and X.shape[-1] > img_size \
                or not oversize_input and X.shape[-1] != img_size:
            X = F.interpolate(X, size=img_size, **resize_args)

        # Predict the parameters
        Y_pred = manager.predict(X)
        params = manager.ds.denormalise_rendering_params(Y_pred, idx=0)
        c_params = params['crystal']

        # Calculate the distance errors
        d_pred = np.array(c_params['distances']) * c_params['scale']
        d_target = measurements['distances'][idx_m] * measurements['scale'][idx_m]
        errors_d[model_key][idx_m] = np.abs(d_pred - d_target).mean() * dist_to_mm

        # Calculate the 3D errors
        crystal.build_mesh(
            scale=init_tensor(c_params['scale']),
            distances=init_tensor(c_params['distances']),
            origin=init_tensor(c_params['origin']),
            rotation=init_tensor(c_params['rotation']),
        )
        v_pred = crystal.vertices.clone().detach()
        dists = torch.cdist(v_pred, vertices_targets[idx_m])
        dists_min_ab = dists.amin(dim=0)
        dists_min_ba = dists.amin(dim=1)
        errors_v[model_key][idx_m] = torch.cat([dists_min_ab, dists_min_ba]).mean().item() * dist_to_mm

    def calculate_measurement_errors_for_model(model_key):
        load_manager(model_key)
        for i, X_path in enumerate(image_paths):
            calculate_prediction_errors(model_key, X_path, i)

    for model_key in models.keys():
        logger.info(f'Calculating measurement errors for model: {model_key}')
        calculate_measurement_errors_for_model(model_key)

    # Calculate means and standard deviations
    errors_d_stats = {
        k: {
            'mean': errors_d[k].mean(),
            'std': errors_d[k].std(),
        }
        for k in models.keys()
    }
    errors_v_stats = {
        k: {
            'mean': errors_v[k].mean(),
            'std': errors_v[k].std(),
        }
        for k in models.keys()
    }

    # Save the results as json
    with open(save_dir / f'{START_TIMESTAMP}_measurement_errors_d.json', 'w') as f:
        json.dump(errors_d_stats, f, indent=2)
    with open(save_dir / f'{START_TIMESTAMP}_measurement_errors_v.json', 'w') as f:
        json.dump(errors_v_stats, f, indent=2)

    # Plot the results across the sequence
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for k in models.keys():
        axes[0].errorbar(range(len(image_paths)), errors_d[k], label=k)
        axes[1].errorbar(range(len(image_paths)), errors_v[k], label=k)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_title('Distance errors')
    axes[1].set_title('3D errors')
    axes[0].set_ylabel('Mean squared error (mm)')
    axes[1].set_ylabel('Mean distance error (mm)')
    axes[1].set_xlabel('Image index')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'{START_TIMESTAMP}_measurement_errors.svg', transparent=True)
    plt.show()


if __name__ == '__main__':
    start_time = time.time()

    # plot_model_comparison()
    calculate_measurement_errors()

    # Print how long this took
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f'Finished in {int(minutes):02d}:{int(seconds):02d}.')
