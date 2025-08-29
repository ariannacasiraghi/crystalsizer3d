import pprint
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from trimesh import Trimesh

from crystalsizer3d import LOGS_PATH, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.sequence.utils import get_image_paths, load_manual_measurements
from crystalsizer3d.util.image_scale import get_pixels_to_mm_scale_factor
from crystalsizer3d.util.utils import hash_data, init_tensor, json_to_numpy, to_numpy

import multiprocessing as mp

mp.set_start_method('spawn', force=True)


def get_groups():
    """
    Load experiment groups from yml file.
    """
    with open(LOGS_PATH / 'groups.yml', 'r') as f:
        groups = yaml.safe_load(f)['groups']
    logger.info('Loaded groups from yml file:')
    for group_name, group_paths in groups.items():
        logger.info(f'Group: {group_name}, N={len(group_paths)}')
    return groups


def process_experiment(
        exp_label: str,
        sf_path: Path,
        measurements_path: Path,
        output_dir: Path,
        pixel_to_mm: float,
        cache_only: bool = False,
        n_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Process a single experiment.
    """
    logger.info(f'Processing experiment: {exp_label}')
    output_dir = output_dir / exp_label
    if cache_only and not output_dir.exists():
        raise RuntimeError(f'Cache for experiment {exp_label} does not exist!')

    # Try loading from cache
    if output_dir.exists():
        try:
            areas = np.load(output_dir / 'areas.npz')['areas']
            volumes = np.load(output_dir / 'volumes.npz')['volumes']
            with open(output_dir / 'nrmse.yml', 'r') as f:
                errors = yaml.safe_load(f)
            return areas, volumes, errors
        except Exception as e:
            if cache_only:
                raise e
            logger.warning(f'Failed to load cached data for {exp_label}: {e}')
            logger.info('Regenerating data...')

    output_dir.mkdir(parents=True, exist_ok=True)
    path_substitutions = [
        ('/users/scsti', '/mnt/aire'),
        ('/users/pmcji/projects/crystalsizer3d/data/_HPC',
         '/mnt/aire/projects/crystalsizer3d/data/experiments/sequences'),
        ('/users/pmcji/projects/crystalsizer3d/data/trained_models/1574c2f7c328b7b4039d605a53a05aca_202410101113.json',
         '/mnt/aire/projects/crystalsizer3d/data/trained_models/alpha/predictor/1574c2f7c328b7b4039d605a53a05aca_202410101113.json'),
        ('/users/pmcji/projects/crystalsizer3d/data/trained_models/d18575199f8a1b4ee782ef9353f30da7_202410091756.json',
         '/mnt/aire/projects/crystalsizer3d/data/trained_models/alpha/denoiser/d18575199f8a1b4ee782ef9353f30da7_202410091756.json'),
        ('/users/pmcji/projects/crystalsizer3d/data/trained_models/9b3f5237a45a2492f287af5f0380d1ad_202410101313.json',
         '/mnt/aire/projects/crystalsizer3d/data/trained_models/alpha/keypoint_detector/9b3f5237a45a2492f287af5f0380d1ad_202410101313.json'),
        ('/users/pmcji/projects/crystalsizer3d/data/trained_models/cf0e0ba32eaa66e93b31d3c7528e1c31_202410241630.json',
         '/mnt/aire/projects/crystalsizer3d/data/trained_models/alpha/keypoint_detector/cf0e0ba32eaa66e93b31d3c7528e1c31_202410241630.json'),
        ('/users/pmcji', '/mnt/aire'),
    ]

    def load_args(path: Path) -> Dict[str, str]:
        """
        Fix paths in the args dictionary to use substituted paths.
        """
        with open(path, 'r') as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in args_dict.items():
            if isinstance(v, str):
                for old, new in path_substitutions:
                    if v.startswith(old):
                        args_dict[k] = v.replace(old, new)
                        break
        return args_dict

    # Load the args
    sf_args = SequenceFitterArgs.from_args(load_args(sf_path / 'args_sequence_fitter.yml'))
    ref_args = RefinerArgs.from_args(load_args(sf_path / 'args_refiner.yml'))

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=output_dir,
        path_substitutions=path_substitutions
    )

    # Set up some variables
    logger.info(f'{exp_label}: Setting up variables.')
    image_paths = get_image_paths(sf_args, load_all=True)

    # Get the pixel-to-mm scale factor
    dist_to_mm = get_pixels_to_mm_scale_factor(sf_args.initial_scene, image_paths[0][1], pixel_to_mm)

    # Load the measurements
    logger.info(f'{exp_label}: Loading manual measurements from {measurements_path}')
    measurements = load_manual_measurements(measurements_dir=measurements_path, manager=manager)
    measurement_idxs = measurements['idx']
    distances_m = measurements['distances']
    scales_m = measurements['scale'] * dist_to_mm
    N, M = distances_m.shape

    # Get a crystal object
    ds = manager.ds
    cs = CSDProxy().load(ds.dataset_args.crystal_id)
    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=ds.miller_indices,
        point_group_symbol=cs.point_group_symbol,
        dtype=torch.float64
    )

    # Load the first scene and adapt the crystal to use buffers instead of parameters
    scene = Scene.from_yml(sf_args.initial_scene)
    crystal = scene.crystal
    crystal.to('cpu')
    crystal.dtype = torch.float64
    crystal = crystal.clone()
    # crystal.to(torch.float64)

    # Reorder the measurement distances to line up with the miller indices order
    mi_ref = manager.crystal.all_miller_indices  # measurements are loaded in this order
    mi_crys = crystal.all_miller_indices
    mi_idxs = ((mi_ref[None, ...] == mi_crys[:, None]).all(dim=2)).nonzero(as_tuple=True)[1]
    distances_m = distances_m[:, mi_idxs]

    # Load the parameters
    param_path = sf_path / 'parameters.json'
    logger.info(f'{exp_label}: Loading parameters from {param_path}.')
    with open(param_path, 'r') as f:
        parameters = json_to_numpy(json.load(f))
    distances = parameters['eval']['distances']
    scales = parameters['eval']['scale'] * dist_to_mm
    areas = np.zeros((N, M, 2))
    volumes = np.zeros((N, 2))

    def calculate_face_areas_and_volumes(scale, distances, crystal):
        # Build the fitted crystal
        v, f = crystal.build_mesh(
            scale=init_tensor(scale, dtype=torch.float64),
            distances=init_tensor(distances, dtype=torch.float64),
        )

        # Calculate the face areas
        areas = np.array([crystal.areas[tuple(hkl.tolist())] for hkl in crystal.all_miller_indices])

        # Calculate the volume
        v, f = to_numpy(v), to_numpy(f)
        m = Trimesh(vertices=v, faces=f)

        return areas, m.volume

    def process_measurement(i, m_idx, crystal_base):
        # Clone the shared crystal for thread safety
        crystal = crystal_base.clone()
        areas_f, volume_f = calculate_face_areas_and_volumes(scales[m_idx], distances[m_idx], crystal)
        areas_m, volume_m = calculate_face_areas_and_volumes(scales_m[i], distances_m[i], crystal)
        return i, areas_f, volume_f, areas_m, volume_m

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            lambda args: process_measurement(*args, crystal_base=crystal),
            enumerate(measurement_idxs)
        ))

    # Populate areas and volumes arrays with results
    for i, areas_f, volume_f, areas_m, volume_m in results:
        areas[i, :, 0] = areas_f
        areas[i, :, 1] = areas_m
        volumes[i, 0] = volume_f
        volumes[i, 1] = volume_m

    # Save the areas and volumes
    np.savez(output_dir / 'areas.npz', areas=areas)
    np.savez(output_dir / 'volumes.npz', volumes=volumes)

    def nrmse(y_true, y_pred):
        ptp = y_true.max() - y_true.min()
        if ptp == 0:
            return 0.0
        return ((y_pred - y_true)**2).mean()**0.5 / ptp

    # Calculate the normalised RMSEs
    nrmse_volumes = nrmse(volumes[:, 1], volumes[:, 0])
    tsa = areas.sum(axis=1)
    nrmse_tsa = nrmse(tsa[:, 1], tsa[:, 0])
    nrmse_areas = []
    for i in range(areas.shape[1]):
        nrmse_areas.append(nrmse(areas[:, i, 1], areas[:, i, 0]))
    nrmse_areas = np.mean(nrmse_areas)
    errors = {
        'face_areas': float(nrmse_areas),
        'total_surface_area': float(nrmse_tsa),
        'volume': float(nrmse_volumes),
    }

    with open(output_dir / 'nrmse.yml', 'w') as f:
        yaml.dump(errors, f)

    return areas, volumes, errors


def process_group(
        group: str,
        group_paths: Dict[str, Dict[str, str]],
        output_dir: Path,
        cache_only: bool = False,
        n_workers: int = 4,
) -> Dict[str, Dict[str, float]]:
    logger.info(f'Processing group: {group} (N={len(group_paths)})')
    output_dir = output_dir / group
    if cache_only and not output_dir.exists():
        raise RuntimeError(f'Cache for group {group} does not exist!')

    # Try loading from cache
    if output_dir.exists():
        try:
            with open(output_dir / 'group_errors.yml', 'r') as f:
                group_errors = yaml.safe_load(f)
            return group_errors
        except Exception as e:
            if cache_only:
                raise e
            logger.warning(f'Failed to load cached data for group "{group}": {e}')
            logger.info('Regenerating data...')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process experiments sequentially
    group_errors = {}
    for exp_label, exp_config in group_paths.items():
        _, _, errors = process_experiment(
            exp_label=exp_label,
            sf_path=Path(exp_config['sf_path']),
            measurements_path=Path(exp_config['measurements_path']),
            output_dir=output_dir,
            pixel_to_mm=float(exp_config['pixel_to_mm']),
            cache_only=cache_only,
            n_workers=n_workers,
        )
        group_errors[exp_label] = errors

    # Add means and standard deviations for the group
    group_errors['means'] = {
        'face_areas': float(np.mean([errors['face_areas'] for errors in group_errors.values()])),
        'total_surface_area': float(np.mean([errors['total_surface_area'] for errors in group_errors.values()])),
        'volume': float(np.mean([errors['volume'] for errors in group_errors.values()])),
    }
    group_errors['stds'] = {
        'face_areas': float(np.std([errors['face_areas'] for errors in group_errors.values()])),
        'total_surface_area': float(np.std([errors['total_surface_area'] for errors in group_errors.values()])),
        'volume': float(np.std([errors['volume'] for errors in group_errors.values()])),
    }

    with open(output_dir / 'group_errors.yml', 'w') as f:
        yaml.dump(group_errors, f)

    return group_errors


def calculate_comparisons(
        cache_only: bool = False,
        regenerate: bool = False,
        n_workers: int = 4,
):
    """
    Calculate the measurement errors across different groups of results.
    """
    groups = get_groups()
    output_dir = LOGS_PATH / hash_data(groups)
    if output_dir.exists() and regenerate:
        shutil.rmtree(output_dir)
    elif not output_dir.exists() and cache_only:
        raise RuntimeError(f'Cache directory {output_dir} does not exist!')

    # Try loading from cache
    if output_dir.exists():
        try:
            with open(output_dir / 'errors.yml', 'r') as f:
                errors = yaml.safe_load(f)
            return errors
        except Exception as e:
            if cache_only:
                raise e
            logger.warning(f'Failed to load cached data: {e}')
            logger.info('Regenerating data...')

    output_dir.mkdir(parents=True, exist_ok=True)

    errors = {}
    for group, group_paths in groups.items():
        group_errors = process_group(
            group=group,
            group_paths=group_paths,
            output_dir=output_dir,
            cache_only=cache_only,
            n_workers=n_workers,
        )
        errors[group] = group_errors

    with open(output_dir / 'errors.yml', 'w') as f:
        yaml.dump(errors, f)

    return errors


if __name__ == '__main__':
    start_time = time.time()
    errors = calculate_comparisons(
        cache_only=False,
        regenerate=False,
        n_workers=8,
    )

    pprint.pprint(errors)

    # Print how long this took
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f'Finished in {int(minutes):02d}:{int(seconds):02d}.')
