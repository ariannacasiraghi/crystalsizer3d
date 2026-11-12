# Project log

MSc project: probabilistic generative models for the reconstruction of 3D crystal growth from optical microscopy images

Started: 2 November 2026. Last updated: 12 November 2026.

Conventions: `<user>` stands for my University username.
---

## 1. Aim

Modify the CrystalSizer3D pipeline (Ilett et al., PNAS Nexus 2026, 5(4)) by replacing its deterministic predictor (RegNetZ) with a probabilistic predictor that outputs a distribution of plausible crystal configurations for a given image, instead of one configuration.

- Samples from the distribution would initialise the existing refinement step (denoiser + keypoint detector, both MAGVITv2-based), which is kept unchanged at first. Refinement may turn out not to be needed if the predictor is sufficiently good.
- The mechanism used in the probabilistic predicor comes from Google's FGN paper (Skillful joint probabilistic weather forecasting from marginals, WeatherNext 2). Only the mechanism is used, not the architecture or weights.

### Planned design 

- Keep the ImageNet-pretrained RegNetZ encoder and the paper's head: the classifier is replaced by a fully connected layer producing the vector p, with |p| = |D| + 3 (position) + 3 (rotation, axis-angle) + 2 (material) + 3 (light) = 31, since |D| = 20.
- Sample a noise vector ε ~ N(0, I)^32 once per forward pass. As in the FGN paper, encode it with a single linear layer (32 → 32), then use it to condition every normalisation layer. This follows the θ = θ\* + Δ·ε interpretation. The RegNetZ weights play the role of θ\*.
- Single model for now (aleatoric uncertainty only, no model ensemble to deal with epistemic uncertainty).
- Training loss: fair (unbiased) energy score over several samples per image (needs at least 2, as in the FGN paper). To be compared later with per-parameter CRPS, as in the FGN paper.
- Possible later extension: a ViT in place of RegNetZ.

### Open design questions

- **Conditional normalisation in RegNetZ.** RegNet probably uses BatchNorm, which has no built-in conditioning, so a conditional scale/shift has to be added.
- **Energy-score metric.** The components of p have different units and scales (mm distances, rotation, radiance), so they need normalising. The rotation part may need a geodesic distance.
- **Symmetry ambiguity.** The reconstruction is unique only up to symmetry-equivalent rotations (the crystal paper's refinement resolves these). This is probably where a distribution helps most, but it complicates the choice of rotation metric.
- **Joint distribution structure.** FGN trains on the marginal distribution only and still gets a good joint distribution; the authors argue this is due to the low-dimensional shared noise but are not sure. The energy score targets the joint distribution directly.
- **Scoring samples against one measurement.** How to score N samples drawn from the obtained distribution against a single manual measurement.

---

## 2. Baseline to reproduce

Crystal paper, Table 1 (one-shot prediction performance, RegNetZ, 21.70 M parameters):

- E_d = 0.088 mm (mean centre-to-plane distance error);
- E_v = 0.192 mm (mean 3D vertex position error, nearest-neighbour distances in both directions).

Both are measured against 28 manual measurements on real experimental images, not on the synthetic test data (20%). Training: synthetic dataset, up to 100 epochs or 48 h (whichever first), on a single V100 with AdamW.

Notes:
- The synthetic 80/20 split is separate. It is used for training and for the refinement result in the paper.
- A faster GPU (H200) may reach 100 epochs where the original run might have stopped on the 48 h limit. Keep in mind when comparing numbers.
- Expect to match the paper only approximately (seeds, library versions, epoch limit).
- The manual measurements are in `experiments/manual` (there are more than 28 of them though - still need to understand which ones were used to generate Table 1).

---

## 3. Code and repository

- Forked the authors' public repo to my GitHub account, cloned on my local Windows machine, as well as on Aire and Calder.
- Structure: `crystalsizer3d/` (library classes), `scripts/` (entry points, e.g. `train_synthetic.py`), `app/` (not needed), `aire_jobs/` and `calder_jobs/` (my job scripts).
- Workflow decisions:
  - Work directly on `main` of the fork with small, clear commits. No pull requests planned.
  - Tag the first reproduced baseline (e.g. `baseline-regnetz`) and later milestones.
- `environment_aire.yaml` and `pip_constraints_aire.txt` are my additions.

---

## 4. Environment

Decision: stay close to the original pinned environment, get it running first, update packages only if something forces it.

`environment_aire.yaml` (built for Aire) differs from the original `environment.yml`:

| | Original | Aire environment |
|---|---|---|
| Python | 3.9 | 3.11 |
| PyTorch / torchvision | 2.3 / 0.18 | 2.4 / 0.19 (CUDA 12.1 runtime via conda) |
| timm | 0.9 | 1.0 |
| transformers | 4.37 | 4.44 |
| tensorboard | 2.15 | 2.17 |
| matplotlib | 3.8 | 3.9 |

- Added: scikit-image, einops, omegaconf, orjson, ruamel.yaml, configobj, lightning, geomloss, torchjd, xxhash.
- Not needed: csd-python-api, mayavi, wxpython, and several inverse-rendering and geometry packages.
- Environment name: `crystalsizer3d`, stored in `~/.conda/envs`. The home directory is shared between Aire and Calder, so the same environment is visible on both. If it fails on Calder (still needs to check this), build a separate one (e.g. `crystalsizer3d_calder`).
- Record exact versions (`pip freeze` / `conda env export`) once the baseline runs, and keep the environment fixed through the main experiments.
- Check whether `timm` 1.0 loads the same pretrained RegNetZ weights as 0.9 (pretrained-weight tags changed between versions). Do this before trusting baseline numbers.

---

## 5. Data

Source: Ilett et al. (2026), *Reconstruction of 3D crystal growth from transmission optical microscopy images – Supporting data*, University of Leeds, CC BY 4.0, https://doi.org/10.5518/1684 (https://archive.researchdata.leeds.ac.uk/1537/).

Contents (from the README):
- `images/` (noisy renderings, 100,000 PNGs, ~14 GB): used for training the predictor.
- `images_clean/`, `keypoints_wfv=1.0_kpv=1.0/`, `keypoints_wfv=1.0_kpv=10.0/`, `crystal_bumpmaps/`, `validation/` (403 images, for checking that the dataset can be re-rendered, not a training validation set), `videos/`: not downloaded because not needed for now.
- `experiments/`: original microscope images and manual measurements for the nine sequence experiments in the paper (manual measurements in `experiments/manual`).
- Metadata (from `support_materials.zip`, ~2.8 GB): `rendering_parameters.json` (ground truth; ~1.1 GB), `vertices.json` (~1.0 GB), `parameters.csv`, `train_test_split_0.80.json` (80/20 split, used in all of the paper's experiments), `ds_stats.yml`, `options.yml`, `segmentations.json` (~0.7 GB, not in the README; purpose unknown), `comlog.json`.
- Counts confirmed: 100,000 images.

Locations:
- Aire: `/mnt/scratch/<user>/crystalsizer3d_data` (~18 GB: images 14 GB, support_materials 2.8 GB, experiments 670 MB).
- Calder: `/scratch-calder/users/<user>/crystalsizer3d_data`.

Links in the home directory (home is shared; scratch is not):
- `~/data_aire`, `~/job_logs_aire` -> Aire scratch (works from Aire only).
- `~/data_calder`, `~/job_logs_calder` -> Calder scratch (works from Calder only).

---

## 6. Compute

### Aire
- Leeds HPC cluster; GPU nodes with NVIDIA L40S (3 per node), partition `gpu`, GRES `nvidia_l40`.
- First test job (a matrix multiplication on a GPU node) ran successfully.

### Calder (new, launched October 2026)
- Hardware: 7 GPU nodes with 8 NVIDIA H200 NVL GPUs each (56 total); 96 CPU nodes (~128 cores each).
- Slurm differences from Aire:
  - GPU partition `calder-gpu-hopper`; GPU request `--gres=gpu:nvidia_h200:<n>`.
  - Module line: `module load calder/cpu miniforge3/26.1.1-3` (Miniforge / conda 26.1.1).
  - With GPUs, CPU and memory scale with the number of GPUs (1 GPU = 1/8 of node cores and memory) unless set explicitly.
  - Limit: 16 GPUs per user, at most 2 nodes.
  - `$SCRATCH` and `$TMP_SHARED` are not defined on Calder: use explicit paths. Node-local temporary storage is not documented yet.
- Storage:
  - Home: shared with Aire, backed up.
  - Scratch `/scratch-calder/users/` and flash `/flash-calder/users/`: not shared with Aire, not backed up; quota 1 TB and 1.5 M files per Slurm project.
  - Scratch files are not auto-deleted; remove what is no longer needed.
- CUDA: system module `cuda/13.3.0` exists, only under `calder/hopper`. It should not be needed, because PyTorch from conda brings its own CUDA 12.1 runtime. Works only if the driver supports it. Check with the first GPU job.

### Calder GPU test job
- Script: `calder_jobs/gpu_test.sh` (1 H200, 10 min limit; loads modules, activates `crystalsizer3d`, runs `nvidia-smi` and a one-line PyTorch CUDA check). Output goes to `~/job_logs_calder`.
- Submitted as job 857380; queued with reason `Priority`; Slurm's estimated start was 13 Nov 2026, 20:14. Result pending.
- It will also be the first test of whether the Aire-built environment works on Calder.

---

## 7. Status and next steps

Done:
- Fork and clones; environment built on Aire; data on both systems; Calder access, storage, links and software checked.

Pending / next:
1. Result of Calder GPU test job 857380 (environment and H200 check). 
2. Check the `timm` 1.0 pretrained RegNetZ weights question.
3. Map the repo: data loading, model, loss, training loop, evaluation. Find the config that gave the paper's RegNetZ result.
4. Smoke test: a tiny training run (a few hundred images, 1-2 epochs) on the full chain; time a few hundred steps to estimate full training time. Keep the paper's batch size and settings at first.
5. Full baseline training; evaluation against the manual measurements (E_d, E_v); tag the commit (`baseline-regnetz`).
6. Then design and implement the FGN-style predictor.
