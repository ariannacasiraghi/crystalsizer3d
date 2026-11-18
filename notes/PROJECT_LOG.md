# Project log

MSc project: probabilistic generative models for the reconstruction of 3D crystal growth from optical microscopy images

Started: 2 November 2026. Last updated: 16 November 2026.

Conventions: `<user>` stands for my University username.
---

## 1. Aim

Modify the CrystalSizer3D pipeline (Ilett et al., PNAS Nexus 2026, 5(4)) by replacing its deterministic predictor (RegNetZ) with a probabilistic predictor that outputs a distribution of plausible crystal configurations for a given image, instead of one configuration.

- Samples from the distribution would initialise the existing refinement step (denoiser + keypoint detector, both MAGVITv2-based), which is kept unchanged at first. Refinement may turn out not to be needed if the predictor is sufficiently good.
- The mechanism used in the probabilistic predictor comes from Google's FGN paper (Skillful joint probabilistic weather forecasting from marginals, WeatherNext 2). Only the mechanism is used, not the architecture or weights.

### Planned design 

- Keep the ImageNet-pretrained RegNetZ encoder and the paper's head: the classifier is replaced by a fully connected layer producing the vector p, with |p| = |D| + 3 (position) + 3 (rotation, axis-angle) + 2 (material) + 3 (light) = 31, since |D| = 20.
- Sample a noise vector ε ~ N(0, I)^32 once per forward pass. As in the FGN paper, encode it with a single linear layer (32 → 32), then use it to condition every normalisation layer. This follows the θ = θ\* + Δ·ε interpretation. The RegNetZ weights play the role of θ\*.
- Single model for now (aleatoric uncertainty only, no model ensemble to deal with epistemic uncertainty).
- Training loss: fair (unbiased) energy score over several samples per image (needs at least 2, as in the FGN paper). To be compared later with per-parameter CRPS, as in the FGN paper.
- Possible later extension: a ViT in place of RegNetZ.

### Open design questions

- **Conditional normalisation in RegNetZ.** RegNet probably uses BatchNorm, which has no built-in conditioning, so a conditional scale/shift has to be added.
- **Energy-score metric.** The components of p have different units and scales (mm distances, rotation, radiance), so they need normalising. The rotation part may need a geodesic distance. `ds_stats.yml` (see section 5) already gives a per-parameter mean/variance and may be usable for this.
- **Symmetry ambiguity.** The reconstruction is unique only up to symmetry-equivalent rotations (the crystal paper's refinement resolves these). This is probably where a distribution helps most, but it complicates the choice of rotation metric.
- **Joint distribution structure.** FGN trains on the marginal distribution only and still gets a good joint distribution; the authors argue this is due to the low-dimensional shared noise but are not sure. The energy score targets the joint distribution directly.
- **Scoring samples against one measurement.** How to score N samples drawn from the obtained distribution against a single manual measurement.

---

## 2. Baseline to reproduce

Crystal paper, Table 1 (one-shot prediction performance, RegNetZ, 21.70 M parameters):

- E_d = 0.088 mm (mean centre-to-plane distance error);
- E_v = 0.192 mm (mean 3D vertex position error, nearest-neighbour distances in both directions).

Both are measured against 28 manual measurements on real experimental images, not on the synthetic test data (20%). Training: synthetic dataset, up to 100 epochs or 48 h (whichever first), on a single V100 with AdamW.

Second target, which needs no manual measurements: SI Table S5 gives the RegNetZ test losses on the 20,000 withheld synthetic images: L_d = 0.0035, L_p = 0.0598, L_m = 0.1820, L_l = 0.0394, L_v = 0.0611 (the caption says L_v is not used for training). Use these as the first reproducibility check.

Notes:
- The synthetic 80/20 split is separate. It is used for training and for the refinement result in the paper.
- A faster GPU (H200) may reach 100 epochs where the original run might have stopped on the 48 h limit. Keep in mind when comparing numbers.
- Expect to match the paper only approximately (seeds, library versions, epoch limit).
- The manual measurements are in `experiments/manual` (there are more than 28 of them though - still need to understand which ones were used to generate Table 1, and which script computes E_d and E_v).

---

## 3. Code and repository

- Forked the authors' public repo to my GitHub account, cloned on my local Windows machine, as well as on Aire and Calder.
- Structure: `crystalsizer3d/` (library classes), `scripts/` (entry points, e.g. `train_synthetic.py`), `app/` (not needed), `aire_jobs/` and `calder_jobs/` (my job scripts). The repo has no README and no training command.
- Workflow decisions:
  - Work directly on `master` of the fork with small, clear commits. No pull requests planned.
  - Tag the first reproduced baseline (e.g. `baseline-regnetz`) and later milestones.
  - Edit locally (Windows), commit, push, then `git pull` on the HPC when needed. Do not edit the same file in both places; if a file is edited on the HPC, commit and push it from there before touching it locally.
  - I add explanatory comments to the code as I understand it. They are committed separately from functional changes, so that `git diff` against the authors' code stays readable.
- `environment_hpc.yaml` and `pip_constraints_hpc.txt` are my additions.

### How `scripts/train_synthetic.py` works

- `parse_arguments()` registers the options of eight settings classes (`crystalsizer3d/args/`) on one parser, parses the command line (`parser.parse_args()`), and builds eight settings objects.
- `train()` builds one `Manager(...)` from these and calls `manager.train(n_epochs=runtime_args.n_epochs)`. The `Manager` class (`crystalsizer3d/nn/manager.py`) holds all networks and the training loop. Whether its constructor builds the networks or whether they are built on first use is not yet checked (`Manager.__init__` not yet read).
- Key places in `manager.py`: `init_data_loaders`, `_init_predictor`, `train`. Denoiser and keypoint-detector setup only runs if `train_denoiser` / `train_keypoint_detector` is set (both default to False, confirmed in `add_args`).

### How the arguments work

- Each args class has `add_args(parser)` (registers its command-line options: names, types, defaults) and `from_args(args)` (builds the settings object).
- `BaseArgs.from_args` calls `cls(**vars(args))`: it passes ALL options to the constructor, and each `__init__` keeps its own and absorbs the rest in `**kwargs`.
- Option names use dashes on the command line and underscores in Python (argparse converts).
- Defaults are written twice (in `__init__` and in `add_args`) and sometimes disagree. When running the script, the parser supplies every value, so the **`add_args` default applies**. Known mismatches:
  - `OptimiserArgs`: `w_3d_v_sinkhorn` (1.0 vs 0), `w_kp_l1`, `w_kp_l2`, `w_wf_l1`, `w_wf_l2` (1.0 vs 0).
  - `RuntimeArgs`: `batch_size` (32 vs **8**), `resume` (True vs False), `plot_every_n_batches` (-1 vs 10), `use_gpu` (False vs the `USE_CUDA` environment variable).
  - `DatasetTrainingArgs`: none.
  - Not yet checked: the other four args classes (generator, denoiser, keypoint detector, transcoder).
- Boolean options take a word (`--augment true`, via `str2bool`, which wraps `distutils.util.strtobool`; believed to accept true/false/yes/no/on/off/1/0; not yet tested; `distutils` was removed in Python 3.12, fine on 3.11 but may print a warning). Exception: `--resume` / `--no-resume` take no value.
- `NetworkArgs` is different: the base network is chosen by a required **sub-command** (`fcnet`, `resnet`, `densenet`, `pyramidnet`, `vitnet`, `timm`), written after the general options, e.g. `python scripts/train_synthetic.py <general options> timm --model-name <name> ...`. General options placed after the sub-command would be read by the sub-parser (not tested). `NetworkArgs` has its own `from_args` that collects the sub-command's options into a `hyperparameters` dictionary.
- For the predictor, `timm` loads RegNetZ via `crystalsizer3d/nn/models/timmnet.py` (pretrained=True, num_classes=0). The default `--model-name` is `resnet51q`, not RegNetZ, so it has to be set. `--classifier-hidden-layers` defaults to empty (a single fully connected layer, as in the paper). It exists only for `vitnet` and `timm`, probably because those are pretrained backbones with an attached head (inferred, not yet verified in the code).

### Workaround for the missing `taming` package

`manager.py` imported `taming` at the top (`VQModel`, `VQLPIPSWithDiscriminator`, `get_ckpt_path`), which is only needed for the denoiser / keypoint detector. I wrapped these imports in `try/except ImportError` with empty placeholders (commit "Made taming import optional"). With the defaults, no `taming` code runs for predictor training. Not yet proven by a run. Remove the workaround once the real `taming` is available (still waiting for the previous postdoc). `refiner.py` has the same import but is not used by `train_synthetic.py`.

---

## 4. Environment

Decision: stay close to the original pinned environment, get it running first, update packages only if something forces it.

`environment_hpc.yaml` differs from the original `environment.yml`:

| | Original environment | New environment |
|---|---|---|
| Python | 3.9 | 3.11 |
| PyTorch / torchvision | 2.3 / 0.18 | 2.4 / 0.19 (CUDA 12.1 runtime via conda) |
| timm | 0.9 | 1.0 |
| transformers | 4.37 | 4.44 |
| tensorboard | 2.15 | 2.17 |
| matplotlib | 3.8 | 3.9 |

- Added: scikit-image, einops, omegaconf, orjson, ruamel.yaml, configobj, lightning, geomloss, torchjd, xxhash.
- Not needed: csd-python-api, mayavi, wxpython, and several inverse-rendering and geometry packages.
- Environment name: `crystalsizer3d`, stored in `~/.conda/envs`. The home directory is shared between Aire and Calder, so the same environment is visible on both and works fine on both.
- Record exact versions (`pip freeze` / `conda env export`) once the baseline runs, and keep the environment fixed through the main experiments.
- Check whether `timm` 1.0 loads the same pretrained RegNetZ weights as 0.9 (pretrained-weight tags changed between versions). 

---

## 5. Data

Source: Ilett et al. (2026), *Reconstruction of 3D crystal growth from transmission optical microscopy images – Supporting data*, University of Leeds, CC BY 4.0, https://doi.org/10.5518/1684 (https://archive.researchdata.leeds.ac.uk/1537/).

Contents (from the README):
- `images/` (noisy renderings, 100,000 PNGs, ~14 GB): used for training the predictor.
- `images_clean/`, `keypoints_wfv=1.0_kpv=1.0/`, `keypoints_wfv=1.0_kpv=10.0/`, `crystal_bumpmaps/`, `validation/` (403 images, for checking that the dataset can be re-rendered, not a training validation set), `videos/`: not downloaded because not needed for now.
- `experiments/`: original microscope images and manual measurements for the nine sequence experiments in the paper (manual measurements in `experiments/manual`).
- Metadata (from `support_materials.zip`, ~2.8 GB): `rendering_parameters.json` (ground truth; ~1.1 GB), `vertices.json` (~1.0 GB), `parameters.csv`, `train_test_split_0.80.json` (80/20 split, used in all of the paper's experiments), `ds_stats.yml`, `options.yml`, `segmentations.json` (~0.7 GB, not in the README; purpose unknown), `comlog.json`.
- Counts confirmed: 100,000 images.

What the small metadata files are (from reading them):
- `options.yml`: the settings used to generate the dataset (100,000 samples, 400 x 400 images, 324 samples per pixel, six Miller-index faces, light and bubble ranges, axis-angle rotation). Not training settings.
- `comlog.json`: completion log of the rendering job (`workers`, `completed_idxs`). Nothing about training.
- `ds_stats.yml`: min, max, mean, variance and M2 per parameter over all 100,000 samples (M2/var = 100,000). Names: `d0_001 ... d19_-1-10` (20 face distances), `x, y, z`, `rax, ray, raz`, `ior, r`, `eb, eg, er` (light radiance): 20 + 3 + 3 + 2 + 3 = 31 = |p|. Also `s`, `si`, `il` (not identified). Probably used to standardise targets (to check in the code).

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

### Calder
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
- GPU test job 857380 ran (13 Nov): NVIDIA H200 NVL (~140 GB), PyTorch 2.4.1, cuda.is_available() True.
- Import test (calder_jobs/import_test.sh, job 861906): all main packages import; Mitsuba cuda_ad_rgb variant is available and can be set; crystalsizer3d, ...nn.models.timmnet and ...crystal_renderer import. Plain import mitsuba fails by design unless a variant is set first (the repo sets it). The script ended with Dr.Jit “jit_shutdown” leak messages and a segmentation fault after all checks had printed. Probably shutdown noise from Mitsuba (it also left a core dump file, deleted). The committed script uses `%u` / `$USER` instead of my username (not tested, job not rerun).
- Settings are read from environment variables in crystalsizer3d/__init__.py (DATA_PATH, N_WORKERS (default 8), USE_CUDA, USE_MLAB, LOG_LEVEL, ...), optionally via a .env file (.env.sample sets USE_CUDA=0, USE_MLAB=0, LOG_LEVEL=DEBUG). The repo is in a home folder shared by both clusters, so set DATA_PATH per job (as in import_test.sh) rather than in a shared .env. Use USE_MLAB=0 (no mayavi in the environment); USE_CUDA=1 makes the Mitsuba CUDA variant mandatory.
- `--use-gpu` defaults to the `USE_CUDA` value: if `USE_CUDA` is not 1 in a job, training would run on the CPU. Set `USE_CUDA=1` in the job script (not yet checked what `import_test.sh` sets).

---

## 7. Baseline training settings: paper vs code defaults

Paper: SI Table S4 (predictor column) and the Methods. A defaults check was done for `DatasetTrainingArgs`, `OptimiserArgs`, `NetworkArgs` and `RuntimeArgs`.

| Setting | Paper | Code default | Action |
|---|---|---|---|
| Pretrained start, end-to-end | ImageNet weights, classifier replaced by one FC layer | `freeze_pretrained` False, no hidden layers | ok |
| Optimiser | AdamW | `AdamW` | ok |
| LR schedule | cosine | `cosine` | ok |
| Initial LR | 1e-4 | `--lr-init` 0.1 | set (but see open item 1) |
| Final LR | 1e-5 | `--lr-min` 1e-6 | set |
| Weight decay | 1e-3 | 1e-5 | set |
| Gradient clipping | norm 1 | -1 (off) | set |
| Gradient accumulation | none | 1 | ok |
| Batch size | 32 | **8** (`add_args`) | set |
| Epochs | max 100 (or 48 h) | 300 | set |
| Augmentation | blur (p 0.3, sigma 0.01-5.0), noise (p 0.3, sigma 0.01-0.1) | probabilities and sigmas match, but `--augment` False | set `--augment true` |
| omega_d | 1.0 | `w_distances` 1.0 | ok |
| omega_m | 0.1 | `w_material` 1.0 | set |
| omega_l | 0.033 | `w_light` 1.0 | set |
| omega_v | 1.0 | `w_3d_v_mindists` 1.0 (mapping is a guess) | check |
| Network | RegNetZ (21.70 M parameters) | `timm` sub-command, `--model-name` default `resnet51q` | set; variant to find |

Draft overrides (NOT yet a tested command):
`--dataset-path <path> --augment true --lr-init 1e-4 --lr-min 1e-5 --weight-decay 1e-3 --clip-grad-norm 1 --w-material 0.1 --w-light 0.033 --batch-size 32 --n-epochs 100 --use-gpu true [--seed N] timm --model-name <regnetz variant>`

---

## 8. Open issues

Not resolvable from the code alone (to read in the Manager / `timmnet.py`, or to ask the author):
1. `--lr-init` vs `--lr-pretrained-init`: which one applies to the predictor.
2. `--train-generator` defaults to True. The paper's only generator is the sequence generator for sequence fitting, which seems unrelated. What the flag does in the Manager is not yet known.
3. Canonical rotations (`--use-canonical-rotations`, default False). The paper says targets are canonicalised.
4. `--resize-input` (default True): does the baseline resize the 400 x 400 images to the pretrained input size?
5. Zingg loss: `train_zingg` defaults to True, `w_zingg` to 1.0. Does the baseline use it?
6. Mapping of the paper's omega symbols (omega_P, omega_R, omega_v) to the code's `w_*` options. Table S4 lists omega_v although the Table S5 caption says L_v is not used for training.
7. Exact RegNetZ variant in timm for 21.70 M parameters (candidate: `regnetz_d8`, from memory; check by counting parameters), and which pretrained-weight tag.
8. How to use only about 1,000 images: no option found in `DatasetTrainingArgs`; to look in the data loading (step 3).
9. Where E_d and E_v are computed, and which 28 manual measurements were used.
10. Real `taming` package still pending from the previous postdoc.
11. Ask the author for the exact `train_synthetic.py` command or config for Table 1.

---

## 9. Plan and status

| Step | Status |
|---|---|
| 1. Understand `train_synthetic.py` | done |
| 2. Understand the settings (args classes) and compare with the paper | done (generator, denoiser, keypoint, transcoder args not read) |
| 3. Data path: how the Manager loads images, ground truth and split; 1k subset | next |
| 4. Predictor: `timmnet.py`, output head, loss | to do |
| 5. Training loop (`Manager.train`): what is logged and saved, and where | to do |
| 6. Tiny test (1k images, a few epochs), then the full 100k run on Calder | to do |
| 7. Evaluate against the manual measurements (E_d, E_v); first check with the Table S5 test losses | to do |