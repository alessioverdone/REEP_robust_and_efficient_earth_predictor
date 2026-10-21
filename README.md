# REEP — Robust and Efficient Earth Predictor

Repository principale del modello *"REEP: Robust and Efficient Earth Predictor"*.

Il modello predice l'evoluzione futura di minicubi Sentinel-2 (EarthNet2021):
20 frame predetti a partire da 10 di contesto, con un diffusion model che lavora
nello spazio latente di un autoencoder addestrato separatamente e poi congelato.

---

## Ordine di esecuzione

Quattro passi, in quest'ordine. Ognuno dipende dall'output del precedente.

### 0. Ambiente (una volta sola)

```bash
./setup_env.sh
source venv/bin/activate
export PYTHONPATH=.:src/model:other
```

Installa PyTorch 2.8 (wheel CUDA 12.8) e le dipendenze di `requirements.txt`, poi
verifica che un kernel giri davvero sulla GPU e che la pipeline si importi.

### 1. Dati grezzi — `src/dataset/download.py`

```bash
python src/dataset/download.py
```

Scarica i minicubi EarthNet2021 in `data/raw/` (~455 GB per lo split di train).
Da lanciare una volta sola: **`data/raw/` è di sola lettura**, niente di quello
che produciamo dopo va scritto lì dentro.

### 2. Preprocessing — `src/dataset/preprocessing.py`

```bash
python src/dataset/preprocessing.py                  # the dataset described by the yaml
python src/dataset/preprocessing.py --stats-only     # without the .npy files (enough for phase 2)
```

This is the **only** place where cubes are selected and split. Its parameters
are the `preprocessing` section of the config (no command-line defaults):

| key | meaning |
|---|---|
| `limit` | number of cubes, `0` = all of them; evenly spaced over the sorted file list |
| `limit_after_drop` | `false`: `limit` cubes are sampled, then the bad ones are dropped (the dataset ends up smaller); `true`: `limit` counts the cubes left **after** the drop |
| `val_frac`, `test_frac`, `seed` | split **by location** into train / val / test |
| `times` | temporal versions of the `.npy` files for the autoencoder (must contain `autoencoder.T`) |
| `name` | folder under `data/preprocessed/`, default `n${limit}_s${seed}` |

Reads `data/raw/train/`, writes into `data/preprocessed/<name>/`:

| file | serve a |
|---|---|
| `dataset.json` | **both phases**: explicit list of the cubes of each split, parameters, dataset id. Written last |
| `stats.json` | phase 2: per-band scales, `keep_map`, dropped cubes, diagnostics |
| `cond_stats.json` | phase 2: min/max of mesodynamic and highresstatic on train |
| `{train,val,test}_T10.npy`, `manifest.json` | phase 1: flattened images normalized to `[-1, 1]` |

A cube is dropped when it has an entirely missing image in the frames the
diffusion model reads (`preprocessing.check_frames` = `data.params.T`). A
folder that already holds a dataset is not overwritten unless `--overwrite` is
passed: change `preprocessing.name` to build another one next to it (e.g. a
light version for experiments and the full one).

### 3. Fase 1 — autoencoder: `src/scripts/run_autoencoder_train.py`

```bash
python src/scripts/run_autoencoder_train.py --project_name vae_v1
```

Trains the AutoencoderKL that compresses the images into the latent the
diffusion model works on (128×128 → 32×32 by default). The training resolution
is `data.params.resize`: for faster experiments, `--set data.params.resize=64`
(the `.npy` files stay at 128, batches are downsampled on the GPU).

Train, validation and test are the three `.npy` files of the preprocessed
dataset (`paths.preprocessed_dir`): nothing is split here. At the end of the
training it measures `diffusion.scale_factor` (1 / std of the latents) and
`diffusion.shift_factor` (their mean) with the best weights, plus the per-channel
mean/std, and saves them in `best.pt`, together with the dataset id.

Every run writes only into its own folder, named after the start time:

```
registry/outputs/autoencoder/<YYYY-MM-DD_HH-MM-SS>/
  artifacts/
    config.json          resolved config
    config.yaml          same config, reloadable with --cfgdir
    training_info.json   run, dataset, model, compute cost, train/val/test metrics, checkpoints
    history.csv          one row per epoch
    train.log
  checkpoints/           best.pt and last.pt (none with autoencoder.save_checkpoints=false)
  results/
    curves/              loss, mse, psnr, ssim (train/val, test as a dashed line), lr
    reconstructions/     val_epoch_XXX.png during training, test_best.png at the end
```

The test set is `autoencoder.test_npy` (the test split of the preprocessing),
evaluated with the best weights at the end of the training.
Training stops early after `autoencoder.early_stopping_patience` epochs without
improving `val_<save_best_by>` by more than `early_stopping_min_delta` (0 = off).

By default phase 2 loads `pretrained_path: latest`, the `best.pt` of the most
recent completed run trained on the same dataset at the same resolution. To use
a specific one: `--set diffusion.first_stage_config.pretrained_path=registry/outputs/autoencoder/<run>/checkpoints/best.pt`.
`registry/` is git-ignored except `registry/configurations/`.

### 4. Fase 2 — diffusion: `src/scripts/run_diffusion_train.py`

```bash
accelerate launch src/scripts/run_diffusion_train.py --project_name diffusion_v1
```

Addestra il DiT sui latenti, con l'autoencoder della fase 1 **congelato**.
Prima di caricarne i pesi controlla che l'architettura descritta dal config sia
quella con cui il checkpoint è stato prodotto, e si ferma subito se non combacia.
It also refuses a VAE trained on a different dataset (dataset id) or at a
different resolution. Its train and validation cubes are the train and val
splits of `dataset.json`, the same the autoencoder used.

The DiT works on normalized latents, chosen by `diffusion.latent_normalization`:

| value | encode | decode |
|---|---|---|
| `shift_scale` (default) | `z = (z_vae - shift_factor) * scale_factor` | `z_vae = z / scale_factor + shift_factor` |
| `shift_scale_per_channel` | `z[c] = (z_vae[c] - channel_shift_factors[c]) * channel_scale_factors[c]` | `z_vae[c] = z[c] / channel_scale_factors[c] + channel_shift_factors[c]` |
| `scale` (formula used up to 2026-10-08) | `z = z_vae * scale_factor` | `z_vae = z / scale_factor` |

With the values at `null` (default) the ones measured by phase 1 are read from
the VAE checkpoint (for a checkpoint written before the shift existed,
`shift_factor` comes from `latent_mean` in its run's `training_info.json`); a
value in the config wins. `shift_scale_per_channel` needs the per-channel
statistics, measured by phase 1 from 2026-10-08; for an older VAE run add them
with `python src/scripts/measure_latent_stats.py --run_dir registry/outputs/autoencoder/<run>`
(it keeps the run's global values as they are). At start-up the console
shows the mode, the formula, both values with their source, and the mean/std the
DiT ends up seeing; the same goes into `training_info.json`
(`first_stage.latent_normalization`) and the resolved `config.yaml`.

Like phase 1, every run writes only into its own folder, named after the start
time (the config is saved after the VAE path, `scale_factor`/`shift_factor` and the resumed
checkpoint are resolved, so it records what actually ran):

```
registry/outputs/diffusion/<YYYY-MM-DD_HH-MM-SS>/
  artifacts/
    config.json / config.yaml  resolved config
    training_info.json   run, dataset, first stage (VAE path, its run, latent
                         normalization: mode, formula, scale/shift_factor and where
                         they came from), model, training, metrics, compute,
                         checkpoints
    history.csv          one row per epoch: train loss and latent RMSE, the
                         validation RMSEs on the epochs with a validation
    lead_time.csv        pixel RMSE per predicted frame, model and context-mean
                         baseline, one block per validation
    train.log
  checkpoints/
    checkpoint-<step>/   accelerate states every accelerate.checkpointing_steps,
                         plus one at the end of the training (for resuming)
    best/                lowest val latter_rmse, with best_metric.json
  results/
    curves/              loss, latent_rmse, pixel_rmse, rmse_by_lead_time, lr
    forecasts/           val_epoch_XXX.png at every validation: the same
                         diffusion.n_forecast_examples val cubes, ground truth /
                         prediction / |error| at forecast_lead_columns lead times;
                         final/ = one PNG per val cube from the last validation
```

`loss.png` has only the training curve: the validation does not compute the
diffusion loss, it samples full forecasts and measures their RMSE (latent in
`latent_rmse.png`, pixels in `pixel_rmse.png` and `rmse_by_lead_time.png`).
In the terminal, one progress bar per epoch (its steps, running loss and
latent RMSE) and one per validation (its batches, running `latter_rmse`), each
followed by a summary in the log.

`--project_name` is a label recorded in `training_info.json`. To continue a
training: `--resume_from_checkpoint <run>/checkpoints/checkpoint-<step>`, or
`latest` for the newest checkpoint of the newest run with the same
`--project_name`. The resumed run gets a new folder, inherits the history rows
and the best `latter_rmse` of the earlier one, and continues from that step.

Per una sola validazione da un checkpoint, senza addestrare (it gets its own
folder too, with `mode: validate_only`):

```bash
accelerate launch src/scripts/run_diffusion_train.py --project_name diffusion_v1 \
    --validate_only --resume_from_checkpoint registry/outputs/diffusion/<run>/checkpoints/best
```

---

## Configurazione

Tutto passa da un unico oggetto `cfg`, tipizzato, definito in **`src/config.py`**
e valorizzato dai yaml in `registry/configurations/`.

```python
cfg     = config_from_cli(config_file="configuration_0.yaml", project_name="run_1")
outputs = create_shared_run_outputs(cfg.diffusion.outputs_dir, "forecasts")  # registry/outputs/diffusion/<timestamp>/
rt      = setup_runtime(cfg, outputs)  # accelerator, dtype, seed, logging
```

Le precedenze, dalla più debole alla più forte:

```
schema (src/config.py)  <  yaml (--cfgdir)  <  flag  <  --set chiave=valore
```

**Da terminale**, tutto passa dalle flag:

```bash
python src/scripts/run_diffusion_train.py \
    --project_name diffusion_v1 \
    --cfgdir configuration_0.yaml \
    --set accelerate.num_train_epochs=30 diffusion.scale_factor=0.091269
```

**Dall'IDE** (Run/Debug, che non passa argomenti) valgono i default in testa ai
due script, `DEFAULT_CONFIG_FILE` e `DEFAULT_PROJECT_NAME`. Quando un ripiego
entra in gioco viene scritto a video, così non passa inosservato.

`--cfgdir` accetta un percorso o il solo nome di un file in
`registry/configurations/`. Ogni altro percorso del config è relativo alla root
del progetto: la working directory ci viene spostata all'avvio, quindi si può
lanciare da qualunque cartella.

Tre regole che tengono il config onesto:

- **un parametro, una sorgente.** Quello che si può derivare è un'interpolazione
  (`${...}`) e nel yaml non si scrive. L'architettura del VAE, per esempio, si
  dichiara solo nella sezione `autoencoder`: la fase 2 la legge da lì.
- **le cartelle stanno in `paths`**, e tutto il resto le interpola.
- **one split, one place.** Cubes are selected and split only by the
  preprocessing (`dataset.json`); the two phases only read it.
- **niente valori liberi nel codice.** Ogni numero che influenza un run viene da
  `cfg`; `validate_config()` controlla valori ammessi e coerenze a ogni caricamento
  e riporta tutti gli errori insieme, prima che il run parta.

---

## Struttura del progetto

```
data/
  raw/                     minicubi .npz di EarthNet -- SOLA LETTURA
  preprocessed/<name>/     one folder per preprocessed dataset (dataset.json, stats,
                           cond_stats, .npy, manifest, config.yaml, log)

registry/
  configurations/          i yaml di configurazione (configuration_0.yaml)
  checkpoints/             checkpoints promoted by hand (the ones in regular use)
  outputs/autoencoder/     one folder per phase-1 run (see step 3)
  outputs/diffusion/       one folder per phase-2 run (see step 4)
  outputs/sweeps/<name>/   one folder per hyperparameter search (see below)

src/
  config.py                schema tipizzato + riga di comando. Punto di partenza
                           di tutto: qui si leggono e si cambiano i parametri
  dataset/                 download, preprocessing, dataset del VAE
    dataset_info.py        dataset.json: split lists and dataset id, read by both phases
  model/                   architetture e funzioni delle due fasi
    sgm/                   DiT, sampler, denoiser, loss (da Stable Diffusion)
    diffusion_utils.py     fase 2: modelli, dataloader, training loop
    autoencoder_utils.py   fase 1: costruzione del VAE, checkpoint, metriche
  training/
    runtime.py             Runtime: accelerator, dtype, logging, device, W&B tracker
    state.py               TrainState / TrainMetrics: progresso e misure del run
    run_outputs.py         the run folder under registry/outputs/ and its files
    tracking.py            live report on Weights & Biases (wandb section)
    sweep.py               hyperparameter search with Optuna over the run scripts
    train_autoencoder.py   fase 1: dati, modello, loop
  scripts/
    run_autoencoder_train.py   punto d'ingresso fase 1
    run_diffusion_train.py     punto d'ingresso fase 2
    measure_latent_stats.py    adds the per-channel latent stats to an older VAE run
    run_sweep.py               entry point of a hyperparameter search

other/                     legacy code and references, not maintained (old trainer,
                           MAE first stage, old configs, old requirements)
```

Every run writes only under `registry/outputs/`: there is no `workdir/` any more.

Gli script di lancio hanno tutti la stessa forma — configurazione, dati, modello,
training — e non contengono parametri: solo il cablaggio fra i pezzi.

I tre oggetti che attraversano il codice:

| | cos'è | dove nasce |
|---|---|---|
| `cfg` | cosa si addestra | `src/config.py`, dai yaml |
| `rt` | su cosa gira: accelerator, dtype, device, log | `src/training/runtime.py` |
| `state` | a che punto è: step, epoca, resume, metriche | `src/training/state.py` |

---

## Weights & Biases

`wandb.enabled: true` (or `--set wandb.enabled=true`) reports both phases live
on W&B: the per-epoch metrics of `history.csv` (`epoch/...`), the per-step
training loss of phase 2 (`step/...`), the best/test values in the run summary,
the files of `artifacts/` and `results/` (followed while they change), and the
example PNGs as images. Checkpoints are never uploaded. The W&B run is named
`<phase>/<project_name>/<timestamp>`, grouped by `project_name`, with the
resolved config; its local files go into `<run folder>/wandb/`.

```bash
wandb login                     # once per machine, for wandb.mode=online
python src/scripts/run_autoencoder_train.py --project_name vae_32 --set wandb.enabled=true
# offline (no network): wandb.mode=offline, then: wandb sync <run folder>/wandb/offline-run-*
```

Keys: `project`, `entity`, `mode` (online | offline), `tags`, `upload_files`,
`log_images`.

## Hyperparameter search (Optuna)

```bash
python src/scripts/run_sweep.py --sweep sweep_diffusion_0.yaml
```

A sweep is a yaml in `registry/configurations/` (schema `SweepCfg` in
`src/training/sweep.py`). Each trial is one ordinary run of the phase script,
launched as a subprocess with `--set` overrides, so the training code is not
aware of the sweep. Trials run one at a time.

| file | what it searches |
|---|---|
| `sweep_diffusion_0.yaml` | phase 2, trains every trial: lr, warmup, DiT depth / hidden_size / heads, MinSNR gamma, past_weight |
| `sweep_autoencoder_0.yaml` | phase 1: lr, weight decay, batch size, block_out_channels, layers_per_block |
| `sweep_sampling_0.yaml` | phase 2, `mode: validate_only`: sampler steps and CFG scale on a trained `checkpoint`, no training |

The keys of a sweep:

| key | meaning |
|---|---|
| `name` | the study and its folder `registry/outputs/sweeps/<name>/` |
| `phase`, `mode` | `autoencoder` / `diffusion`; `train`, or `validate_only` (diffusion, with `checkpoint`) |
| `base_config` | the run config(s), as `--cfgdir` |
| `fixed` | dotlist overrides applied to every trial (e.g. fewer epochs) |
| `params` | the search space: `{type: float\|int, low, high, step?, log?}` or `{type: categorical, choices: [...]}`; `choices: {name: value}` for list values such as `block_out_channels` |
| `metric`, `direction` | a `val_` column of `history.csv`, minimize / maximize |
| `n_trials` | total of the study, earlier launches included |
| `sampler` | `tpe` (default, `n_startup_trials` random trials first, `multivariate`) or `random`; `seed` |
| `pruning` | `MedianPruner`: `n_startup_trials`, `n_warmup_steps` (epochs), `interval_steps` |
| `launcher` | null = `python` (phase 1) / `python -m accelerate.commands.launch` (phase 2) |
| `poll_seconds`, `stop_timeout_seconds` | how often `history.csv` is read; how long to wait after SIGINT |

How it works:
- **Metric.** The value of a trial is the best `metric` over the rows of its
  `history.csv` (the epoch of `checkpoints/best`). In `validate_only` it is the
  value of the single validation.
- **Pruning.** While a trial runs, its new `history.csv` rows are reported to
  Optuna, at step = epoch. When the pruner stops a trial, the sweep sends
  SIGINT to the trial: its run folder says `interrupted`, Optuna says `PRUNED`.
  In phase 2 rows exist only at validations, so `validate_epoch_interval` sets
  how often trials are compared.
- **Resume.** Rerunning the same yaml resumes the study (`study.db`) and runs
  only the missing trials. Only `n_trials`, `launcher`, `poll_seconds` and
  `stop_timeout_seconds` may change; any other change is refused, so give a new
  search space a new `name`. Trials stopped by Ctrl-C, or left behind by a
  sweep that was killed, do not count toward `n_trials` and are run again.
- **Failures.** A trial whose config is invalid, or whose run crashes, is
  recorded as `FAIL` with its reason, and the sweep goes on.
- **VAE.** A diffusion sweep resolves `pretrained_path: latest` once, when the
  study is created, and every trial uses that VAE. Trial folders live under
  `outputs/sweeps/`, so `latest` never picks a trial run. To use an
  autoencoder trial in phase 2, pass its `checkpoints/best.pt` as
  `pretrained_path`.
- **Not searchable.** `preprocessing.*` (the dataset must already exist), and
  `autoencoder.*` in a diffusion sweep. The keys the sweep sets itself:
  `run.project_name` (= `name`), the `outputs_dir` of the phase, `wandb.tags`
  (`sweep`, `<name>`, `trial_XXXX`), and, in `validate_only`,
  `run.validate_only` / `run.resume_from_checkpoint`.
- **W&B.** With `wandb.enabled` (in `fixed` or in the base config), all the
  trials of a study are in the same W&B group (`project_name` = `name`).

```
registry/outputs/sweeps/<name>/
    sweep.yaml       the study definition
    study.db         Optuna study (SQLite); optuna-dashboard sqlite:///<path> browses it (pip install optuna-dashboard)
    sweep.log        log of the sweep
    trials.csv       one row per trial: state, value, last epoch, sampled values, run folder, fail reason
    best.json        the best completed trial
    trial_XXXX/
        console.log  stdout + stderr of the trial
        <timestamp>/ the run folder, same layout as registry/outputs/<phase>/
```

Selection uses validation metrics only. Re-train the best configuration
at full scale (more epochs, full dataset) before reporting it. Every
phase-2 trial keeps `checkpoints/best` and one final `checkpoint-<step>`:
delete the trial folders you no longer need.

## Note

- `PYTHONPATH` deve contenere la root del progetto e `src/model` (per `sgm`).
  `other` is only needed for the MAE first stage (`first_stage_config.type: mae`).
- A phase-2 run starts a new training unless `--resume_from_checkpoint` is
  given (a checkpoint folder, or `latest` for the same `--project_name`).
- `other/requirements-legacy.txt` è il vecchio elenco di dipendenze, tenuto solo come
  riferimento: non installarlo (chiede torch 2.1/cu118, che non gira sulle GPU
  Blackwell).
