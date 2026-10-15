import argparse
import os
from argparse import Namespace
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

# Schema tipizzato della configurazione (vedi registry/configurations/configuration_0.yaml).
#
# Convenzioni:
#   - un blocco "target + params" diventa una dataclass con i campi `target` e
#     `params`; `params` contiene ESATTAMENTE gli argomenti del costruttore,
#     perche' instantiate_from_config li passa tutti con ** ;
#   - i blocchi con il solo `target` usano TargetCfg (niente `params`, cosi'
#     instantiate_from_config ricade su dict());
#   - i parametri DERIVATI da altri sono interpolazioni OmegaConf (${...}):
#     hanno una sola sorgente, si aggiornano da soli quando la sorgente cambia
#     e nel yaml non vanno scritti. Se vengono comunque fissati a mano,
#     validate_config() controlla che siano coerenti con la sorgente;
#   - le costanti di codice sono ClassVar: OmegaConf le ignora.
#
# I percorsi relativi (data/, registry/) si intendono SEMPRE
# rispetto a PROJECT_DIR, la root del progetto qui sotto: config_from_cli() ci
# porta la working directory prima di leggere qualunque cosa, cosi' un lancio
# dall'IDE -- che parte dalla cartella dello script o da chissa' dove -- vede
# gli stessi percorsi di un lancio da terminale dalla root.

PROJECT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_DIR = PROJECT_DIR / "registry"
CONFIGS_DIR = REGISTRY_DIR / "configurations"


def project_path(path: Union[str, Path]) -> Path:
    """Percorso assoluto: quelli relativi si agganciano alla root del progetto."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def use_project_dir() -> Path:
    """
    Porta la working directory sulla root del progetto.

    Tutti i percorsi del config sono relativi a li' (data/, registry/), e non li si puo' risolvere uno per uno: buona parte finisce
    dentro oggetti costruiti da instantiate_from_config, che i percorsi li
    riceve come stringhe e li apre per conto suo. Spostare la cwd una volta
    sola all'avvio e' il modo per farli funzionare tutti, ovunque si lanci.
    """
    if Path.cwd() != PROJECT_DIR:
        print(f"[config] working directory -> {PROJECT_DIR}")
        os.chdir(PROJECT_DIR)
    return PROJECT_DIR


# ============================================== RESOLVER ==============================================

def _first_stage_down_ratio(first_stage: DictConfig) -> int:
    """Fattore di downsampling spaziale del first stage (immagine -> latente)."""
    if first_stage.type == "mae":
        return first_stage.params.patch_size
    # VAE: un conv di stride 2 per ogni blocco tranne l'ultimo
    return 2 ** (len(first_stage.params.block_out_channels) - 1)


_RESOLVERS = {
    "tuple": lambda *args: tuple(args),                 # accelerate.optimizer.params.betas
    "floordiv": lambda a, b: a // b,
    "first_stage_down_ratio": _first_stage_down_ratio,
}
for _name, _fn in _RESOLVERS.items():
    if not OmegaConf.has_resolver(_name):
        OmegaConf.register_new_resolver(_name, _fn)


@dataclass
class TargetCfg:
    target: str = ""


# ================================================ RUN ================================================
# Un campo per ogni flag della riga di comando (vedi build_parser()):
# update_from_args() copia qui quelle passate davvero.

@dataclass
class RunCfg:
    project_name: str = ""                              # label of the run, recorded in training_info.json
    cfgdir: list[str] = field(default_factory=list)     # yaml caricati, da sinistra a destra
    seed: int = 42
    # Phase 2. null = new training. A checkpoint-<step> folder of an earlier
    # run (registry/outputs/diffusion/<run>/checkpoints/checkpoint-<step>), or
    # 'latest' = the newest checkpoint-<step> of the newest diffusion run with
    # the same project_name. The resumed run still writes into its own new folder.
    resume_from_checkpoint: Optional[str] = None
    val_num_samples: int = 1                            # campioni per cubo in validazione (K)
    validate_only: bool = False                         # solo una validazione dal checkpoint, niente training
    scale_lr: bool = False                              # vedi effective_learning_rate()
    allow_tf32: bool = True
    save_val_images_max: int = 0                        # cap on the per-cube PNGs of the final validation, 0 = all
    set: list[str] = field(default_factory=list)        # override in dotlist, applicati per ultimi
    show_debug_log: bool = False                        # verbose log


# =============================================== PATHS ===============================================
# Dove stanno le cose. E' l'unico posto in cui si scrivono le cartelle del
# progetto: tutto il resto del config le interpola da qui (percorsi relativi a
# PROJECT_DIR, vedi use_project_dir()).
#
#   data/raw/          minicubi .npz scaricati da EarthNet -- SOLA LETTURA,
#                      qui dentro non si scrive e non si modifica niente
#   data/preprocessed/<preprocessing.name>/
#                      one folder per preprocessed dataset, everything that
#                      src/dataset/preprocessing.py produces: dataset.json (the
#                      split lists), stats.json, cond_stats.json, the .npy files
#                      for the VAE
#   registry/          pesi e configurazioni versionate del progetto
#     checkpoints/     checkpoints promoted by hand (the ones in regular use)
#     outputs/         one timestamped folder per training run, see
#                      src/training/run_outputs.py
#
# Un solo giro di preprocessing serve tutte e due le fasi, che leggono la stessa
# cartella. Which folder is decided by preprocessing.name: switching dataset
# (e.g. a 1000-cube version and the full one) is a single key.

@dataclass
class PathsCfg:
    raw_dir: str = "data/raw/train"
    preprocessed_dir: str = "data/preprocessed/${preprocessing.name}"   # derived
    checkpoints_dir: str = "registry/checkpoints"
    outputs_dir: str = "registry/outputs"


# ============================================ PREPROCESSING ============================================
# Parameters of src/dataset/preprocessing.py. It is the ONLY place where cubes
# are selected (limit) and split by location into train / val / test: the
# result goes into <preprocessed_dir>/dataset.json, and both phases read their
# cubes from there instead of splitting again. See src/dataset/dataset_info.py.

@dataclass
class PreprocessingCfg:
    # Folder under data/preprocessed/. The default encodes limit and seed
    # (n0 = all the cubes); give it a name of your own when changing other
    # parameters. The preprocessing refuses to overwrite a folder that already
    # holds a dataset unless --overwrite is passed.
    name: str = "n${.limit}_s${.seed}"
    raw_dir: str = "${paths.raw_dir}"                   # derived

    # Cube selection. 0 = all the cubes in raw_dir. Otherwise `limit` cubes
    # evenly spaced over the sorted file list (so over all the locations).
    limit: int = 0
    # false: `limit` cubes are sampled and then checked, so the dataset ends up
    #        with fewer cubes (those with empty frames are dropped);
    # true:  `limit` is the number of cubes AFTER the drop: candidates are
    #        checked until `limit` of them survive.
    limit_after_drop: bool = False

    # Split by location (cubes of the same tile overlap spatially): fractions
    # of the locations held out for validation and test.
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0

    # Temporal versions of the .npy files for the autoencoder (first T frames
    # of each cube). Must contain autoencoder.T and data.params.stats_time_key.
    times: list[int] = field(default_factory=lambda: [10])
    # Frames checked for entirely missing images: the whole sequence the
    # diffusion model reads, not only the frames of the .npy files (derived).
    check_frames: int = "${data.params.T}"
    n_bands: int = "${data.params.n_channels}"          # derived
    dtype: str = "float16"                              # dtype of the .npy files

    # Quality filters (see the docstring of preprocessing.py)
    drop_threshold: float = 0.5                         # max fraction of missing values in a kept frame
    keep_empty_frame_cubes: bool = False                # true = drop only the empty frame, not the cube
    drop_static_nan_frac: Optional[float] = None        # drop cubes whose DEM has more missing pixels

    # Execution
    range_sample: int = 256                             # cubes used to estimate the histogram range
    workers: int = 8
    chunk: int = 128                                    # cubes per task


# ============================================= AUTOENCODER =============================================

@dataclass
class AutoencoderCfg:
    # Cartella prodotta da preprocessing.py (derivata: vedi PathsCfg)
    preprocessed_dir: str = "${paths.preprocessed_dir}"

    # Le due versioni temporali. T=10 -> ~213k immagini, T=30 -> ~637k.
    # Conviene partire da T=10 e passare a T=30 solo se si vede overfitting.
    T: int = 10

    # The three splits decided by the preprocessing (dataset.json), one file
    # each: the same cubes and locations the diffusion model uses for its
    # train / val / test. Nothing is split here.
    train_npy: str = "${.preprocessed_dir}/train_T${.T}.npy"
    val_npy: str = "${.preprocessed_dir}/val_T${.T}.npy"
    test_npy: Optional[str] = "${.preprocessed_dir}/test_T${.T}.npy"    # null = no test evaluation
    # cubes and locations of each file, for training_info.json
    manifest: str = "${.preprocessed_dir}/manifest.json"
    dataset_json: str = "${.preprocessed_dir}/dataset.json"

    # Training resolution (derived). The .npy files stay at the native 128x128;
    # batches are downsampled on the GPU to image_size x image_size. Single
    # source: data.params.resize, so phase 1 and phase 2 cannot disagree.
    # For faster experiments: --set data.params.resize=64
    image_size: int = "${data.params.resize}"

    # training: se pesante, ridurre batch_size e/o num_workers
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 100
    lr: float = 1e-3

    # Architettura del VAE. E' l'unica fonte: la fase 2 (diffusion) non la
    # ridichiara, la legge da qui per interpolazione -- vedi VAEParams -- cosi'
    # le due fasi non possono divergere.
    block_out_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    layers_per_block: int = 2
    norm_num_groups: int = 32
    latent_channels: int = 4                # canali del latente su cui lavora il DiT

    # Outputs: every run writes ONLY into <outputs_dir>/<start timestamp>/
    # (artifacts/, checkpoints/, results/), see src/training/run_outputs.py.
    outputs_dir: str = "${paths.outputs_dir}/autoencoder"
    # Name under paths.checkpoints_dir of the VAE that phase 2 loads by default.
    # Phase 1 never writes it: a run's best.pt is copied there by hand once it
    # is the checkpoint in regular use.
    save_path: str = "vae_s2_T${.T}_best.pth"
    # checkpoints/ holds only best.pt and last.pt; false = no checkpoint on
    # disk (quick experiments): the test then uses the best weights kept in memory
    save_checkpoints: bool = True
    resume_from: Optional[str] = None
    # reconstruction examples saved in results/reconstructions/
    n_recon_examples: int = 8
    examples_every: int = 5                 # epochs; 0 = only at the end (test)

    # ottimizzazione
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    # metrica che decide il checkpoint migliore, fra quelle che evaluate_vae()
    # calcola sul validation set. Per psnr e ssim vince il valore piu' alto.
    save_best_by: str = "loss"
    # ReduceLROnPlateau sulla metrica di save_best_by
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    # Early stopping on the same metric as save_best_by: the run stops after
    # early_stopping_patience epochs without an improvement larger than
    # early_stopping_min_delta (absolute, in the metric's units). 0 = disabled.
    # Keep the patience above lr_scheduler_patience, so the lr gets reduced at
    # least once before the run gives up.
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0

    # Escursione dei valori in ingresso, che il preprocessing normalizza in
    # [-1, 1]: serve a PSNR e SSIM.
    data_range: float = 2.0

    # At the end of the training, diffusion.scale_factor (1 / std of the
    # latents) and diffusion.shift_factor (their mean) are measured with the
    # best weights on this many training batches and saved in best.pt, where
    # phase 2 reads them.
    scale_factor_batches: int = 50

    # dataloader
    prefetch_factor: int = 2
    persistent_workers: bool = True


# ============================================== DIFFUSION ==============================================

# --------------------------------------------- network (DiT) ---------------------------------------------

@dataclass
class TemporalModuleCfg:
    # lasciare invariati
    num_attention_heads: int = 4
    attention_block_types: list[str] = field(default_factory=lambda: ["Temporal_Self"])
    temporal_position_encoding: bool = True
    temporal_position_encoding_max_len: int = 40        # deve essere >= image_time_length
    temporal_attention_dim_div: int = 1
    zero_initialize: bool = True
    gradient_checkpoint: bool = True


@dataclass
class DiTParams:
    # Dimensioni DiT-B. depth, hidden_size, patch_size e num_heads vanno
    # modificati solo in caso di grid search.
    depth: int = 12
    hidden_size: int = 512
    patch_size: int = 2
    num_heads: int = 8
    learn_sigma: bool = False
    use_checkpoint: bool = True

    # derivato: canali del latente = latent_channels del first stage
    in_channels: int = "${diffusion.first_stage_config.params.latent_channels}"
    # derivato: rapporto di compressione del first stage. VAE: block_out_channels
    # ha 3 elementi -> 2 downsample -> 4x. MAE: patch_size.
    vae_down_ratio: int = "${first_stage_down_ratio:${diffusion.first_stage_config}}"
    # derivato: grandezza del LATENTE in input, 128 (immagine) / 4 (vae_down_ratio) = 32
    input_size: int = "${floordiv:${data.params.resize},${.vae_down_ratio}}"
    # derivato: quanti step predire
    noise_image_num: int = "${data.params.noise_image_num}"
    # derivato: lunghezza totale della sequenza, 10 di contesto + 20 predetti.
    # climate_var deve avere esattamente questa lunghezza temporale
    # (meso_time_span: all nel dataloader).
    image_time_length: int = "${data.params.T}"
    # derivato: mesodynamic ha 5 variabili
    climate_channel: int = "${data.params.meso_channels}"
    # highresstatic ha 1 banda
    static_channel: int = 1

    # MODIFICA 2: come l'embedding di sigma viene replicato sui 30 istanti.
    # 'repeat' = comportamento storico (t.repeat -> ordine b veloce, NON
    # allineato a x e climate_var che hanno b lento);
    # 'interleave' = t.repeat_interleave, allineato.
    time_embed_repeat: str = "repeat"
    temporal_module_kwargs: TemporalModuleCfg = field(default_factory=TemporalModuleCfg)


@dataclass
class NetworkCfg:
    target: str = "sgm.modules.diffusionmodules.models.DiT.DiT"
    params: DiTParams = field(default_factory=DiTParams)


# --------------------------------------------- conditioner ---------------------------------------------

@dataclass
class IdentityMappingParams:
    param1: bool = True


@dataclass
class EmbedderCfg:
    is_trainable: bool = True
    input_key: str = ""
    ucg_rate: float = 0.1
    target: str = "sgm.modules.encoders.modules.Identify_Mapping"
    params: IdentityMappingParams = field(default_factory=IdentityMappingParams)


@dataclass
class ConditionerParams:
    emb_models: list[EmbedderCfg] = field(default_factory=lambda: [
        # meso condition (mesodynamic) -> chiave 'vector' -> climate_var del DiT.
        # Il dataloader consegna gia' (B, 30, 5): media dei 4 pixel centrali di
        # mesodynamic, aggregata a blocchi di 5 giorni e normalizzata in [-1,1].
        EmbedderCfg(input_key="meso_condition_image"),
        # highres condition (highresstatic) -> chiave 'crossattn' -> static_var del DiT.
        # Dal dataloader arriva (B, 1, 128, 128) normalizzata in [-1,1].
        EmbedderCfg(input_key="highres_condition_image"),
    ])


@dataclass
class ConditionerCfg:
    target: str = "sgm.modules.GeneralConditioner"
    params: ConditionerParams = field(default_factory=ConditionerParams)


# --------------------------------------------- first stage ---------------------------------------------
# Il first stage (congelato) codifica le immagini nel latente su cui lavora il
# DiT. Due varianti, scelte con `type`, con parametri diversi:
#   - 'vae': AutoencoderKL, costruito da build_first_stage_model();
#   - 'mae': Masked Autoencoder, costruito da mae/first_stage.py.
# Lo schema giusto viene scelto da load_config() / update_config() in base a
# `type`, cosi' i params dell'una non vengono validati con lo schema dell'altra.

@dataclass
class FirstStageCfg:
    type: str = "vae"                                   # 'vae' | 'mae'
    pretrained_path: str = ""


@dataclass
class VAEParams:
    # Architettura del VAE congelato: DERIVATA dalla sezione `autoencoder`, che
    # e' quella con cui la fase 1 lo ha addestrato. Non si scrive nel yaml, e
    # se la fase 1 cambia (per esempio block_out_channels) la fase 2 segue da
    # sola. build_first_stage_model() ricontrolla comunque questi valori contro
    # il checkpoint vero, cosi' un checkpoint vecchio non passa inosservato.
    in_channels: int = "${data.params.n_channels}"      # derivato
    out_channels: int = "${.in_channels}"               # derivato
    latent_channels: int = "${autoencoder.latent_channels}"
    block_out_channels: list[int] = "${autoencoder.block_out_channels}"
    layers_per_block: int = "${autoencoder.layers_per_block}"
    norm_num_groups: int = "${autoencoder.norm_num_groups}"
    # resolution the images reach the VAE at: checked against the resolution
    # the checkpoint was trained at (arch.image_size), when it declares one
    image_size: int = "${data.params.resize}"           # derivato


@dataclass
class VAEFirstStageCfg(FirstStageCfg):
    type: str = "vae"
    # The ONLY link between the two phases: the weights produced by phase 1.
    # 'latest' = best.pt of the most recent run under autoencoder.outputs_dir
    # trained on the current dataset (same dataset id) at data.params.resize.
    # Or a path: the promoted ${paths.checkpoints_dir}/${autoencoder.save_path},
    # or a specific run with
    #   --set diffusion.first_stage_config.pretrained_path=registry/outputs/autoencoder/<run>/checkpoints/best.pt
    pretrained_path: str = "latest"
    params: VAEParams = field(default_factory=VAEParams)


@dataclass
class MAEParams:
    # Il checkpoint porta con se' gli iperparametri del MAE: se questi sono
    # diversi, build_mae_first_stage() solleva un errore.
    img_size: int = "${data.params.resize}"             # derivato
    in_channels: int = "${data.params.n_channels}"      # derivato
    patch_size: int = 4
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    latent_channels: int = 4


@dataclass
class MAEFirstStageCfg(FirstStageCfg):
    type: str = "mae"
    pretrained_path: str = "checkpoints_mae_T10/best_finetune.pt"
    params: MAEParams = field(default_factory=MAEParams)


FIRST_STAGE_SCHEMAS: dict[str, type[FirstStageCfg]] = {
    "vae": VAEFirstStageCfg,
    "mae": MAEFirstStageCfg,
}


# ---------------------------------------------- denoiser ----------------------------------------------
# Denoiser, loss e sampler vanno bene cosi'.

@dataclass
class MinSNRWeightParams:
    gamma: float = 5.0
    # MODIFICA 3. 'min' = comportamento storico: clip del MINIMO, il peso
    # resta >= gamma ma sui sigma piccoli cresce fino a ~1e4.
    # 'max' = min-SNR-gamma vero: torch.clamp(sigma**-2, max=gamma).
    mode: str = "min"


@dataclass
class WeightingCfg:
    # alternativa: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
    target: str = "sgm.modules.diffusionmodules.denoiser_weighting.MinSNRWeight"
    params: MinSNRWeightParams = field(default_factory=MinSNRWeightParams)


@dataclass
class DenoiserParams:
    num_idx: int = 1000
    weighting_config: WeightingCfg = field(default_factory=WeightingCfg)
    scaling_config: TargetCfg = field(default_factory=lambda: TargetCfg(
        target="sgm.modules.diffusionmodules.denoiser_scaling.EpsScaling"))
    discretization_config: TargetCfg = field(default_factory=lambda: TargetCfg(
        target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"))


@dataclass
class DenoiserCfg:
    target: str = "sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser"
    params: DenoiserParams = field(default_factory=DenoiserParams)


# ------------------------------------------------ loss ------------------------------------------------

@dataclass
class SigmaSamplerParams:
    num_idx: int = 1000
    discretization_config: TargetCfg = field(default_factory=lambda: TargetCfg(
        target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"))


@dataclass
class SigmaSamplerCfg:
    target: str = "sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling"
    params: SigmaSamplerParams = field(default_factory=SigmaSamplerParams)


@dataclass
class LossParams:
    noise_image_num: int = "${data.params.noise_image_num}"    # derivato
    past_weight: float = 0.1
    sigma_sampler_config: SigmaSamplerCfg = field(default_factory=SigmaSamplerCfg)


@dataclass
class LossCfg:
    # EDM loss
    target: str = "sgm.modules.diffusionmodules.loss.StandardDiffusionLoss"
    params: LossParams = field(default_factory=LossParams)


# ---------------------------------------------- sampler ----------------------------------------------

@dataclass
class GuiderParams:
    scale: float = 2.0


@dataclass
class GuiderCfg:
    # CFG guider
    target: str = "sgm.modules.diffusionmodules.guiders.VanillaCFG"
    params: GuiderParams = field(default_factory=GuiderParams)


@dataclass
class SamplerParams:
    # Valore vecchio 10: con 10 passi di Eulero l'errore di discretizzazione
    # domina (verificato numericamente su un problema giocattolo con denoiser
    # ideale: la varianza dei campioni resta ~12% sotto il target; a 30 passi il
    # deficit scende a ~4%). Costo: la validazione passa da ~20 a ~60 minuti,
    # ogni 10 epoche.
    num_steps: int = 30
    discretization_config: TargetCfg = field(default_factory=lambda: TargetCfg(
        target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"))
    guider_config: GuiderCfg = field(default_factory=GuiderCfg)


@dataclass
class SamplerCfg:
    # EDM sampler
    target: str = "sgm.modules.diffusionmodules.sampling.EulerEDMSampler"
    params: SamplerParams = field(default_factory=SamplerParams)


# ------------------------------------------------------------------------------------------------------

@dataclass
class DiffusionCfg:
    # MODIFICA 4. Il DiT lavora su z * scale_factor e assume che quei valori
    # abbiano varianza ~1: e' l'ipotesi con cui sono tarati i sigma e la loss.
    # Il valore giusto e' 1/std(z) sui latenti del VAE congelato.
    # null (default) = read from the VAE checkpoint: phase 1 measures it at the
    # end of its training and saves it in best.pt, so it cannot go stale when
    # the VAE changes. A number overrides it: --set diffusion.scale_factor=<value>
    # (0.13025 is the Stable Diffusion value, tuned on THEIR VAE).
    scale_factor: Optional[float] = None
    # How the VAE latents are normalized before the DiT sees them:
    #   "shift_scale" (default): z = (z_vae - shift_factor) * scale_factor,
    #       decode with z_vae = z / scale_factor + shift_factor (as SDXL/Flux).
    #       The latents come out with mean ~0 and std ~1.
    #   "shift_scale_per_channel": the same, with one shift and one scale per
    #       latent channel: z[c] = (z_vae[c] - channel_shift_factors[c]) * channel_scale_factors[c]
    #       (every channel gets mean ~0 and std ~1, not only their union).
    #   "scale": z = z_vae * scale_factor, the formula used up to 2026-10-08.
    #       Only the std is fixed: the mean of the latents stays as it is.
    latent_normalization: str = "shift_scale"
    # Mean of the VAE latents, subtracted with "shift_scale". null (default) =
    # read from the VAE checkpoint, measured by phase 1 together with
    # scale_factor. Must stay null with "scale", which has no shift.
    shift_factor: Optional[float] = None
    # "shift_scale_per_channel" only: per-channel mean and 1/std of the
    # latents, one value per latent channel. null (default) = read from the
    # VAE checkpoint (phase 1 measures them from 2026-10-08; for an older VAE
    # run: python src/scripts/measure_latent_stats.py --run_dir <run folder>). With this
    # mode scale_factor and shift_factor stay null, and vice versa.
    channel_shift_factors: Optional[list[float]] = None
    channel_scale_factors: Optional[list[float]] = None

    disable_first_stage_autocast: bool = True
    # pretrained_image_ckpt (non usato): checkpoints/sd_xl_base_1.0.safetensors
    # oppure checkpoints/768-v-ema.safetensors
    input_key: str = "imgs"
    img_log_step_inteval: int = 1000000
    log_keys: list[str] = field(default_factory=lambda: ["txt"])   # TODO

    network_config: NetworkCfg = field(default_factory=NetworkCfg)
    network_wrapper: str = "sgm.modules.diffusionmodules.wrappers.DiTWrapper"
    compile_model: bool = False

    conditioner_config: ConditionerCfg = field(default_factory=ConditionerCfg)
    first_stage_config: FirstStageCfg = field(default_factory=VAEFirstStageCfg)
    denoiser_config: DenoiserCfg = field(default_factory=DenoiserCfg)
    loss_fn_config: LossCfg = field(default_factory=LossCfg)
    sampler_config: SamplerCfg = field(default_factory=SamplerCfg)

    # Outputs: every run writes ONLY into <outputs_dir>/<start timestamp>/
    # (artifacts/, checkpoints/, results/), see src/training/run_outputs.py.
    outputs_dir: str = "${paths.outputs_dir}/diffusion"
    # results/forecasts/val_epoch_XXX.png at every validation: this many
    # validation cubes (always the same ones), each shown at
    # forecast_lead_columns lead times evenly spread over the predicted frames.
    # 0 = no figure. The final validation also writes one PNG per cube
    # (run.save_val_images_max) in results/forecasts/final/.
    n_forecast_examples: int = 4
    forecast_lead_columns: int = 5

    # Accumulatori della validazione.
    # mean_k_se / best_k_se: valutazione a K campioni (--val_num_samples).
    # mean_k = SE della MEDIA dei K campioni (stima della media condizionata del
    # modello: e' il confronto onesto con la baseline past_mean, che e' anch'essa
    # una predizione deterministica); best_k = SE del campione migliore per cubo
    # (best-of-K, stile EarthNet). Con K=1 entrambe coincidono con predict_se.
    # Le chiavi nuove stanno in fondo: le prime 8 posizioni conservano l'ordine
    # storico degli accumulatori.
    VAL_SUM_KEYS: ClassVar[tuple[str, ...]] = (
        "past_mean_se", "predict_se", "n_latent",
        "fromer_se", "n_fromer", "latter_se", "n_latter",
        "n_cubes",
        "mean_k_se", "best_k_se",
    )
    # allowed values of latent_normalization (see the field above)
    LATENT_NORMALIZATIONS: ClassVar[tuple[str, ...]] = ("shift_scale", "shift_scale_per_channel", "scale")


# ================================================ DATA ================================================

@dataclass
class DataParams:
    # minicubi .npz grezzi: servono tutti e 30 gli istanti, mentre il .npy del
    # preprocessing ne contiene solo i primi 10 (quelli usati per il VAE)
    data_root: str = "${paths.raw_dir}"
    # The cubes of each split, decided by the preprocessing: the same the
    # autoencoder was trained, validated and tested on. Nothing is split here.
    dataset_json: str = "${paths.preprocessed_dir}/dataset.json"
    # lo/hi per banda di highresdynamic, gli stessi usati per allenare il VAE.
    stats_json: str = "${paths.preprocessed_dir}/stats.json"
    stats_time_key: str = "10"
    # min/max of mesodynamic and highresstatic on the train split, written by
    # the preprocessing (recomputed by the dataloader only if the parameters
    # below no longer match)
    cond_stats_json: str = "${paths.preprocessed_dir}/cond_stats.json"

    # sequenza: 10 frame di contesto + 20 da predire
    T: int = 30
    noise_image_num: int = 20
    n_channels: int = 4
    resize: int = 128

    # condizionamenti
    meso_channels: int = 5
    meso_steps_per_frame: int = 5           # 150 step giornalieri / 30 frame = 5
    meso_center_crop: int = 2               # solo i 4 pixel centrali, poi mediati
    # Sottocampionamento, non media: si prende il giorno dell'acquisizione
    # Sentinel-2, che e' l'ULTIMO della finestra (frame i -> giorno 5i+4).
    # 'mean'/'sum' aggregano i 5 giorni, 'first' prende il giorno 5i.
    meso_agg: str = "last"
    # 'all' = 30 step (serve al DiT); 'future' = solo i 20 target
    meso_time_span: str = "all"

    # dataloader
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    do_hflip: bool = True
    stats_workers: int = 0                  # processi usati solo per il calcolo iniziale di cond_stats.json
    dtype: str = "float16"


@dataclass
class DataCfg:
    target: str = "src.dataset.dataloader_s2_dif.S2TemporalDataModule"
    params: DataParams = field(default_factory=DataParams)


# ============================================== ACCELERATE ==============================================

@dataclass
class OptimizerParams:
    betas: Any = "${tuple:0.9, 0.999}"
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass
class OptimizerCfg:
    target: str = "torch.optim.AdamW"
    params: OptimizerParams = field(default_factory=OptimizerParams)


@dataclass
class AccelerateCfg:
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"           # 'no' | 'fp16' | 'bf16'
    # MODIFICA 1: formato in cui sono TENUTI i pesi dei moduli ALLENATI
    # (DiT + conditioner). Il VAE e' congelato e resta sempre al dtype di
    # mixed_precision: nessun optimizer lo aggiorna, e tenerlo in fp32
    # rallenterebbe l'encode dei 240 frame per step senza alcun beneficio.
    # 'auto' = come mixed_precision, quindi bf16: AdamW aggiorna parametri con 8
    # bit di mantissa e gli update piccoli si perdono.
    # 'float32' = pesi in fp32 con autocast bf16 sopra (mixed precision standard).
    # Costa ~2x memoria e checkpoint ~2x piu' grandi.
    weight_dtype: str = "auto"

    # training
    num_train_epochs: int = 100
    # None = ricavato dal trainer come num_train_epochs * step per epoca
    # (TODO: da controllare se viene davvero usato)
    max_train_steps: Optional[int] = None

    # optimizer
    learning_rate: float = 2e-4             # tarato su 8 GPU x 16 batch size
    learning_rate_base_batch_size: int = 16
    max_grad_norm: float = 1.0
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500

    # checkpoint
    checkpointing_epochs: bool = False
    checkpointing_steps: int = 1000
    checkpointing_steps_list: list[int] = field(default_factory=lambda: [
        2500, 5000, 10000, 20000, 40000, 80000, 160000, 320000, 640000])
    checkpoints_total_limit: int = 50
    logging_steps: int = 500

    # Validazione ogni 10 epoche, sull'INTERO validation set (183 batch da 8 cubi):
    # le quattro RMSE aggregate finiscono in logs/validation_metrics.log e il
    # modello con il latter_rmse piu' basso in checkpoints/best-latter_rmse.
    # L'ultima epoca viene sempre validata a prescindere dall'intervallo, cosi'
    # lo stato finale ha una misura e puo' aggiornare il best.
    # (portato a 5 per la coda di diffusion_v1, poi rimesso a 10)
    validate_epoch_interval: int = 10
    validate_batch_num: int = 10000


# ================================================= ENS =================================================
# Metriche EarthNet (ENS con MAD, OLS, EMD, SSIM, piu' SSIM_RAW, P-MAE e N-MAE) calcolate
# durante il training di fase 2, le stesse di src/scripts/evaluate_ens.py (il codice sta
# in src/utils/ens_metrics.py). Servono predizioni vere, cioe' il sampling completo:
# per questo si fanno solo dove si fa gia' inferenza, non sui batch di training.
# Ogni cubo e' predetto run.val_num_samples (K) volte e di ognuno si tiene la predizione
# con l'ENS piu' alto (best-of-K). Le colonne finiscono in history.csv (val_ENS, ...,
# train_ENS, ...) e nei grafici results/curves/ens*.png.

@dataclass
class EnsCfg:
    enabled: bool = False                   # false = niente ENS (nessun costo)
    validation: bool = True                 # ENS sul validation set, nelle epoche di validazione
    train: bool = False                     # ENS anche su un campione di cubi di train
    train_cubes: int = 100                  # quanti cubi di train per valutazione
    train_every_epochs: int = 1             # ogni quante epoche (1 = a fine di ogni epoca)
    # true = sempre gli stessi cubi (curve confrontabili tra epoche); false = un campione
    # casuale nuovo a ogni valutazione
    train_fixed_subset: bool = True
    # 'fast' = MAD ed EMD vettorizzati di evaluate_ens.py (stesso numero), 'toolkit' = originali
    scoring: str = "fast"
    # data_range della SSIM: 2.0 e' quello con cui sono calibrati i punteggi pubblicati
    ssim_data_range: float = 2.0
    # true = i pixel nuvolosi sono esclusi dal confronto, come nello scoring ufficiale
    # (la maschera si legge dai .npz originali, il modello non la vede); false = tutti i pixel
    cloud_mask: bool = True
    cloud_prob_threshold: float = 0.20      # sotto questa probabilita' di nuvola un pixel e' valido
    cloud_prob_scale: float = 100.0         # fondo scala del canale (EarthNet la da' in %)

    SCORING_MODES: ClassVar[tuple[str, ...]] = ("fast", "toolkit")


# ================================================ WANDB ================================================
# Live report of the training on Weights & Biases (src/training/tracking.py),
# for both phases. The run folder under registry/outputs/ stays the reference:
# W&B gets the metrics of history.csv as they are written, the files of
# artifacts/ and results/, and the example PNGs as images. Checkpoints are
# never uploaded. Online mode needs `wandb login` once on the machine.

@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "REEP"
    entity: Optional[str] = None                        # null = the default entity of the login
    mode: str = "online"                                # online | offline (sync later with `wandb sync`)
    tags: list[str] = field(default_factory=list)       # phase and project_name are added anyway
    upload_files: bool = True                           # artifacts/ and results/ (no checkpoints)
    log_images: bool = True                             # example PNGs as W&B images

    MODES: ClassVar[tuple[str, ...]] = ("online", "offline")


# ================================================ ROOT ================================================
# Da fare: grid search su dimensioni del DiT e learning rate; capire i
# parametri di training (worker e batch size); sistemare i checkpoint.

@dataclass
class TrainCfg:
    run: RunCfg = field(default_factory=RunCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    preprocessing: PreprocessingCfg = field(default_factory=PreprocessingCfg)
    autoencoder: AutoencoderCfg = field(default_factory=AutoencoderCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)
    data: DataCfg = field(default_factory=DataCfg)
    accelerate: AccelerateCfg = field(default_factory=AccelerateCfg)
    ens: EnsCfg = field(default_factory=EnsCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)


# ======================================== FUNZIONI AUSILIARIE ========================================
# Tutte lavorano su una DictConfig tipizzata (OmegaConf.structured(TrainCfg)),
# non sull'oggetto dataclass: resta compatibile con instantiate_from_config
# (`"target" in cfg`, `cfg.get(...)`, `**params`) e controlla tipi e nomi delle
# chiavi a ogni assegnazione, quindi un refuso fa fallire subito il lancio.

_FIRST_STAGE_TYPE_KEY = "diffusion.first_stage_config.type"


def _set_first_stage_schema(cfg: DictConfig, fs_type: Optional[str]) -> None:
    """Sostituisce il nodo first_stage_config con lo schema della variante `fs_type`."""
    if fs_type is None or fs_type == cfg.diffusion.first_stage_config.type:
        return
    if fs_type not in FIRST_STAGE_SCHEMAS:
        raise ValueError(f"{_FIRST_STAGE_TYPE_KEY} deve essere uno di "
                         f"{sorted(FIRST_STAGE_SCHEMAS)}, non {fs_type!r}")
    cfg.diffusion.first_stage_config = FIRST_STAGE_SCHEMAS[fs_type]()


def load_config(*paths: Union[str, Path], overrides: Optional[list[str]] = None) -> TrainCfg:
    """
    Schema TrainCfg <- yaml in `paths` (da sinistra a destra) <- `overrides` in
    dotlist (es. ["diffusion.scale_factor=0.2"]), poi validate_config().
    """
    sources = [OmegaConf.load(p) for p in paths]
    if overrides:
        sources.append(OmegaConf.from_dotlist(list(overrides)))

    cfg = OmegaConf.structured(TrainCfg)
    # la variante del first stage va scelta PRIMA del merge (vince l'ultima sorgente)
    for src in sources:
        _set_first_stage_schema(cfg, OmegaConf.select(src, _FIRST_STAGE_TYPE_KEY))
    cfg = OmegaConf.merge(cfg, *sources)

    validate_config(cfg)
    return cfg  # type: ignore[return-value]


def _merge_into(cfg: TrainCfg, source: DictConfig) -> TrainCfg:
    """Merge sul posto di `source` in `cfg` (chiavi e tipi controllati), poi validate_config()."""
    _set_first_stage_schema(cfg, OmegaConf.select(source, _FIRST_STAGE_TYPE_KEY))
    cfg.merge_with(source)
    validate_config(cfg)
    return cfg


def update_config(cfg: TrainCfg, combo: dict[str, Any]) -> TrainCfg:
    """
    Aggiorna `cfg` sul posto con un dict {chiave in dotlist: valore}, per
    esempio una combinazione della grid search:
        {"accelerate.learning_rate": 1e-4, "diffusion.network_config.params.depth": 8}
    I parametri derivati seguono da soli. Restituisce `cfg` per concatenare.
    """
    source = OmegaConf.create()
    for key, value in combo.items():
        OmegaConf.update(source, key, value)
    return _merge_into(cfg, source)


def update_from_args(cfg: TrainCfg, args: Namespace) -> TrainCfg:
    """
    Copia in cfg.run le flag della riga di comando che hanno un campo omonimo
    in RunCfg (quelle a None -- cioe' non passate -- vengono saltate), poi
    applica gli override di --set
    ("chiave=valore", valori interpretati come nel yaml). La riga di comando
    vince sul yaml.
    """
    for f in fields(RunCfg):
        value = getattr(args, f.name, None)
        if value is not None:
            cfg.run[f.name] = value
    # Lo schema e' tipizzato, quindi una chiave inesistente o un valore del tipo
    # sbagliato fanno fallire il merge: senza, "--set diffusion.ablaton.x=1"
    # passerebbe in silenzio e il run girerebbe identico al baseline. L'errore di
    # OmegaConf da solo dice poco, qui si aggiunge cosa si stava applicando.
    try:
        return _merge_into(cfg, OmegaConf.from_dotlist(list(cfg.run.set)))
    except OmegaConfBaseException as err:
        raise SystemExit(f"--set: override non valido ({err}).\n"
                         f"Override richiesti: {list(cfg.run.set)}\n"
                         "Le chiavi devono esistere nello schema di src/config.py "
                         "e il valore deve essere del tipo giusto.") from err


def resolve_config_path(config_file: Union[str, Path]) -> Path:
    """Percorso di un yaml: un path esplicito oppure il nome di un file in
    registry/configurations/."""
    path = Path(config_file)
    if path.is_file():
        return path
    candidate = CONFIGS_DIR / config_file
    if candidate.is_file():
        return candidate
    raise SystemExit(f"configurazione non trovata: {config_file} "
                     f"(cercata anche in {CONFIGS_DIR})")


def config_from_args(args: Namespace) -> TrainCfg:
    """Config completo di un run: yaml di --cfgdir + flag della riga di comando + --set."""
    if not args.cfgdir:
        raise SystemExit("--cfgdir: nessun file di configurazione indicato.\n\n" + CLI_EPILOG)
    paths = [resolve_config_path(p) for p in args.cfgdir]
    return update_from_args(load_config(*paths), args)


def initialize_configuration(config_file: Optional[str] = None,
                             overrides: Optional[list[str]] = None) -> TrainCfg:
    """
    Carica `config_file` (nome in registry/configurations/ o percorso) oppure,
    se None, i default dello schema. Stampa la configurazione risultante.
    """
    paths = [resolve_config_path(config_file)] if config_file is not None else []
    cfg = load_config(*paths, overrides=overrides)
    print(f"Loaded configuration: {paths[0] if paths else 'default'}")
    print(config_to_str(cfg))
    return cfg


# ============================================ RIGA DI COMANDO ============================================

CLI_EPILOG = """\
Esempio, dalla root del progetto (quella con data/ e registry/):

  accelerate launch src/scripts/run_diffusion_train.py \\
      --project_name diffusion_v1 \\
      --cfgdir configuration_0.yaml \\
      --set accelerate.num_train_epochs=30 diffusion.scale_factor=0.091269

--cfgdir accetta sia un percorso sia il nome di un file in
registry/configurations/. Gli altri percorsi del config (data/, registry/) sono relativi alla root del progetto, e la working directory ci viene
spostata all'avvio: si puo' lanciare da qualunque cartella, IDE compreso. Le flag NON passate non toccano il config: i loro
valori restano quelli del yaml e, per quello che il yaml non dice, quelli di
RunCfg in src/config.py.
"""


def build_parser() -> argparse.ArgumentParser:
    """
    Una flag per ogni campo di RunCfg, con lo stesso nome: e' cosi' che
    update_from_args() le ritrova.

    Tutti i default sono None -- i valori veri stanno in RunCfg e nel yaml --
    altrimenti una flag mai passata sovrascriverebbe il yaml con il default di
    argparse, e configurare un run dal file diventerebbe impossibile. Per lo
    stesso motivo i booleani usano BooleanOptionalAction (--allow_tf32 /
    --no-allow_tf32) invece di store_true, che non saprebbe distinguere "non
    passato" da "passato a False".
    """
    parser = argparse.ArgumentParser(
        description="Training del modello di diffusione video.",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project_name", type=str, default=None,
        help="label of the run, recorded in training_info.json; every run "
             "writes into its own timestamped folder under registry/outputs/")
    parser.add_argument(
        "--cfgdir", nargs="*", default=None, metavar="YAML",
        help="config da caricare, da sinistra a destra (l'ultimo vince); nome "
             "di un file in registry/configurations/ oppure percorso")
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help=f"seed di set_seed (default: {RunCfg.seed})")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="phase 2: a checkpoint-<step> folder of an earlier run, or 'latest' "
             "for the newest one of the newest run with the same --project_name. "
             "The resumed run writes into its own new folder "
             f"(default: {RunCfg.resume_from_checkpoint!r} = new training)")
    parser.add_argument(
        "--val_num_samples", type=int, default=None,
        help="numero di campioni generati per cubo in validazione. Con K>1 si "
             "calcolano anche mean_of_k_latent_rmse (RMSE della media dei K "
             "campioni, il confronto onesto con la baseline past_mean) e "
             "best_of_k_latent_rmse (best-of-K). Il tempo di validazione scala "
             "linearmente con K.")
    parser.add_argument(
        "--validate_only", action=argparse.BooleanOptionalAction, default=None,
        help="carica solo i pesi del checkpoint di --resume_from_checkpoint "
             "(model.safetensors, senza optimizer ne' stato dei dataloader), "
             "esegue una validazione completa e termina. The run gets its own "
             "folder (mode validate_only in training_info.json); the checkpoint "
             "is only read.")
    parser.add_argument(
        "--scale_lr", action=argparse.BooleanOptionalAction, default=None,
        help="scala il learning rate con batch size, gradient accumulation e "
             "numero di GPU (vedi effective_learning_rate)")
    parser.add_argument(
        "--allow_tf32", action=argparse.BooleanOptionalAction, default=None,
        help="TF32 sulle GPU Ampere: piu' veloce, meno preciso "
             f"(default: {RunCfg.allow_tf32})")
    parser.add_argument(
        "--save_val_images_max", type=int, default=None,
        help="cap on the per-cube PNGs saved in results/forecasts/final/ by "
             "the last validation (0 = every validation cube)")
    parser.add_argument(
        "--set", nargs="*", default=None, metavar="CHIAVE=VALORE",
        help="override del config in notazione dotlist, applicati DOPO i yaml "
             "di --cfgdir. Esempio: --set accelerate.weight_dtype=float32 "
             "diffusion.scale_factor=0.65 . Una chiave inesistente o di tipo "
             "sbagliato fa fallire il lancio, cosi' un refuso non produce un "
             "run identico al baseline.")
    # Non e' configurazione: lo passa il launcher (torchrun/accelerate) e vive
    # nel Runtime, non in cfg. Sta qui solo perche' argparse non lo rifiuti.
    parser.add_argument("--local_rank", type=int, default=-1,
                        help=argparse.SUPPRESS)
    return parser


def check_run_args(cfg: TrainCfg) -> None:
    """
    The minimum to start a run: project_name labels it in training_info.json
    and is what resume_from_checkpoint='latest' looks for.
    """
    if not cfg.run.project_name:
        raise SystemExit(
            "--project_name NAME not set: it labels the run in training_info.json "
            "(the run folder itself is named after the start time).\n\n" + CLI_EPILOG)


def config_from_cli(argv: Optional[list[str]] = None,
                    *,
                    config_file: Optional[Union[str, Path, list]] = None,
                    project_name: Optional[str] = None,
                    overrides: Optional[list[str]] = None) -> TrainCfg:
    """
    Config di un run: flag -> yaml di --cfgdir -> --set -> validate_config().
    Sostituisce parse_args() + check_sanity().

    I tre argomenti a parola chiave sono i RIPIEGHI per quando non arrivano
    dalla riga di comando, cioe' il lancio dall'IDE (Run/Debug, che non passa
    argomenti):

        cfg = config_from_cli(config_file="configuration_0.yaml",
                              project_name="prova")

    Lo stesso identico codice, lanciato da terminale con --cfgdir e
    --project_name, usa quelli: la riga di comando vince sempre, il ripiego
    entra in gioco solo se la flag manca. Quando succede lo dice, cosi' un
    --cfgdir dimenticato sul cluster non passa inosservato.

    `config_file` accetta un nome, un percorso o una lista (piu' yaml, mergiati
    da sinistra a destra); `overrides` e' l'equivalente di --set.
    """
    args = build_parser().parse_args(argv)

    if not args.cfgdir and config_file is not None:
        args.cfgdir = [config_file] if isinstance(config_file, (str, Path)) else list(config_file)
        print(f"[config] nessun --cfgdir: uso il default del codice -> {args.cfgdir}")
    if not args.project_name and project_name is not None:
        args.project_name = project_name
        print(f"[config] nessun --project_name: uso il default del codice -> {project_name}")
    if not args.set and overrides:
        args.set = list(overrides)
        print(f"[config] nessun --set: uso gli override del codice -> {args.set}")

    # I yaml si risolvono rispetto alla cartella da cui si e' lanciato (un
    # --cfgdir relativo deve funzionare come se lo si fosse scritto li'), tutto
    # il resto rispetto alla root del progetto: quindi prima si assolutizzano i
    # percorsi dei config, poi ci si sposta.
    if args.cfgdir:
        args.cfgdir = [str(resolve_config_path(path).resolve()) for path in args.cfgdir]
    use_project_dir()

    cfg = config_from_args(args)
    check_run_args(cfg)
    return cfg


def validate_config(cfg: TrainCfg) -> None:
    """
    Controlla valori ammessi e coerenza fra parametri, raccogliendo TUTTI gli
    errori prima di sollevare. Chiamata da load_config() e update_config().
    """
    errors = []
    data = cfg.data.params
    dit = cfg.diffusion.network_config.params
    first_stage = cfg.diffusion.first_stage_config

    choices = {
        "accelerate.mixed_precision": ("no", "fp16", "bf16"),
        "accelerate.weight_dtype": ("auto", "float32"),
        "diffusion.network_config.params.time_embed_repeat": ("repeat", "interleave"),
        "diffusion.denoiser_config.params.weighting_config.params.mode": ("min", "max"),
        _FIRST_STAGE_TYPE_KEY: tuple(FIRST_STAGE_SCHEMAS),
        "autoencoder.save_best_by": ("loss", "mse", "psnr", "ssim"),
        "data.params.meso_agg": ("last", "first", "mean", "sum"),
        "data.params.meso_time_span": ("all", "future"),
    }
    for key, allowed in choices.items():
        value = OmegaConf.select(cfg, key)
        if value not in allowed:
            errors.append(f"{key}={value!r} non ammesso, valori possibili: {allowed}")

    # Parametri derivati fissati a mano (nel yaml o con --set): devono
    # coincidere con la loro sorgente.
    derived = {
        "diffusion.network_config.params.in_channels": first_stage.params.latent_channels,
        "diffusion.network_config.params.vae_down_ratio": _first_stage_down_ratio(first_stage),
        "diffusion.network_config.params.input_size": data.resize // dit.vae_down_ratio,
        "diffusion.network_config.params.noise_image_num": data.noise_image_num,
        "diffusion.network_config.params.image_time_length": data.T,
        "diffusion.network_config.params.climate_channel": data.meso_channels,
        "diffusion.loss_fn_config.params.noise_image_num": data.noise_image_num,
        "diffusion.first_stage_config.params.in_channels": data.n_channels,
        "autoencoder.image_size": data.resize,
        "preprocessing.check_frames": data.T,
        "preprocessing.n_bands": data.n_channels,
    }
    # I params del first stage vengono dalla sezione `autoencoder`: la fase 2
    # deve ricostruire lo STESSO modello che la fase 1 ha addestrato.
    if first_stage.type == "vae":
        auto = cfg.autoencoder
        derived.update({
            "diffusion.first_stage_config.params.latent_channels": auto.latent_channels,
            "diffusion.first_stage_config.params.block_out_channels": auto.block_out_channels,
            "diffusion.first_stage_config.params.layers_per_block": auto.layers_per_block,
            "diffusion.first_stage_config.params.norm_num_groups": auto.norm_num_groups,
            "diffusion.first_stage_config.params.image_size": data.resize,
        })
    for key, expected in derived.items():
        value = OmegaConf.select(cfg, key)
        if value != expected:
            errors.append(f"{key}={value} incoerente: dovrebbe essere {expected} "
                          f"(e' derivato, meglio non fissarlo a mano)")

    pre = cfg.preprocessing
    if pre.limit < 0:
        errors.append(f"preprocessing.limit={pre.limit} must be >= 0 (0 = all the cubes)")
    if pre.val_frac < 0 or pre.test_frac < 0 or pre.val_frac + pre.test_frac >= 1:
        errors.append(f"preprocessing.val_frac={pre.val_frac} and test_frac={pre.test_frac} "
                      f"must be >= 0 and leave locations for train (sum < 1)")
    if pre.val_frac <= 0:
        errors.append("preprocessing.val_frac must be > 0: both phases validate on that split")
    if pre.dtype not in ("float16", "float32"):
        errors.append(f"preprocessing.dtype={pre.dtype!r} not allowed, values: ('float16', 'float32')")
    if cfg.autoencoder.T not in pre.times:
        errors.append(f"autoencoder.T={cfg.autoencoder.T} is not among preprocessing.times="
                      f"{list(pre.times)}: the preprocessing would not write its .npy files")
    if data.stats_time_key not in [str(t) for t in pre.times]:
        errors.append(f"data.params.stats_time_key={data.stats_time_key!r} is not among "
                      f"preprocessing.times={list(pre.times)}: stats.json would not have it")
    if cfg.autoencoder.scale_factor_batches < 1:
        errors.append("autoencoder.scale_factor_batches must be >= 1")
    if cfg.diffusion.scale_factor is not None and cfg.diffusion.scale_factor <= 0:
        errors.append(f"diffusion.scale_factor={cfg.diffusion.scale_factor} must be > 0 "
                      f"(or null to read it from the VAE checkpoint)")
    if cfg.wandb.mode not in WandbCfg.MODES:
        errors.append(f"wandb.mode={cfg.wandb.mode!r} not allowed, values: {WandbCfg.MODES}")
    if cfg.diffusion.latent_normalization not in DiffusionCfg.LATENT_NORMALIZATIONS:
        errors.append(f"diffusion.latent_normalization={cfg.diffusion.latent_normalization!r} "
                      f"not allowed, values: {DiffusionCfg.LATENT_NORMALIZATIONS}")
    if cfg.diffusion.latent_normalization == "scale" and cfg.diffusion.shift_factor is not None:
        errors.append(f"diffusion.shift_factor={cfg.diffusion.shift_factor} is not used with "
                      f"latent_normalization='scale': leave it null, or use 'shift_scale'")
    channel_keys = ("channel_shift_factors", "channel_scale_factors")
    if cfg.diffusion.latent_normalization == "shift_scale_per_channel":
        for key in ("scale_factor", "shift_factor"):
            if cfg.diffusion[key] is not None:
                errors.append(f"diffusion.{key}={cfg.diffusion[key]} is not used with "
                              f"latent_normalization='shift_scale_per_channel': leave it null "
                              f"(the per-channel values are diffusion.{'channel_' + key + 's'})")
        n_channels = cfg.diffusion.network_config.params.in_channels
        for key in channel_keys:
            values = cfg.diffusion[key]
            if values is not None and len(values) != n_channels:
                errors.append(f"diffusion.{key} has {len(values)} values, the latents have "
                              f"{n_channels} channels")
        if (cfg.diffusion.channel_scale_factors is not None
                and min(cfg.diffusion.channel_scale_factors) <= 0):
            errors.append("diffusion.channel_scale_factors must all be > 0")
    else:
        for key in channel_keys:
            if cfg.diffusion[key] is not None:
                errors.append(f"diffusion.{key} is only used with latent_normalization="
                              f"'shift_scale_per_channel': leave it null")

    ens = cfg.ens
    if ens.scoring not in EnsCfg.SCORING_MODES:
        errors.append(f"ens.scoring={ens.scoring!r} must be one of {EnsCfg.SCORING_MODES}")
    if ens.enabled and not (ens.validation or ens.train):
        errors.append("ens.enabled=true but ens.validation and ens.train are both false: "
                      "nothing would be computed")
    if ens.train and (ens.train_cubes < 1 or ens.train_every_epochs < 1):
        errors.append("ens.train_cubes and ens.train_every_epochs must be >= 1")
    if ens.enabled and cfg.run.val_num_samples < 1:
        errors.append("run.val_num_samples must be >= 1")
    if cfg.autoencoder.early_stopping_patience < 0 or cfg.autoencoder.early_stopping_min_delta < 0:
        errors.append("autoencoder.early_stopping_patience and "
                      "autoencoder.early_stopping_min_delta must be >= 0")
    if cfg.diffusion.n_forecast_examples < 0 or cfg.diffusion.forecast_lead_columns < 1:
        errors.append("diffusion.n_forecast_examples must be >= 0 and "
                      "diffusion.forecast_lead_columns >= 1")
    if cfg.autoencoder.n_recon_examples < 0 or cfg.autoencoder.examples_every < 0:
        errors.append("autoencoder.n_recon_examples and autoencoder.examples_every "
                      "must be >= 0")
    if not 0 < data.noise_image_num < data.T:
        errors.append(f"data.params.noise_image_num={data.noise_image_num} deve stare "
                      f"fra 1 e data.params.T-1={data.T - 1} (serve almeno un frame di contesto)")
    if data.resize % dit.vae_down_ratio:
        errors.append(f"data.params.resize={data.resize} non divisibile per il "
                      f"vae_down_ratio={dit.vae_down_ratio} del first stage")
    if dit.input_size % dit.patch_size:
        errors.append(f"input_size={dit.input_size} del DiT non divisibile per "
                      f"patch_size={dit.patch_size}")
    max_len = dit.temporal_module_kwargs.temporal_position_encoding_max_len
    if dit.image_time_length > max_len:
        errors.append(f"image_time_length={dit.image_time_length} supera "
                      f"temporal_position_encoding_max_len={max_len}")
    if data.meso_time_span != "all":
        errors.append("data.params.meso_time_span deve essere 'all': climate_var del DiT "
                      "deve coprire tutti gli image_time_length istanti")

    if errors:
        raise ValueError("Configurazione non valida:\n  - " + "\n  - ".join(errors))


def effective_learning_rate(cfg: TrainCfg, num_processes: int = 1) -> float:
    """
    Learning rate da passare all'optimizer. Con run.scale_lr il valore del
    config, tarato su learning_rate_base_batch_size, scala linearmente con il
    batch effettivo (batch_size x gradient accumulation x numero di GPU).
    """
    acc = cfg.accelerate
    if not cfg.run.scale_lr:
        return acc.learning_rate
    effective_batch = acc.gradient_accumulation_steps * cfg.data.params.batch_size * num_processes
    return acc.learning_rate * effective_batch / acc.learning_rate_base_batch_size


def save_config(cfg: TrainCfg, path: Union[str, Path], resolve: bool = False) -> None:
    """
    Salva `cfg` in yaml. Con resolve=False le interpolazioni restano tali e il
    file si ricarica con load_config(); con resolve=True si salvano i valori
    effettivi, utile come traccia di un run.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(config_to_str(cfg, resolve=resolve), encoding="utf-8")


def config_to_str(cfg: TrainCfg, resolve: bool = True) -> str:
    if not resolve:
        return OmegaConf.to_yaml(cfg)
    # le tuple prodotte dai resolver (betas) diventano liste: yaml non ha tuple
    # e scriverebbe un tag python/tuple che OmegaConf non sa ricaricare
    return OmegaConf.to_yaml(_tuples_to_lists(OmegaConf.to_container(cfg, resolve=True)))


def _tuples_to_lists(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(v) for v in obj]
    return obj


# Lancio dall'IDE (Run/Debug, senza argomenti): valgono questi. Da terminale
# bastano --cfgdir e --project_name per scavalcarli.
DEFAULT_CONFIG_FILE = "configuration_0.yaml"
DEFAULT_PROJECT_NAME = "prova_config"


def main():
    # Loads and validates the config without training anything (and without
    # creating a run folder):
    #     python -m src.config                                     (o Run dall'IDE)
    #     python -m src.config --cfgdir altro.yaml --project_name x
    cfg = config_from_cli(config_file=DEFAULT_CONFIG_FILE,
                          project_name=DEFAULT_PROJECT_NAME)
    print(config_to_str(cfg))


if __name__ == '__main__':
    main()
