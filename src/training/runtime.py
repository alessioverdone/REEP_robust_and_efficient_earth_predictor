"""
Stato di runtime di un training.

Qui sta tutto quello che si RICAVA dal config ma non e' configurazione, e che
quindi non va ne' nello schema di src/config.py ne' nel yaml: l'Accelerator, i
dtype di torch, gli handler di logging, il timestamp del run. Sono oggetti vivi,
non serializzabili, e dipendono dall'hardware su cui gira il lancio.

La divisione dei compiti all'avvio di un run e' questa:

    cfg     = config_from_cli()                        # configurazione (src/config.py)
    outputs = create_shared_run_outputs(...)           # run folder     (src/training/run_outputs.py)
    rt      = setup_runtime(cfg, outputs)              # stato runtime  (questo file)
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from src.config import TrainCfg, effective_learning_rate
from src.training.run_outputs import RunOutputs
from src.training.tracking import Tracker

logger = get_logger(__name__, log_level="INFO")

# dtype corrispondente a ogni valore di accelerate.mixed_precision
_MIXED_PRECISION_DTYPES = {
    "no": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class StdoutToLogger:
    """
    Manda al logger anche quello che viene scritto con print().

    Il pacchetto sgm/ ha qualche centinaio di print sparsi (caricamento pesi,
    scelta del backend di attention, setup del sampler...): riscriverli tutti
    sarebbe invasivo, ma senza questo non finirebbero nel file di log del run.
    La console continua a vedere l'output originale, e stderr non viene toccato
    cosi' le progress bar di tqdm restano fuori dal file.
    """

    def __init__(self, stream, log, level=logging.INFO):
        self._stream = stream
        self._log = log
        self._level = level
        self._buffer = ""

    def write(self, data):
        self._stream.write(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._log.log(self._level, line.rstrip())
        return len(data)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


@dataclass
class Runtime:
    accelerator: Accelerator
    # dtype dei moduli ALLENATI (DiT + conditioner)
    weight_dtype: torch.dtype
    # dtype del first stage, congelato: resta sempre quello di mixed_precision
    first_stage_dtype: torch.dtype
    # learning rate da passare all'optimizer (gia' scalato se run.scale_lr)
    learning_rate: float
    # name of the run folder (its start time)
    datenow: str
    local_rank: int
    # live report on W&B: set by start_tracker() once the config is resolved,
    # a no-op until then and with wandb.enabled=false
    tracker: Tracker = field(default_factory=Tracker)


def _resolve_dtypes(cfg: TrainCfg, accelerator: Accelerator) -> tuple[torch.dtype, torch.dtype]:
    """
    Dtype dei moduli allenati e del first stage.

    MODIFICA 1 (accelerate.weight_dtype).
    'auto' (default) = comportamento storico: i pesi seguono mixed_precision,
    quindi con bf16 il modello viene castato a bf16 e AdamW aggiorna
    direttamente parametri bf16 -- che hanno 8 bit di mantissa, per cui un
    update piu' piccolo di ~2^-8 rispetto al peso viene perso.
    'float32' = i pesi restano in fp32 e l'autocast di Accelerate continua a
    calcolare in bf16: e' la mixed precision standard, con i master weights in
    fp32. Costa circa il doppio di memoria per pesi/gradienti/stati AdamW, e
    raddoppia la dimensione dei checkpoint.

    Il first stage (VAE) e' congelato e non passa da accelerator.prepare(),
    quindi gira sempre al dtype dei suoi pesi, senza autocast: lo si tiene a
    quello di mixed_precision anche quando i moduli ALLENATI passano a fp32,
    cosi' l'encode dei 240 frame per step non rallenta. Il cast ai suoi confini
    lo fanno encode/decode_first_stage.
    """
    # accelerate puo' sovrascrivere il valore del config (variabili d'ambiente,
    # `accelerate config`, flag di `accelerate launch`): vince quello vero, e il
    # config lo registra cosi' il file salvato in configs/ dice com'e' andata.
    mixed_precision = accelerator.mixed_precision
    if mixed_precision not in _MIXED_PRECISION_DTYPES:
        raise SystemExit(f"accelerate.mixed_precision={mixed_precision!r} non gestito, "
                         f"valori possibili: {tuple(_MIXED_PRECISION_DTYPES)}")
    cfg.accelerate.mixed_precision = mixed_precision

    first_stage_dtype = _MIXED_PRECISION_DTYPES[mixed_precision]
    weight_dtype = (torch.float32 if cfg.accelerate.weight_dtype == "float32"
                    else first_stage_dtype)
    return weight_dtype, first_stage_dtype


def _basic_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def _add_log_file(path: str) -> str:
    """
    Aggiunge il file di log del run e ci dirotta anche i print().

    L'handler va sul logger ROOT, non su quello di questo modulo: cosi' nel file
    finiscono anche i messaggi del dataloader, di diffusers e di accelerate.
    """
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    sys.stdout = StdoutToLogger(sys.stdout, logging.getLogger("stdout"))
    return file_handler.baseFilename


def _setup_logging(outputs: RunOutputs,
                   accelerator: Accelerator) -> None:
    """Log a schermo su tutti i processi, su file solo sul main."""
    _basic_logging()
    logger.info(accelerator.state, main_process_only=False)

    if not accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        return

    logger.info(f"Log del run in {_add_log_file(outputs.log_file)}")

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()


def setup_runtime(cfg: TrainCfg,
                  outputs: RunOutputs) -> Runtime:
    """
    Prepara l'ambiente di un run a partire da `cfg`: Accelerator, logging, seed,
    TF32, dtype e learning rate effettivo. The log goes to the run's
    artifacts/train.log and the timestamp is the run folder's name.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerate.gradient_accumulation_steps,
        mixed_precision=cfg.accelerate.mixed_precision,
        project_config=ProjectConfiguration(project_dir=outputs.root,
                                            logging_dir=outputs.artifacts),
    )

    _setup_logging(outputs,
                   accelerator)

    set_seed(cfg.run.seed)
    if cfg.run.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype, first_stage_dtype = _resolve_dtypes(cfg, accelerator)
    learning_rate = effective_learning_rate(cfg, accelerator.num_processes)
    if cfg.run.scale_lr:
        logger.info(f"learning rate scalato: {cfg.accelerate.learning_rate} -> {learning_rate}")

    # Riepilogo delle quattro varianti, cosi' il log del run dice da solo che
    # configurazione era: artifacts/config.yaml e' la traccia
    # completa, questa riga e' quella che si legge al volo in coda al file.
    net = cfg.diffusion.network_config.params
    weighting = cfg.diffusion.denoiser_config.params.weighting_config.params
    logger.info(
        "[cfg] varianti attive: "
        f"weight_dtype={cfg.accelerate.weight_dtype} (allenati={weight_dtype}, "
        f"vae={first_stage_dtype}) | "
        f"time_embed_repeat={net.time_embed_repeat} | "
        f"minsnr_mode={weighting.mode}(gamma={weighting.gamma}) | "
        f"latent_normalization={cfg.diffusion.latent_normalization} (values below, "
        f"after the VAE is loaded)"
    )

    return Runtime(
        accelerator=accelerator,
        weight_dtype=weight_dtype,
        first_stage_dtype=first_stage_dtype,
        learning_rate=learning_rate,
        datenow=outputs.timestamp,
        # lo passa il launcher (torchrun / accelerate launch)
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )


# =========================== FASE 1: AUTOENCODER ===========================
# Il training del VAE e' torch puro, senza accelerate: il runtime qui si riduce
# al device e al timestamp, ma la struttura dello script resta la stessa della
# fase 2 (cfg -> dirs -> rt).

@dataclass
class AutoencoderRuntime:
    device: torch.device
    datenow: str
    # live report on W&B, see Runtime.tracker
    tracker: Tracker = field(default_factory=Tracker)


def setup_autoencoder_runtime(cfg: TrainCfg,
                              outputs: RunOutputs) -> AutoencoderRuntime:
    """
    Device, seed, TF32 e logging su file per il training dell'autoencoder. The
    log goes to the run's artifacts/ and the timestamp is the run folder's name.
    """
    # No Accelerator in this phase, so the accelerate logger would raise: use plain logging.
    ae_logger = logging.getLogger(__name__)

    _basic_logging()
    ae_logger.info(f"Log del run in {_add_log_file(outputs.log_file)}")

    set_seed(cfg.run.seed)
    if cfg.run.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_logger.info(f"[autoencoder] device={device} | seed={cfg.run.seed} | "
                   f"latent_channels={cfg.autoencoder.latent_channels} | "
                   f"block_out_channels={list(cfg.autoencoder.block_out_channels)}")
    return AutoencoderRuntime(device=device, datenow=outputs.timestamp)


# =========================== CONDITION PREDICTOR ===========================
# Torch puro come la fase 1: device, timestamp e tracker.

@dataclass
class ConditionRuntime:
    device: torch.device
    datenow: str
    # live report on W&B, see Runtime.tracker
    tracker: Tracker = field(default_factory=Tracker)


def setup_condition_runtime(cfg: TrainCfg,
                            outputs: RunOutputs) -> ConditionRuntime:
    """
    Device, seed, TF32 and file logging for the training of the Condition
    Predictor. The log goes to the run's artifacts/ and the timestamp is the
    run folder's name.
    """
    cond_logger = logging.getLogger(__name__)

    _basic_logging()
    cond_logger.info(f"Log del run in {_add_log_file(outputs.log_file)}")

    set_seed(cfg.run.seed)
    if cfg.run.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond = cfg.condition
    cond_logger.info(f"[condition] device={device} | seed={cfg.run.seed} | model={cond.model.name} | "
                     f"{cond.n_context} -> {cond.n_predict} frames x {cond.n_channels} variables | "
                     f"meso_agg={cond.meso_agg}")
    return ConditionRuntime(device=device, datenow=outputs.timestamp)
