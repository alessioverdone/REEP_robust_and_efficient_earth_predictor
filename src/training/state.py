"""
Stato di un training in corso: a che punto e', e cosa ha misurato.

Divisione dei compiti, insieme agli altri due moduli:

    cfg   (src/config.py)           cosa si vuole addestrare -- si legge da yaml
    rt    (src/training/runtime.py) su cosa gira -- accelerator, dtype, log
    state (questo file)             a che punto e' -- progresso, resume, metriche

configure_train() costruisce lo stato (leggendo il checkpoint da cui si
riprende, se c'e') e train_loop() lo fa avanzare: cosi' le due funzioni si
scambiano un oggetto solo invece di sette valori sciolti.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class TrainMetrics:
    """
    Le misure del run, di due famiglie.

    Di task: le metriche dell'ultima validazione (`val_metrics`) con epoca, step
    e numero di cubi su cui sono calcolate. Nella fase 2 sono le RMSE prodotte
    da validation_rmse(), nella fase 1 loss/mse/psnr/ssim di evaluate_vae().

    Computazionali: quanto tempo se n'e' andato in training e quanto in
    validazione -- che e' inferenza, molto piu' lenta per campione -- e il picco
    di memoria sulla GPU. Sono i numeri che servono per la tabella dei costi.
    """

    # computazionali
    start_time: float = 0.0
    inference_time: float = 0.0             # tempo passato dentro le validazioni

    # di task (ultima validazione fatta)
    val_metrics: dict = field(default_factory=dict)
    n_cubes: int = 0
    val_epoch: int = -1
    val_global_step: int = -1

    def start(self) -> None:
        """Azzera cronometro e picco di memoria. Da chiamare a inizio training,
        non prima: il caricamento del checkpoint alloca memoria che non c'entra
        con il costo di un passo di training."""
        self.start_time = time.time()
        self.inference_time = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_validation(self, val_metrics: dict, n_cubes: int, epoch: int,
                          global_step: int, elapsed: float) -> None:
        """Una validazione conclusa: metriche di task piu' il tempo che e' costata."""
        self.val_metrics = dict(val_metrics)
        self.n_cubes = n_cubes
        self.val_epoch = epoch
        self.val_global_step = global_step
        self.inference_time += elapsed

    @property
    def wall_time(self) -> float:
        """Secondi dall'inizio del training."""
        return time.time() - self.start_time

    @property
    def train_time(self) -> float:
        """Tempo di solo training, cioe' al netto delle validazioni."""
        return self.wall_time - self.inference_time

    @staticmethod
    def gpu_memory_gb() -> Optional[tuple]:
        """(allocata, riservata) al picco, in GB. None senza GPU."""
        if not torch.cuda.is_available():
            return None
        return (torch.cuda.max_memory_allocated() / (1024 ** 3),
                torch.cuda.max_memory_reserved() / (1024 ** 3))

    def summary_lines(self) -> list:
        """Il riepilogo finale, una riga per elemento: lo stampa il chiamante
        con il suo logger."""
        lines = [
            "=" * 60,
            "               TRAINING & INFERENCE SUMMARY              ",
            "=" * 60,
            f"Tempo di Training Netto : {self.train_time / 3600:.2f} ore ({self.train_time:.2f} s)",
            f"Tempo di Inferenza Totale: {self.inference_time / 3600:.2f} ore ({self.inference_time:.2f} s)",
            f"Tempo Totale Esecuzione  : {self.wall_time / 3600:.2f} ore ({self.wall_time:.2f} s)",
        ]
        memory = self.gpu_memory_gb()
        if memory is not None:
            allocated, reserved = memory
            lines.append(f"Memoria GPU Max Allocata : {allocated:.2f} GB")
            lines.append(f"Memoria GPU Max Riservata: {reserved:.2f} GB")
        if self.val_metrics:
            lines.append(f"Ultima validazione       : epoca {self.val_epoch}, "
                         f"step {self.val_global_step}, {self.n_cubes} cubi")
            for key, value in self.val_metrics.items():
                lines.append(f"  {key:24} {value:.6f}")
        lines.append("=" * 60)
        return lines


@dataclass
class TrainState:
    """
    Dove siamo nel training. I primi quattro campi governano il loop, il resto
    e' quello che finisce in training_info.json, history.csv e nelle figure.
    """

    global_step: int = 0
    # epoca da cui partire: 0 per un run nuovo, quella del checkpoint se si riprende
    first_epoch: int = 0
    # batch gia' consumati nell'epoca da cui si riprende, saltati una volta sola
    resume_step: int = 0
    # minimo storico del criterio di selezione del modello migliore
    best_latter_rmse: float = float("inf")

    best_ckpt_path: str = ""
    history_csv: str = ""
    # one row per (validation, lead time): the RMSE curve over the predicted frames
    lead_time_csv: str = ""
    metrics: TrainMetrics = field(default_factory=TrainMetrics)

    # What training_info.json reports. `static` is filled once at start (run,
    # dataset, first stage, model); the rest advances with the loop.
    static: dict = field(default_factory=dict)
    mode: str = "train"                     # train | validate_only
    status: str = "running"                 # running | completed | failed | interrupted
    error: Optional[str] = None
    started_at: str = ""
    ended_at: Optional[str] = None
    resumed_from: Optional[str] = None
    epochs_completed: int = 0
    best_epoch: Optional[int] = None        # 1-based, as in history.csv
    best_global_step: Optional[int] = None
    train_metrics: dict = field(default_factory=dict)       # last epoch, mean over its steps
    val_metrics: dict = field(default_factory=dict)         # last validation
    val_epoch: Optional[int] = None         # 1-based epoch of the last validation
    best_val_metrics: dict = field(default_factory=dict)
    # pixel RMSE per lead time (model and context-mean baseline), last and best validation
    val_lead_time: dict = field(default_factory=dict)
    best_val_lead_time: dict = field(default_factory=dict)
    # ENS statistics of the last validation / training sample (config section `ens`)
    ens_val: dict = field(default_factory=dict)
    ens_train: dict = field(default_factory=dict)
    epoch_seconds: list = field(default_factory=list)       # training only, per epoch
    cubes_per_second: list = field(default_factory=list)    # training throughput, per epoch
    checkpoints: dict = field(default_factory=dict)         # folder name -> info


# =========================== FASE 1: AUTOENCODER ===========================

@dataclass
class AutoencoderState:
    """
    Dove siamo nel training dell'autoencoder. Lo stesso ruolo di TrainState per
    la fase 2, con quello che serve qui: da che epoca ripartire e qual e' il
    minimo (o massimo) storico della metrica scelta con save_best_by.
    """

    start_epoch: int = 0
    global_step: int = 0                    # batch di training visti finora
    # valore di riferimento del criterio di selezione, gia' orientato: si
    # confronta sempre con <, e per psnr/ssim ci si mette dentro il segno meno
    best_metric: float = float("inf")
    history_csv: str = ""
    metrics: TrainMetrics = field(default_factory=TrainMetrics)

    # What training_info.json reports. `static` is filled once at start (run,
    # dataset, model, FLOPs); the rest advances with the loop.
    static: dict = field(default_factory=dict)
    status: str = "running"                 # running | completed | failed | interrupted
    error: Optional[str] = None
    started_at: str = ""
    ended_at: Optional[str] = None
    resumed_from: Optional[str] = None
    epochs_completed: int = 0
    best_epoch: Optional[int] = None        # 1-based, as in history.csv
    train_metrics: dict = field(default_factory=dict)       # last epoch
    val_metrics: dict = field(default_factory=dict)         # last epoch
    best_val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)        # best.pt on the test set
    epoch_seconds: list = field(default_factory=list)       # training only, per epoch
    images_per_second: list = field(default_factory=list)   # training throughput, per epoch
    test_seconds: float = 0.0
    lr_reductions: list = field(default_factory=list)       # [{"epoch", "lr"}]
    # early stopping: reference value (oriented to the minimum) and epochs since
    # it last improved by more than min_delta
    early_stop_reference: float = float("inf")
    epochs_without_improvement: int = 0
    stopped_early: bool = False
    # best weights on CPU: the test uses them also when no checkpoint is saved
    best_state_dict: Optional[dict] = None
    checkpoints: dict = field(default_factory=dict)         # file name -> info
    # diffusion.scale_factor / shift_factor measured on the best weights at the
    # end (values, latent mean/std, per channel too, images used); empty until
    # measured
    scale_factor: dict = field(default_factory=dict)


# Metriche di evaluate_vae() per cui vince il valore PIU' ALTO: si confrontano
# cambiate di segno, cosi' il resto del codice guarda sempre il minimo.
HIGHER_IS_BETTER = ("psnr", "ssim")


def monitored_value(val_metrics: dict, save_best_by: str) -> float:
    """Valore da confrontare con best_metric, orientato al minimo."""
    value = float(val_metrics[save_best_by])
    return -value if save_best_by in HIGHER_IS_BETTER else value
