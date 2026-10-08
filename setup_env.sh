#!/bin/bash
#
# Crea da zero l'ambiente virtuale del progetto. Da lanciare dalla root:
#
#     ./setup_env.sh
#     source venv/bin/activate
#
# Se esiste gia' un venv viene rinominato in venv.old invece di essere
# cancellato: finche' il nuovo non e' verificato, il vecchio resta li'.
#
# Le versioni stanno in requirements.txt. PyTorch fa eccezione e viene
# installato prima, dal suo indice, per avere le wheel CUDA 12.8: sono le
# prime con i kernel sm_120 delle GPU Blackwell (RTX 50xx).

set -e

VENV="venv"

if [ -d "$VENV" ]; then
    BACKUP="${VENV}.old.$(date +%Y%m%d_%H%M%S)"
    echo ">>> Ambiente esistente spostato in $BACKUP"
    mv "$VENV" "$BACKUP"
fi

echo ">>> Creo l'ambiente virtuale con Python 3.10..."
uv venv "$VENV" --python=3.10

# --python punta uv al venv giusto senza doverlo attivare; dentro al venv non
# c'e' pip (uv venv non lo installa), quindi si usa sempre `uv pip`.
PIP="uv pip install --python $VENV/bin/python"

# PyTorch prima e da solo, dal suo indice: le wheel cu128 sono le prime con i
# kernel sm_120 (Blackwell / RTX 50xx). L'indice di PyTorch NON va lasciato
# attivo per il resto dell'installazione -- ospita copie ferme al 2023 di altri
# pacchetti (torchmetrics 1.0.3 e compagnia) e uv, per non esporsi agli attacchi
# di dependency confusion, considera solo il primo indice in cui trova un nome.
echo ">>> Installo PyTorch 2.8.0 (CUDA 12.8)..."
$PIP torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Il resto da PyPI. torch e' gia' a posto: 2.8.0+cu128 soddisfa il pin
# torch==2.8.0 del file, quindi non viene reinstallato.
echo ">>> Installo le altre dipendenze da requirements.txt..."
$PIP -r requirements.txt

echo ""
echo ">>> Verifica: torch, GPU e import della pipeline"
"$VENV/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"arch compilate: {torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} - compute capability sm_{cap[0]}{cap[1]}")
    # non basta che cuda sia "available": se l'arch della GPU non e' fra quelle
    # compilate, e' proprio qui che si scopre (no kernel image available)
    x = torch.randn(512, 512, device="cuda")
    print(f"prodotto di prova sulla GPU: {(x @ x).sum().item():.3f}")
else:
    print("ATTENZIONE: nessuna GPU visibile a torch")
PY

PYTHONPATH=.:src/model:other "$VENV/bin/python" -c "import diffusion_utils; print('import della pipeline: ok')"

echo ""
echo ">>> Fatto. Per usarlo:"
echo "      source $VENV/bin/activate"
echo "      export PYTHONPATH=.:src/model:other"
