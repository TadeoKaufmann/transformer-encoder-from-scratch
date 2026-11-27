import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import accuracy_by_length  # noqa: E402
from model import TransformerEncoderClassifier  # noqa: E402
from train_imdb import EXPERIMENTS, INITS, ImdbDataset, init_xavier, parse_args  # noqa: E402

TEXTS = [{"text": "a good good movie", "label": 1}, {"text": "bad " * 40, "label": 0}]


@pytest.fixture
def tokenizer():
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    tok.train_from_iterator((t["text"] for t in TEXTS), trainer=trainer)
    return tok


def test_short_review_is_padded_and_masked(tokenizer):
    item = ImdbDataset(TEXTS, tokenizer, seq_len=10)[0]
    ids = item["encoder_input"].tolist()
    assert len(ids) == 10
    assert ids[0] == tokenizer.token_to_id("[SOS]")
    assert ids[5] == tokenizer.token_to_id("[EOS]")
    assert ids[6:] == [tokenizer.token_to_id("[PAD]")] * 4
    assert item["encoder_mask"].shape == (1, 1, 10)
    assert item["encoder_mask"].sum().item() == 6


def test_long_review_is_truncated_but_keeps_eos(tokenizer):
    item = ImdbDataset(TEXTS, tokenizer, seq_len=10)[1]
    ids = item["encoder_input"].tolist()
    assert len(ids) == 10
    assert ids[-1] == tokenizer.token_to_id("[EOS]")
    assert item["encoder_mask"].sum().item() == 10


def test_accuracy_by_length_buckets():
    table = accuracy_by_length([10, 62, 63, 600], [1, 0, 1, 1]).set_index("bucket")
    assert table.loc["1-62", "n"] == 2
    assert table.loc["1-62", "accuracy"] == pytest.approx(0.5)
    assert table.loc["63-126", "n"] == 1
    assert table.loc["511+", "n"] == 1


def test_five_configurations_are_the_ones_reported():
    assert [e["name"] for e in EXPERIMENTS] == ["small", "base", "wider", "deep", "longer_seq"]


@pytest.mark.parametrize(
    "init, embedding_changes, layers_change",
    [("xavier", True, True), ("xavier_embedding", True, False), ("xavier_layers", False, True)],
)
def test_init_variants_touch_only_their_part(init, embedding_changes, layers_change):
    torch.manual_seed(0)
    model = TransformerEncoderClassifier(vocab_size=30, seq_len=8, num_classes=2, d_model=16, N=1, h=2, d_ff=32)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    init_xavier(model, INITS[init])
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    embedding = {name for name, p in before.items() if p.dim() > 1 and name.startswith("embed.")}
    layers = {name for name, p in before.items() if p.dim() > 1} - embedding
    assert changed == (embedding if embedding_changes else set()) | (layers if layers_change else set())


def test_training_plots_do_not_open_a_gui_backend(tmp_path):
    """Several trainings in one process crashed when figures used the Tk backend."""
    root = Path(__file__).resolve().parents[1]
    history = {"experiment": "x", "epoch": [1], "train_loss": [0.5], "train_acc": [0.5], "val_acc": [0.5]}
    code = (
        "import matplotlib; from train_imdb import plot_results; "
        f"plot_results([{history!r}], {str(tmp_path)!r}); print(matplotlib.get_backend())"
    )
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1].lower() == "agg"


def test_cli_defaults():
    args = parse_args([])
    assert args.init == "default" and args.seed is None and args.experiments is None
