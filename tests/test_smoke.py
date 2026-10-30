"""Fast CPU checks that the core pipeline still works. No real data needed.

Run from the repo root:   python -m pytest tests/ -q
These catch broken imports, shape bugs, and metric regressions before a GPU job does.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import cv2
import numpy as np
import pytest
import torch

from dataset import OCTADataset, create_datasets, get_kfold_splits
from evaluate import dice, evaluate
from model import MODEL_REGISTRY, build_model, remap_legacy_state_dict
from train import CombinedLoss

TINY = dict(in_channels=1, num_classes=1, features=(8, 16, 32, 64))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_variant_builds_and_keeps_shape(name):
    model = build_model(name, **TINY).eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64), "output must be logits at input resolution"


def test_plain_unet_has_no_extra_modules():
    """FlexUNet with all flags off must stay the plain U-Net (the ablation depends on it)."""
    names = [n for n, _ in build_model("unet", **TINY).named_modules()]
    assert not any(k in n for n in names for k in ("att", "se.", "shortcut"))


def test_one_training_step_lowers_loss():
    torch.manual_seed(0)
    model = build_model("res_unet", **TINY)
    loss_fn = CombinedLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(2, 1, 64, 64)
    y = (x > 0.7).float()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_same_seed_same_init():
    torch.manual_seed(1)
    a = build_model("unet", **TINY).state_dict()
    torch.manual_seed(1)
    b = build_model("unet", **TINY).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_metrics_on_known_masks():
    gt = np.zeros((10, 10), np.uint8)
    gt[:5] = 1                      # 50 vessel pixels
    pred = np.zeros_like(gt)
    pred[:5, :5] = 1                # 25 pixels, all correct
    m = evaluate(pred, gt, prob_map=pred.astype(float))
    assert dice(gt, gt) == pytest.approx(1.0)
    assert m["dice"] == pytest.approx(2 * 25 / (25 + 50))
    assert m["precision"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1.0)


def test_kfold_splits_keep_test_fixed_and_cover_train():
    train = [f"img{i}" for i in range(30)]
    test = [f"t{i}" for i in range(9)]
    splits = get_kfold_splits(train, train, test, test, n_splits=5)
    assert splits["test"][0] == test
    val_union = sorted(v for f in splits["folds"] for v in f[2])
    assert val_union == sorted(train), "every training image is validated exactly once"
    again = get_kfold_splits(train, train, test, test, n_splits=5)
    assert [f[2] for f in again["folds"]] == [f[2] for f in splits["folds"]], "folds must be deterministic"


def test_dataset_loads_and_binarizes(tmp_path):
    img = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:30] = 255
    ip, mp = tmp_path / "i.png", tmp_path / "m.png"
    cv2.imwrite(str(ip), img)
    cv2.imwrite(str(mp), mask)
    x, y = OCTADataset([ip], [mp], augment=False)[0]
    assert x.shape == (1, 64, 64) and y.shape == (1, 64, 64)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    assert 0.0 <= x.min() and x.max() <= 1.0
    tr, va, te = create_datasets(([ip], [mp], [ip], [mp]), ([ip], [mp]))
    assert len(tr) == len(va) == len(te) == 1


def test_legacy_checkpoint_remap():
    model = build_model("unet", **TINY)
    current = model.state_dict()
    legacy = {k.replace(".conv.double_conv.", ".double_conv."): v for k, v in current.items()}
    model.load_state_dict(remap_legacy_state_dict(legacy))
