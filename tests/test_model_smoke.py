"""Smoke tests for ZeusTransformerModel after the loss_type / decoder refactor.

These don't load the released checkpoint — they just verify the model can be
constructed, runs a forward pass, and produces the expected shape.
"""
import torch
import pytest

from zeus import _config as c


def _build_model():
    from zeus.model.encoders import Linear
    from zeus.model.zeus import ZeusTransformerModel

    encoder = Linear(c.INPUT_DIM, c.EMBED_DIM, replace_nan_by_zero=True)
    return ZeusTransformerModel(
        encoder,
        ninp=c.EMBED_DIM,
        nhead=c.N_HEAD,
        nhid=c.HID_DIM,
        nlayers=c.N_LAYERS,
        dropout=c.DROPOUT,
        n_clusters=c.NUM_GAUSSIANS,
        efficient_eval_masking=c.EFFICIENT_EVAL_MASKING,
    )


def test_constructor_takes_no_loss_type_or_decoder_args():
    """The refactored constructor should not accept loss_type, decoder, n_out, or distance_based_logit."""
    import inspect
    from zeus.model.zeus import ZeusTransformerModel
    sig = inspect.signature(ZeusTransformerModel.__init__)
    for forbidden in ("loss_type", "decoder", "n_out", "distance_based_logit"):
        assert forbidden not in sig.parameters, (
            f"{forbidden} should have been removed in the refactor"
        )


def test_forward_returns_correct_shape():
    """Forward should return (N + NUM_GAUSSIANS, 1, EMBED_DIM)."""
    model = _build_model()
    model.eval()
    n = 5
    x = torch.randn(n, 1, c.INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (n + c.NUM_GAUSSIANS, 1, c.EMBED_DIM)


def test_model_has_no_configs_import():
    """zeus.model.zeus should no longer import from zeus.configs."""
    import zeus.model.zeus as zm
    src = open(zm.__file__).read()
    assert "from zeus.configs" not in src
    assert "import zeus.configs" not in src
