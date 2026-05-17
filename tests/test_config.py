"""Frozen checkpoint constants are stable and match the released zeus.pt."""
from zeus import _config as c


def test_frozen_constants_present_and_correct():
    assert c.EMBED_DIM == 512
    assert c.N_HEAD == 4
    assert c.HID_DIM == 1024
    assert c.N_LAYERS == 12
    assert c.NUM_GAUSSIANS == 10
    assert c.INPUT_DIM == 30
    assert c.DROPOUT == 0.0
    assert c.EFFICIENT_EVAL_MASKING is True
