import torch


def check_nan_inf(t, s):
    assert not torch.isinf(t).any(), f"{s} is inf, {t}"
    assert not torch.isnan(t).any(), f"{s} is nan, {t}"
