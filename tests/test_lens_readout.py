import pytest

torch = pytest.importorskip('torch')

from jspace_policy.lens_readout import check_selected_unembed, selected_unembed  # noqa: E402


class Adapter:
    def __init__(self, dtype=torch.float32, bias=False, cap=None):
        self._final_norm = torch.nn.RMSNorm(2, eps=1e-6, dtype=dtype)
        self._lm_head = torch.nn.Linear(2, 3, bias=bias, dtype=dtype)
        self._logit_softcap = cap
        with torch.no_grad():
            self._final_norm.weight.copy_(torch.tensor([0.1, 3.]))
            self._lm_head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 2.]]))
            if bias:
                self._lm_head.bias.copy_(torch.tensor([0.2, -0.4, 1.]))

    def unembed(self, residual):
        logits = self._lm_head(self._final_norm(residual.to(self._lm_head.weight.dtype)))
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('bias,cap', [(False, None), (True, 2.)])
def test_selected_matches_full_on_nonzero_scales(dtype, bias, cap):
    adapter = Adapter(dtype, bias, cap)
    states = torch.tensor([[2., 1.], [-3., 7.], [0.02, -0.01], [200., 100.]])
    check_selected_unembed(adapter, states, [2, 0, 1])


def test_missing_norm_can_reverse_rank_while_zero_check_passes():
    adapter = Adapter()
    states = torch.tensor([[2., 1.], [0., 0.]])
    bare = states @ adapter._lm_head.weight[:2].T
    corrected = selected_unembed(adapter, states, [0, 1])
    torch.testing.assert_close(bare[1], corrected[1])
    assert bare[0].argmax().item() == 0
    assert corrected[0].argmax().item() == 1
    with pytest.raises(ValueError, match='zero-only'):
        check_selected_unembed(adapter, states[1:], [0, 1])
