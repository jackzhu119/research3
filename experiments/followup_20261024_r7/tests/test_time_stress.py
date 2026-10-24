"""Declared finite-trial time stress preserves raw covariance and identities."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from paper3r7.diagnostics import perturb_episode


@dataclass
class Support:
    x: torch.Tensor
    y: torch.Tensor
    ids: tuple


def test_same_covariance_time_stress_preserves_paid_positions_labels_and_rng():
    generator=torch.Generator().manual_seed(71)
    x=torch.randn(28,11,32,generator=generator,dtype=torch.float64)
    y=torch.arange(28)%2
    support=Support(x[:8],y[:8],tuple(range(8)))
    episode=SimpleNamespace(support=support,query_x=x[8:],query_y=y[8:],query_ids=tuple(range(8,28)),run_ids=('unknown',)*20)
    original=x.clone();rng=torch.get_rng_state().clone()
    for scenario in ('time_cov_plus35','time_cov_minus35'):
        result,metadata=perturb_episode(episode,None,None,scenario)
        assert result.support.ids==support.ids and result.query_ids==episode.query_ids
        assert torch.equal(result.support.y,support.y) and torch.equal(result.query_y,episode.query_y)
        assert metadata['support_covariance_relative_error']<1e-12
        assert metadata['query_covariance_relative_error']<1e-12
        assert not metadata['filter_bank_covariance_invariance_claimed']
        assert not torch.equal(result.query_x,episode.query_x)
    assert torch.equal(x,original) and torch.equal(torch.get_rng_state(),rng)
