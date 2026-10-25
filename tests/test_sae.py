"""Tests for lsne.sae (TopK SAE in sparsify format). Unit tests use random tensors;
the hub download test is slow/network and only runs with LSNE_SLOW=1."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import SAE_REPOS, HookSite
from lsne.sae import (TopKSAE, check_sparsify_cfg, feature_direction, feature_table, fvu,
                      language_selective_features, sae_fvu, sae_repo_for)


def random_sae(d=32, F=96, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(F, d, generator=g) / d ** 0.5
    b_enc = 0.1 * torch.randn(F, generator=g)
    W_dec = torch.randn(F, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    b_dec = 0.1 * torch.randn(d, generator=g)
    return TopKSAE.from_tensors(W_enc, b_enc, W_dec, b_dec, k)


def orthonormal_sae(d=32, F=32, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    W = Q.T[:F].contiguous()                       # orthonormal rows
    return TopKSAE.from_tensors(W, torch.zeros(F), W.clone(), torch.zeros(d), k), W


def test_from_tensors_shapes_and_validation():
    sae = random_sae()
    assert sae.d_in == 32 and sae.num_latents == 96 and sae.k == 8
    assert sae.W_enc.shape == (96, 32) and sae.W_dec.shape == (96, 32)
    assert sae.b_enc.shape == (96,) and sae.b_dec.shape == (32,)
    assert not any(p.requires_grad for p in sae.parameters())
    with pytest.raises(ValueError):
        TopKSAE.from_tensors(torch.zeros(4, 3), torch.zeros(4), torch.zeros(4, 2), torch.zeros(3), 2)
    with pytest.raises(ValueError):
        TopKSAE.from_tensors(torch.zeros(4, 3), torch.zeros(4), torch.zeros(4, 3), torch.zeros(3), 9)


def test_encode_decode_uses_exactly_k_latents():
    sae = random_sae(d=32, F=96, k=8)
    x = torch.randn(10, 32, generator=torch.Generator().manual_seed(1))
    top_acts, top_idx, pre = sae.encode(x)
    assert top_acts.shape == (10, 8) and top_idx.shape == (10, 8) and pre.shape == (10, 96)
    assert torch.allclose(pre, (x - sae.b_dec) @ sae.W_enc.T + sae.b_enc, atol=1e-6)
    assert (top_acts >= 0).all()
    acts_full = torch.relu(pre)
    for i in range(10):
        assert len(set(top_idx[i].tolist())) == 8                       # k distinct latents
        assert torch.allclose(top_acts[i], acts_full[i, top_idx[i]])
        # the selected ones are the k largest post-ReLU values
        assert float(top_acts[i].min()) >= float(acts_full[i].sort(descending=True).values[7]) - 1e-6
    x_hat = sae.decode(top_acts, top_idx)
    manual = torch.stack([(top_acts[i][:, None] * sae.W_dec[top_idx[i]]).sum(0) + sae.b_dec for i in range(10)])
    assert torch.allclose(x_hat, manual, atol=1e-5)
    dense = sae.features(x)
    assert dense.shape == (10, 96) and int((dense > 0).sum(-1).max()) <= 8
    assert torch.allclose(dense @ sae.W_dec + sae.b_dec, x_hat, atol=1e-5)
    # batched leading dims
    x3 = x.reshape(2, 5, 32)
    xh3, f3 = sae(x3)
    assert xh3.shape == (2, 5, 32) and torch.allclose(xh3.reshape(10, 32), x_hat, atol=1e-6)


def test_fvu_range_and_consistency():
    sae = random_sae()
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(2))
    x_hat, f = sae(x)
    assert isinstance(f, float) and np.isfinite(f) and f >= 0
    assert np.isclose(f, fvu(x, x_hat)) and np.isclose(f, sae_fvu(sae, x), rtol=1e-5)
    assert np.isclose(sae_fvu(sae, x, chunk=7), f, rtol=1e-5)          # chunking is exact
    # orthonormal partial projection on centred data: FVU in [0,1]
    sae_o, _ = orthonormal_sae(d=32, F=32, k=4)
    xc = x - x.mean(0, keepdim=True)
    _, fo = sae_o(xc)
    assert 0.0 <= fo <= 1.0
    # perfect reconstruction -> 0; predicting the mean -> 1
    assert np.isclose(fvu(x, x.clone()), 0.0)
    assert np.isclose(fvu(x, x.mean(0, keepdim=True).expand_as(x)), 1.0)


def test_orthonormal_sae_reconstructs_sparse_nonneg_combinations():
    d, F, k = 32, 32, 4
    sae, W = orthonormal_sae(d, F, k)
    g = torch.Generator().manual_seed(3)
    xs, supports = [], []
    for _ in range(6):
        idx = torch.randperm(F, generator=g)[:3]
        coef = torch.rand(3, generator=g) + 0.5
        xs.append((coef[:, None] * W[idx]).sum(0))
        supports.append(set(idx.tolist()))
    x = torch.stack(xs)
    top_acts, top_idx, _ = sae.encode(x)
    x_hat = sae.decode(top_acts, top_idx)
    assert torch.allclose(x_hat, x, atol=1e-5)
    for i in range(6):
        active = {int(j) for j, a in zip(top_idx[i], top_acts[i]) if float(a) > 1e-6}
        assert active == supports[i]
    assert sae(x)[1] < 1e-8


def test_feature_table_and_selective_features():
    sae = random_sae(d=16, F=40, k=4, seed=5)
    g = torch.Generator().manual_seed(4)
    xE = torch.randn(50, 16, generator=g)
    xD = torch.randn(50, 16, generator=g)
    xD[:, 0] += 3.0                                   # D samples shifted -> some features D-selective
    acts = torch.cat([xE, xD, torch.zeros(2, 16)])
    labels = ["E"] * 50 + ["D"] * 50 + ["N"] * 2       # foreign label ignored
    tab = feature_table(sae, acts, labels)
    assert isinstance(tab, pd.DataFrame) and len(tab) == 40
    assert list(tab.columns) == ["feature", "rate_E", "rate_D", "mean_E", "mean_D", "selectivity"]
    assert tab.attrs["n_E"] == 50 and tab.attrs["n_D"] == 50
    assert ((tab.rate_E >= 0) & (tab.rate_E <= 1) & (tab.rate_D >= 0) & (tab.rate_D <= 1)).all()
    assert (tab.mean_E >= 0).all() and (tab.mean_D >= 0).all()
    assert np.allclose(tab.selectivity, np.log((tab.rate_D + 1e-6) / (tab.rate_E + 1e-6)))
    # rates: each sample activates at most k features -> sum of rates <= k
    assert tab.rate_E.sum() <= 4 + 1e-6 and tab.rate_D.sum() <= 4 + 1e-6
    # cross-check one feature against the dense code
    dense_E = sae.features(xE)
    j = int(tab.feature.iloc[int(np.argmax(tab.rate_E.values))])
    assert np.isclose(tab.rate_E[j], float((dense_E[:, j] > 0).float().mean()))
    assert np.isclose(tab.mean_E[j], float(dense_E[:, j].mean()), atol=1e-6)
    sel = language_selective_features(tab, n=5)
    assert set(sel) == {"D", "E"} and len(sel["D"]) <= 5 and len(sel["E"]) <= 5
    assert all(isinstance(i, int) for i in sel["D"] + sel["E"])
    assert not set(sel["D"]) & set(sel["E"])
    assert tab.selectivity[sel["D"][0]] == tab.selectivity.max()
    assert tab.selectivity[sel["D"][0]] > 0 > tab.selectivity[sel["E"][0]]
    # feature directions are unit decoder rows
    v = feature_direction(sae, sel["D"][0])
    assert v.shape == (16,) and torch.isclose(v.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(v, sae.W_dec[sel["D"][0]] / sae.W_dec[sel["D"][0]].norm())
    with pytest.raises(ValueError):
        feature_table(sae, acts, labels[:-1])


def _sparsify_state(F=8, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"encoder.weight": torch.randn(F, d, generator=g), "encoder.bias": torch.zeros(F),
            "W_dec": torch.randn(F, d, generator=g), "b_dec": torch.zeros(d)}


def test_from_state_rejects_unsupported_sparsify_configs():
    """Round 2, G4 item 3: configs whose forward pass is not plain TopK must raise."""
    tensors = _sparsify_state()
    base = dict(expansion_factor=2, normalize_decoder=True, num_latents=8, k=2, d_in=4, signed=False)
    # the published format (sparsify defaults) builds
    sae = TopKSAE.from_state(tensors, base)
    assert sae.k == 2 and sae.num_latents == 8 and sae.cfg["signed"] is False
    assert TopKSAE.from_state(tensors, dict(base, activation="topk", multi_topk=True)).k == 2   # training-only flag
    for bad, needle in ((dict(base, signed=True), "signed"),
                        (dict(base, activation="groupmax"), "activation"),
                        (dict(base, skip_connection=True), "skip_connection"),
                        (dict(base, transcode=True), "transcode")):
        with pytest.raises(ValueError, match=needle):
            TopKSAE.from_state(tensors, bad)
        with pytest.raises(ValueError, match=needle):
            check_sparsify_cfg(bad)
    with pytest.raises(ValueError, match="W_skip"):
        TopKSAE.from_state(dict(tensors, W_skip=torch.zeros(4, 4)), base)
    with pytest.raises(ValueError, match="W_skip"):
        check_sparsify_cfg(base, dict(tensors, W_skip=torch.zeros(4, 4)))
    # several problems are all reported; missing keys take the defaults (accepted)
    with pytest.raises(ValueError) as ei:
        check_sparsify_cfg(dict(k=2, signed=True, transcode=True))
    assert "signed" in str(ei.value) and "transcode" in str(ei.value)
    check_sparsify_cfg(dict(k=2))
    check_sparsify_cfg(None)


def test_sae_repo_for():
    assert sae_repo_for("EleutherAI/pythia-160m") == ("EleutherAI/sae-pythia-160m-32k", "layers.{n}")
    assert sae_repo_for("pythia-70m") == SAE_REPOS["EleutherAI/pythia-70m"]
    repo, fmt = sae_repo_for("EleutherAI/pythia-410m")
    assert repo == "EleutherAI/sae-pythia-410m-65k" and fmt == "layers.{n}.mlp"
    assert sae_repo_for("EleutherAI/pythia-14m") is None
    # hookpoint naming matches config.HookSite.sae_hookpoint
    assert fmt.format(n=3) == HookSite(3, "mlp_output").sae_hookpoint
    assert "layers.{n}".format(n=10) == HookSite(11, "block_input").sae_hookpoint


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("LSNE_SLOW") != "1", reason="set LSNE_SLOW=1 to download SAEs")
def test_load_from_hub_pythia70m_layer3():
    """Downloads EleutherAI/sae-pythia-70m-32k hookpoint layers.3 (~134 MB) and checks the
    format; then measures FVU on real pythia-70m block-3 outputs (the SAE was trained on the
    final checkpoint, so FVU must be well below 1)."""
    from lsne.config import ModelSpec
    from lsne.model import load_model, run_with_cache

    repo, fmt = sae_repo_for("EleutherAI/pythia-70m")
    sae = TopKSAE.load_from_hub(repo, fmt.format(n=3), device="cpu")
    assert sae.num_latents == 32768 and sae.cfg["k"] == sae.k and sae.d_in == sae.cfg["d_in"]
    assert sae.W_enc.shape == (32768, sae.d_in) and sae.W_dec.shape == (32768, sae.d_in)
    if sae.cfg.get("normalize_decoder", False):
        assert torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(32768), atol=1e-2)
    mb70 = load_model(ModelSpec(name="EleutherAI/pythia-70m", revision="main", device="cpu"))
    assert mb70.d_model == sae.d_in
    ids = mb70.encode("The report is about the soldier who wanted to mediate this in reports and parables.")
    site = HookSite(3, "block_output")
    assert site.sae_hookpoint == "layers.3"
    h = run_with_cache(mb70, ids, [site]).resid[site.key][0]
    f = sae_fvu(sae, h)
    assert 0.0 <= f < 0.5
    top_acts, top_idx, _ = sae.encode(h)
    assert top_idx.shape == (ids.shape[1], sae.k)
