"""Tests for lsne.lambda_dynamics (Biggio et al. Algorithm 1 in activation space).

Model-based tests use pythia-14m (conftest fixtures) and keep the number of forward passes
small; the dynamics themselves are exercised with the (cheap) linear probe discriminant and
synthetic clouds.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.geometry import layer_covariance
from lsne.lambda_dynamics import (Discriminant, KDEMimicry, Trajectory, endpoint_effects,
                                  evasion_trajectory, feature_lambda_correlation,
                                  geometry_lambda_correlation, lambda_scale, lambda_sweep,
                                  make_discriminant, manifold_stats, partial_forward,
                                  trajectory_geometry)
from lsne.model import entropy_gradient, run_with_cache
from lsne.sae import TopKSAE


# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def site(mb):
    return HookSite(mb.n_layers - 1, "block_input")


@pytest.fixture(scope="module")
def h_unpert(mb, sample_ids, site):
    co = run_with_cache(mb, sample_ids, [site])
    return co.resid[site.key][0, -1].clone(), co.logits[0, -1].clone()


def two_clusters(d=16, n=200, sep=3.0, sigma=1.0, seed=0):
    """Source cloud A around +sep*e1, target cloud B around -sep*e1 (isotropic noise)."""
    g = torch.Generator().manual_seed(seed)
    mu_a = torch.zeros(d); mu_a[0] = sep
    mu_b = torch.zeros(d); mu_b[0] = -sep
    A = mu_a + sigma * torch.randn(n, d, generator=g)
    B = mu_b + sigma * torch.randn(n, d, generator=g)
    h0 = mu_a + 0.5 * torch.randn(d, generator=g)
    w = mu_a - mu_b
    return A, B, h0, w / w.norm()


def probe_disc(w, b=0.0):
    class _MB:                       # make_discriminant only reads mb for model kinds
        pass
    return make_discriminant(_MB(), "probe", HookSite(0, "block_input"), w=w, b=b)


def fd_grad(f, x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for j in range(x.size):
        e = np.zeros_like(x); e[j] = eps
        g[j] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


# --------------------------------------------------------------------------------------
# 1. probe discriminant
# --------------------------------------------------------------------------------------
def test_probe_discriminant_is_linear_with_grad_w(mb, site):
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(mb.d_model, generator=gen)
    disc = make_discriminant(mb, "probe", site, w=w, b=0.5)
    assert isinstance(disc, Discriminant) and disc.kind == "probe"
    h1 = torch.randn(mb.d_model, generator=gen)
    h2 = torch.randn(mb.d_model, generator=gen)
    assert torch.allclose(disc.grad(h1), w) and torch.allclose(disc.grad(h2), w)
    assert torch.isclose(disc.fn(h1), w @ h1 + 0.5)
    # linearity: g(a h1 + (1-a) h2) = a g(h1) + (1-a) g(h2)
    a = 0.3
    lhs = float(disc.fn(a * h1 + (1 - a) * h2))
    rhs = a * float(disc.fn(h1)) + (1 - a) * float(disc.fn(h2))
    assert abs(lhs - rhs) < 1e-4
    v, g = disc.evaluate(h1.double())           # any dtype / returns CPU float32
    assert g.dtype == torch.float32 and abs(v - float(disc.fn(h1))) < 1e-5
    with pytest.raises(ValueError):
        make_discriminant(mb, "nope", site, w=w)


# --------------------------------------------------------------------------------------
# 2. partial_forward reproduces the model logits
# --------------------------------------------------------------------------------------
def test_partial_forward_reproduces_logits(mb, sample_ids, site, h_unpert):
    h, logits_ref = h_unpert
    f = partial_forward(mb, sample_ids, site, sample_ids.shape[1] - 1)
    with torch.no_grad():
        lo = f(h)
    assert lo.shape == (mb.vocab,) and lo.dtype == torch.float32
    assert torch.allclose(lo, logits_ref, atol=1e-4)
    # the target_logit discriminant equals -log_softmax at the cached logits (+ offset b)
    tid = int(logits_ref.argmax())
    disc = make_discriminant(mb, "target_logit", site, input_ids=sample_ids,
                             position=sample_ids.shape[1] - 1, target_id=tid, b=0.25)
    expected = -float(torch.log_softmax(logits_ref, -1)[tid]) + 0.25
    assert abs(float(disc.fn(h)) - expected) < 1e-4
    assert disc.meta["target_id"] == tid and disc.meta["position"] == sample_ids.shape[1] - 1


@pytest.mark.parametrize("which, offset", [("last", 1), ("middle", 5)])
def test_partial_forward_block_path_matches_injector_path(mb, sample_ids, which, offset):
    """H1: the block-input fast path (captured input of block L, blocks L..n-1 only, LM head
    on the last position) gives the logits and the gradient of the full forward with the
    Injector replacement, for the last block at T-1 and a middle block at T-5."""
    from lsne import lambda_dynamics as LD
    from lsne.model import _get_hidden
    layer = mb.n_layers - 1 if which == "last" else mb.n_layers // 2
    site = HookSite(layer, "block_input")
    T = int(sample_ids.shape[1])
    pos = T - offset
    fast = partial_forward(mb, sample_ids, site, pos)
    slow = LD._partial_forward_injector(mb, sample_ids.to(mb.device), site, pos)
    assert (fast.site, fast.position, fast.n_tokens) == (site, pos, T)
    h0 = run_with_cache(mb, sample_ids, [site]).resid[site.key][0, pos].clone()
    h = h0 + 0.3 * h0.norm() * torch.randn(h0.shape, generator=torch.Generator().manual_seed(0)) / h0.numel() ** 0.5
    with torch.no_grad():
        assert torch.allclose(fast(h), slow(h), atol=1e-4) and not torch.allclose(fast(h), fast(h0), atol=1e-3)
    tid = int(slow(h).argmax())
    grads = []
    for f in (fast, slow):
        x = h.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(torch.log_softmax(f(x), -1)[tid], x)
        grads.append(g)
    assert torch.allclose(grads[0], grads[1], atol=1e-4) and grads[0].abs().max() > 0
    # only blocks L..n-1 run per call: count block calls
    calls = []
    hds = [blk.register_forward_pre_hook(lambda m, a, k, i=i: calls.append(i), with_kwargs=True)
           for i, blk in enumerate(mb.layers)]
    try:
        with torch.no_grad():
            fast(h)
    finally:
        for hd in hds:
            hd.remove()
    assert calls == list(range(layer, mb.n_layers))


# --------------------------------------------------------------------------------------
# 3. entropy discriminant gradient = -(entropy gradient)
# --------------------------------------------------------------------------------------
def test_entropy_discriminant_gradient(mb, sample_ids, site, h_unpert):
    h, _ = h_unpert
    pos = sample_ids.shape[1] - 1
    disc = make_discriminant(mb, "entropy", site, input_ids=sample_ids, position=pos)
    g_ref, H_ref = entropy_gradient(mb, sample_ids, site, pos)
    val, grad = disc.evaluate(h)
    # sign convention: g = -H, hence grad g = -grad H.  Both are float32 results of different
    # kernels (block-input fast path with the LM head on the last position vs. the full
    # forward); against a float64 reference they deviate by ~3e-5 and ~9e-5 (max abs, |g|<=1.8)
    assert torch.allclose(grad, -g_ref, rtol=1e-3, atol=2e-4)
    assert abs(val + H_ref) < 1e-4
    assert grad.dtype == torch.float32 and grad.device.type == "cpu"
    assert torch.allclose(disc.grad(h), grad)


# --------------------------------------------------------------------------------------
# 4. KDE: density maximal at a reference, gradient vs finite differences
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("metric", ["l2", "mahalanobis"])
@pytest.mark.parametrize("kernel", ["rbf", "laplacian"])
def test_kde_density_and_gradient(metric, kernel):
    g = torch.Generator().manual_seed(1)
    H = torch.randn(40, 8, generator=g) * torch.linspace(0.5, 2.0, 8)
    kw = {}
    if metric == "mahalanobis":
        U_r, lam_r = layer_covariance(H.numpy(), r=6)
        kw = dict(U_r=U_r, lam_r=lam_r)
    for k_nearest in (None, 10):
        kde = KDEMimicry(H, bandwidth="median", metric=metric, k_nearest=k_nearest, kernel=kernel, **kw)
        assert kde.bandwidth > 0 and np.isfinite(kde.median_distance)
        if kernel == "rbf":
            assert abs(kde.bandwidth - kde.median_distance ** 2) < 1e-9
        else:
            assert abs(kde.bandwidth - kde.median_distance) < 1e-9
        # density maximal at a reference point among random far-away queries
        ref = H[3]
        d_ref = kde.density(ref)
        assert 0.0 < d_ref <= 1.0
        queries = 3.0 * torch.randn(5, 8, generator=g) + 4.0
        assert all(kde.density(q) < d_ref for q in queries)
        # closed-form gradient vs central finite differences (float64 densities)
        for q in list(queries[:2]) + [ref + 0.3 * torch.randn(8, generator=g)]:
            x = q.double().numpy()
            g_cf = kde.grad(x).double().numpy()
            g_fd = fd_grad(kde.density, x, eps=1e-5)
            assert g_cf.shape == (8,)
            assert np.linalg.norm(g_cf - g_fd) <= 1e-3 * max(np.linalg.norm(g_fd), 1e-12)
            dens, g2 = kde.density_and_grad(x)
            assert abs(dens - kde.density(x)) < 1e-12 and torch.allclose(g2, kde.grad(x))


def test_kde_mahalanobis_consistency_and_validation():
    g = torch.Generator().manual_seed(2)
    H = torch.randn(30, 6, generator=g) * torch.tensor([3.0, 1.0, 1.0, 0.5, 0.5, 0.2])
    U_r, lam_r = layer_covariance(H.numpy(), r=6)
    kde = KDEMimicry(H, bandwidth=2.0, metric="mahalanobis", U_r=U_r, lam_r=lam_r, k_nearest=None)
    a, b = torch.randn(6, generator=g), torch.randn(6, generator=g)
    from lsne.geometry import mahalanobis
    assert abs(kde.metric_distance(a, b) - mahalanobis(a, b, U_r, lam_r)) < 1e-9
    with pytest.raises(ValueError):
        KDEMimicry(H, metric="mahalanobis")                   # needs U_r / lam_r
    with pytest.raises(ValueError):
        KDEMimicry(H, bandwidth=-1.0)
    with pytest.raises(ValueError):
        KDEMimicry(H, kernel="cosine")
    assert lambda_scale(2.0, 50, 4.0) == 400.0


# --------------------------------------------------------------------------------------
# 5. lam = 0 on the probe: monotone decrease, ball respected, crossing found
# --------------------------------------------------------------------------------------
def test_evasion_lambda0_probe(mb, h_unpert):
    h0, _ = h_unpert
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(mb.d_model, generator=gen)
    w = w / w.norm()
    b = -float(w @ h0) + 1.0                                   # g(h0) = 1 > 0
    disc = make_discriminant(mb, "probe", HookSite(mb.n_layers - 1, "block_input"), w=w, b=b)
    traj = evasion_trajectory(h0, disc, None, lam=0.0, step=0.1, d_max=2.0, n_steps=100)
    assert isinstance(traj, Trajectory) and traj.lam == 0.0
    assert abs(traj.g[0] - 1.0) < 1e-5
    assert np.all(np.diff(traj.g) <= 1e-6)                     # monotone decrease
    assert np.all(traj.dist_from_start <= 2.0 + 1e-6)
    recomputed = (traj.points.double() - h0.double()[None, :]).norm(dim=1).numpy()
    assert np.all(recomputed <= 2.0 * (1 + 1e-5) + 1e-5)
    assert traj.steps_to_cross == 11                           # g = 1 - 0.1 m < 0 first at m = 11
    assert traj.converged and traj.reason == "no_decrease"
    assert traj.points.shape[1] == mb.d_model and traj.points.dtype == torch.float32
    assert len(traj.g) == len(traj.density) == len(traj.F) == traj.points.shape[0]
    assert np.allclose(traj.F, traj.g)                         # no density term
    # too small a ball: never crosses, every point inside
    traj2 = evasion_trajectory(h0, disc, None, lam=0.0, step=0.1, d_max=0.5, n_steps=50)
    assert traj2.steps_to_cross is None and np.all(traj2.g > 0)
    assert np.all(traj2.dist_from_start <= 0.5 + 1e-6)
    # zero gradient -> "flat"
    flat = probe_disc(torch.zeros(mb.d_model), b=1.0)
    traj3 = evasion_trajectory(h0, flat, None, lam=0.0, step=0.1, d_max=1.0)
    assert traj3.reason == "flat" and not traj3.converged and traj3.n_steps == 0
    with pytest.raises(ValueError):
        evasion_trajectory(h0, disc, None, lam=1.0, step=0.1, d_max=1.0)


# --------------------------------------------------------------------------------------
# 6. synthetic two-cluster problem: large lam ends closer to the target cloud
# --------------------------------------------------------------------------------------
def test_mimicry_pulls_towards_target_cloud():
    A, B, h0, w = two_clusters()
    disc = probe_disc(w, b=0.0)                                # g > 0 on A, < 0 on B
    assert float(disc.fn(h0)) > 0
    kde = KDEMimicry(B, bandwidth="median", metric="l2", k_nearest=50)
    g_range = float(disc.fn(h0)) - float((B @ w).min())
    lam_big = lambda_scale(g_range, kde.n_eff, kde.bandwidth)
    kw = dict(step=0.25, d_max=10.0, n_steps=300)
    t0 = evasion_trajectory(h0, disc, kde, lam=0.0, **kw)
    t1 = evasion_trajectory(h0, disc, kde, lam=lam_big, **kw)
    m0 = manifold_stats(t0.endpoint[None], B, k=5)
    m1 = manifold_stats(t1.endpoint[None], B, k=5)
    geo0, geo1 = trajectory_geometry(t0), trajectory_geometry(t1)
    print("\nlam=0  :", {k: round(v, 3) for k, v in geo0.items() if isinstance(v, float)},
          "knn", float(m0.knn_mean_dist[0]))
    print("lam>0  :", {k: round(v, 3) for k, v in geo1.items() if isinstance(v, float)},
          "knn", float(m1.knn_mean_dist[0]))
    assert float(m1.knn_mean_dist[0]) < float(m0.knn_mean_dist[0])
    assert t1.density[-1] > t0.density[-1]
    assert t1.steps_to_cross is not None and t0.steps_to_cross is not None
    assert t1.converged and t1.reason == "no_decrease"
    assert geo1["tortuosity"] >= geo0["tortuosity"] - 1e-9 and abs(geo0["tortuosity"] - 1.0) < 1e-6
    assert np.all(t1.dist_from_start <= 10.0 + 1e-6)
    # Mahalanobis metric variant runs and also ends in the cloud
    U_r, lam_r = layer_covariance(B.numpy(), r=16)
    kde_m = KDEMimicry(B, bandwidth="median", metric="mahalanobis", U_r=U_r, lam_r=lam_r, k_nearest=50)
    lam_m = lambda_scale(g_range, kde_m.n_eff, kde_m.bandwidth)
    tm = evasion_trajectory(h0, disc, kde_m, lam=lam_m, step=0.25, d_max=10.0, n_steps=300, metric="mahalanobis")
    mm = manifold_stats(tm.endpoint[None], B, U_r=U_r, lam_r=lam_r, k=5)
    assert np.all(tm.dist_from_start <= 10.0 + 1e-6)
    assert float(mm.knn_mean_dist[0]) < float(m0.knn_mean_dist[0])
    assert mm.metric[0] == "mahalanobis" and np.isfinite(mm.knn_mean_mahal[0])


# --------------------------------------------------------------------------------------
# 7. lambda_sweep + trajectory_geometry
# --------------------------------------------------------------------------------------
def test_lambda_sweep_and_geometry():
    A, B, h0, w = two_clusters(seed=4)
    disc = probe_disc(w)
    kde = KDEMimicry(B, bandwidth="median", k_nearest=50)
    base = lambda_scale(6.0, kde.n_eff, kde.bandwidth)
    lambdas = [0.0, 0.1 * base, base, 10 * base]
    sweep = lambda_sweep(h0, disc, kde, lambdas, step=0.25, d_max=8.0, n_steps=200)
    assert list(sweep.keys()) == lambdas
    for lam, t in sweep.items():
        assert isinstance(t, Trajectory) and t.lam == lam
        geo = trajectory_geometry(t)
        for key in ("path_length", "net_displacement", "tortuosity", "mean_turning_angle_deg",
                    "final_g", "final_density", "final_F", "dist_from_start"):
            assert np.isfinite(geo[key]), (lam, key, geo[key])
        assert geo["n_steps"] == t.n_steps and geo["steps_to_cross"] == t.steps_to_cross
        assert geo["tortuosity"] >= 1.0 - 1e-9
    geo_big = trajectory_geometry(sweep[10 * base])
    assert geo_big["mean_turning_angle_deg"] > 0.0 and geo_big["tortuosity"] > 1.0
    # degenerate trajectory: a single point
    single = Trajectory(points=h0[None].clone(), g=np.array([1.0]), density=np.array([0.0]), F=np.array([1.0]),
                        dist_from_start=np.array([0.0]), lam=0.0, converged=False, steps_to_cross=None)
    geo1 = trajectory_geometry(single)
    assert geo1["path_length"] == 0.0 and np.isnan(geo1["tortuosity"]) and np.isnan(geo1["mean_turning_angle_deg"])


# --------------------------------------------------------------------------------------
# 8. feature_lambda_correlation on a synthetic TopK SAE
# --------------------------------------------------------------------------------------
def test_feature_lambda_correlation_synthetic_sae():
    d, F, k = 16, 24, 4
    g = torch.Generator().manual_seed(5)
    W_enc = torch.randn(F, d, generator=g) / d ** 0.5
    b_enc = -0.05 * torch.rand(F, generator=g)
    b_enc[7] = 0.0
    W_dec = torch.randn(F, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    b_dec = 0.1 * torch.randn(d, generator=g)
    sae = TopKSAE.from_tensors(W_enc, b_enc, W_dec, b_dec, k)
    lams = [0.0, 0.5, 1.0, 2.0, 4.0]
    endpoints = {lam: b_dec + lam * W_enc[7] for lam in lams}   # pre_7 = lam * ||W_enc[7]||^2
    df = feature_lambda_correlation(endpoints, sae)
    assert set(df.columns) >= {"feature", "rho", "pvalue", "act_at_min_lambda", "act_at_max_lambda",
                               "active_only_lambda0", "active_only_lambda_pos"}
    row = df[df.feature == 7].iloc[0]
    assert abs(row.rho - 1.0) < 1e-9
    assert row.act_at_min_lambda == 0.0 and row.act_at_max_lambda > 0.0
    assert bool(row.active_only_lambda_pos) and not bool(row.active_only_lambda0)
    assert np.isfinite(df.rho).any()
    assert (df.max_act > 0).all()
    # manifold_stats with the SAE: finite fvu / shared-feature counts
    ms = manifold_stats(torch.stack(list(endpoints.values())), b_dec + 0.3 * torch.randn(50, d, generator=g),
                        sae=sae, k=5)
    assert len(ms) == len(lams) and np.isfinite(ms.sae_fvu).all() and np.isfinite(ms.local_log_volume).all()
    assert ((ms.n_shared_features_with_ref >= 0) & (ms.n_shared_features_with_ref <= k)).all()
    # TopK-restricted code also works
    df2 = feature_lambda_correlation(endpoints, sae, topk_only=True)
    assert 7 in set(df2.feature)


# --------------------------------------------------------------------------------------
# 9. endpoint_effects and geometry_lambda_correlation
# --------------------------------------------------------------------------------------
def test_endpoint_effects_unperturbed_B_is_zero(mb, sample_ids, site, h_unpert):
    h, logits_ref = h_unpert
    tid = int(logits_ref.argmax())
    gen = torch.Generator().manual_seed(6)
    endpoints = {0.0: h, 1.0: h + 0.5 * h.norm() / mb.d_model ** 0.5 * torch.randn(mb.d_model, generator=gen)}
    df = endpoint_effects(mb, sample_ids, site, sample_ids.shape[1] - 1, endpoints, tid,
                          freq_ids=torch.tensor([13, 15, 253]))
    assert list(df.lam) == [0.0, 1.0]
    assert abs(float(df.B[0])) < 1e-5
    assert abs(float(df.logp_target[0]) - float(torch.log_softmax(logits_ref, -1)[tid])) < 1e-4
    assert np.isfinite(df[["logp_target", "entropy", "p_freq", "B"]].to_numpy()).all()
    assert 0 < float(df.p_freq[0]) < 1
    assert abs(df.attrs["logp_target_unperturbed"] - float(df.logp_target[0])) < 1e-5
    assert float(df.B[1]) != 0.0


def test_geometry_lambda_correlation_rows():
    rows = pd.DataFrame(dict(lam=[0.0, 1.0, 2.0, 3.0], path_length=[1.0, 2.0, 3.0, 4.0],
                             tortuosity=[1.0, 1.0, 1.0, 1.0], final_g=[4.0, 3.0, 1.0, 0.5],
                             converged=[True, True, False, True], reason=["a", "b", "c", "d"],
                             steps_to_cross=[3, 5, None, 9]))
    out = geometry_lambda_correlation(rows)
    numeric = [c for c in rows.columns if c != "lam" and pd.api.types.is_numeric_dtype(rows[c])]
    assert list(out.quantity) == numeric and len(out) == len(numeric)
    by = out.set_index("quantity")
    assert abs(by.loc["path_length", "rho"] - 1.0) < 1e-12
    assert abs(by.loc["final_g", "rho"] + 1.0) < 1e-12
    assert np.isnan(by.loc["tortuosity", "rho"]) and by.loc["tortuosity", "n"] == 4
    assert by.loc["steps_to_cross", "n"] == 3
    with pytest.raises(ValueError):
        geometry_lambda_correlation(rows.drop(columns=["lam"]))
