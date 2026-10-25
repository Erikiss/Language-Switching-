# LS-NEQ module specification (developer contract)

All modules live in `lsne/`. Python ≥ 3.10, torch ≥ 2.2, transformers ≥ 4.45 (verified
with 5.20). Everything must run on CPU with `EleutherAI/pythia-14m` (6 layers, d=128,
4 heads) for tests, and on GPU with pythia-160m/410m/1b/1.4b. **Never** hard-code
d_model, n_layers or vocab size; read them from the model config.

Verified facts (transformers 5.20, GPT-NeoX):
- `model.gpt_neox.layers[i]` is `GPTNeoXLayer`; children `input_layernorm`,
  `post_attention_layernorm`, `attention`, `mlp`. `model.gpt_neox.final_layer_norm`,
  `model.lm_head` (not `embed_out`), `model.gpt_neox.embed_in`.
- `output_hidden_states=True`: `hidden_states[i]` == input of block i (i < L);
  `hidden_states[L]` == final_layer_norm(output of block L-1) (NOT raw residual).
- Block output: forward hook on the block returns a plain Tensor `[B,T,d]` (or a tuple
  whose first element is the Tensor — handle both).
- Forward pre-hook with `with_kwargs=True`: `hidden_states` may arrive in `args[0]` or
  `kwargs['hidden_states']` — handle both (see `Injector`).
- `output_attentions=True` requires `attn_implementation="eager"`; attentions are a list
  of `[B,H,T,T]`.
- During `generate()` with KV cache the block pre-hook sees `[B,1,d]` for each new token.
- `logits == lm_head(final_layer_norm(block_output[L-1]))`.
- Entropy gradient wrt a block input is non-zero at *all* earlier positions (attention
  reads them); the State-2143 direction is the gradient component at the *last* position only.

## Shared types — `lsne/config.py` (written; do not change signatures)

```python
STATE2143: dict              # anchor constants incl. hashes, hook layer 11, position 207, eps 0.01
@dataclass ModelSpec(name:str, revision:str="main", dtype:str="float32", device:str="auto", attn_implementation:str="eager")
@dataclass HookSite(layer:int, site:str)   # site in {"block_input","block_output","mlp_output","attn_output"}
    .key -> str   e.g. "L11.block_input"
    .sae_hookpoint -> str|None  # block_output of N -> "layers.N"; block_input of N -> "layers.N-1" (None for N=0); mlp_output of N -> "layers.N.mlp"
DOSES, DOSES_STEER, K_LEVELS, C_LEVELS, MASKS_UNIT_LEVEL, LAYER_GRID  # constants
```

## `lsne/anchor.py` (written) and `lsne/pile_fetch.py` (written)

`fetch_sequence(seq_index, cache_dir) -> np.ndarray[uint16, 2049]` by HTTP range request;
`verify_state2143(tokens) -> dict`; `load_anchor(cache_dir) -> AnchorData(tokens, context_ids[208], target_id, text, ...)`;
`sample_unigram_counts(n_sequences, seed, cache_dir) -> np.ndarray[vocab]`.

## `lsne/model.py`

```python
class ModelBundle:
    model, tokenizer, spec: ModelSpec, device: torch.device
    n_layers:int, d_model:int, n_heads:int, vocab:int
    layers -> list[nn.Module]; final_ln; lm_head
def load_model(spec: ModelSpec) -> ModelBundle        # eval mode, no grad by default, attn eager
def encode(mb, text:str) -> torch.LongTensor[1,T]     # no special tokens added
@dataclass CacheOut: logits [B,T,V] (float32); resid: dict[str, Tensor[B,T,d]] keyed by HookSite.key; attn: list[Tensor[B,H,T,T]]|None
def run_with_cache(mb, input_ids, sites: list[HookSite]|str="all_block_inputs", need_attn=False, grad=False) -> CacheOut
    # "all_block_inputs" = block_input for 0..L-1 plus block_output for L-1 (so L+1 residual snapshots, keys "L0.block_input"... "L{L-1}.block_output")
class Injector:   # context manager
    Injector(mb, site: HookSite, delta: Tensor[d]|Tensor[B,T,d]|Callable[[Tensor],Tensor], positions: list[int]|slice|None=None, mode="add"|"replace", every_token:bool=False)
    # positions=None & every_token=True -> add to every position seen (used for steering during generation)
    # mode="replace": delta is Tensor[B,len(positions),d] replacing the residual at positions
    # .audit -> dict(intended_delta_norm, written_delta_norm, rel_err)  computed on the first call (Soll/Ist)
def make_dose_delta(h_ref: Tensor[d], direction: Tensor[d], epsilon: float) -> Tensor[d]   # eps*RMS(h_ref)*dir/RMS(dir)
def logit_lens(mb, resid: Tensor[...,d]) -> Tensor[...,V]      # final_ln + lm_head
def next_token_stats(logits_last: Tensor[B,V], target_id:int, freq_ids: Tensor|None=None) -> dict[str, Tensor[B]]
    # keys: logp_target, p_target, entropy, top1_id, p_freq (mass on freq_ids)
def entropy_gradient(mb, input_ids:[1,T], site: HookSite, position:int) -> tuple[Tensor[d] grad_at_position, float entropy]
def generate_steered(mb, input_ids, site, direction, eps_schedule: Callable[[int], float], n_new:int, n_samples:int, temperature:float, seed:int, top_p: float=1.0) -> dict(tokens [n_samples,n_new], logprobs [n_samples,n_new], entropies [n_samples,n_new], eps [n_new])
    # per generated step k: delta_k = eps_schedule(k) * RMS(h_ref) * dir/RMS(dir) where h_ref is the current residual at the site.
    # Semantics (verified): the dose eps(k) is applied at the forward position that PRODUCES generated token k
    # (absolute position T0-1+k; the last prompt position produces token 0). Prompt positions < T0-1 are untouched.
    # result["eps"] has exactly n_new entries.
```

## `lsne/stimuli.py` + `data/stimuli/families/*.json`

Family JSON schema (one file per family and formulation, e.g. `F01A.json`):
```json
{"family_id":"F01","formulation":"A","title":"...", "topic":"...",
 "units":[{"id":1,"E":{"subj":"Kraft","pred":"reads the report","adj":"in the morning"},
                  "D":{"subj":"Kraft","pred":"liest den Bericht","adj":"am Morgen"}}, ... 10 units],
 "tail_E":"The report is about the soldier", "target":" Fewkoombey" , "notes":"..."}
```
Rendering rules: segment strings carry no leading/trailing spaces; a unit renders as
`subj + " " + pred + " " + adj + "."`; units are joined with a single space; the full
context is `units_text + " " + tail_E` (tail has no trailing punctuation unless the
target follows directly). `target` is a string beginning with a space; it must
tokenize to exactly one token id when appended to the context (verified).

```python
@dataclass Variant: family_id, formulation, K:int, C:str, mask: list[str] (per segment 'E'/'D'), text:str, ids: list[int], target_id:int,
                    annot: dict  # lists per token: lang, unit, klass, dist_to_last_switch; ints: last_switch_pos, tail_start, n_words, n_tokens, F100
def load_families(dir) -> list[Family]
def unit_level_masks(K) -> list[str]        # from config.MASKS_UNIT_LEVEL
def segment_masks(n_units, K, C) -> list[str]   # length 3*n_units, implements C1/C2/C3 (see DESIGN §2.1) and C="seg" for K in {16, 28}
def render(family, mask) -> str
def build_variants(family, tokenizer, K_levels, C_levels, include_seg=True) -> list[Variant]
def check_variant(v, ref_variant) -> list[str]   # problems: target id differs, tail ids differ, target leak, ...
def annotate(tokenizer, text, segments_with_lang) -> annot    # token classes: ws, punct, word_start, word_cont, boundary
def original_2143_family(anchor: AnchorData, tokenizer) -> Family   # from data/stimuli/original_2143.json (token boundaries + German translations)
```

## `lsne/lang_id.py`

Deterministic, offline word-level E/D/N classifier: function-word lexicons
(`data/lexicon/function_words.json`), character cues (ä ö ü ß, 'sch', suffixes),
capitalised mid-sentence nouns. `label_tokens(tokenizer, ids) -> list[str]` ('E','D','N'),
`language_fraction(labels, lang='D') -> float`, `switch_positions(labels) -> list[int]`,
`switch_count`, `inter_switch_intervals`. Must score ≥ 0.95 accuracy on the stimulus
bank segments (test).

## `lsne/directions.py`

```python
@dataclass Direction: name:str, vec: Tensor[d] (unit L2), site: HookSite, meta: dict
def entropy_direction(mb, input_ids, site, position, remove_axis: Tensor|None=None) -> Direction   # name "ent"
def diff_of_means(acts_D: Tensor[N,d], acts_E: Tensor[N,d], denoise_pcs:int=1) -> Tensor[d]
def language_axis(mb, variants, site, pool="segment_last") -> Direction          # "lang_dom"
def probe_axis(acts, labels, C=1.0) -> tuple[Tensor[d], float acc]               # "lang_probe"
def random_directions(d, n, seed) -> Tensor[n,d]
def frequency_axis(mb, unigram_counts, top_n=100) -> Direction                   # "freq"  (mean lm_head row of top-n minus mean of all; expressed in residual space via lm_head.weight; note final LN ignored)
def cosine_table(dirs: list[Direction]) -> pandas.DataFrame
def unembed_projection(mb, direction, top_k=25) -> dict(promoted: list[(token, score)], suppressed: ...)
def collect_activations(mb, variants, site, which="segment_last") -> (acts Tensor[N,d], labels list[str], meta DataFrame)
```

## `lsne/sae.py`

Minimal TopK SAE in EleutherAI `sparsify` format (safetensors keys `encoder.weight [F,d]`,
`encoder.bias [F]`, `W_dec [F,d]`, `b_dec [d]`; cfg.json keys `k, num_latents, d_in, normalize_decoder`).
```python
class TopKSAE(nn.Module):
    @classmethod load_from_hub(repo:str, hookpoint:str, device, cache_dir=None) -> TopKSAE   # hf_hub_download cfg.json + sae.safetensors
    @classmethod from_tensors(W_enc, b_enc, W_dec, b_dec, k)                                  # for tests
    encode(x[...,d]) -> (top_acts[...,k], top_idx[...,k], pre_acts[...,F])   # pre = (x - b_dec) @ W_enc.T + b_enc ; topk on pre (post-ReLU)
    decode(top_acts, top_idx) -> x_hat[...,d]
    forward(x) -> (x_hat, fvu: float)
def sae_fvu(sae, acts) -> float
def feature_table(sae, acts, labels) -> DataFrame(feature, rate_E, rate_D, mean_E, mean_D, selectivity=log((rate_D+1e-6)/(rate_E+1e-6)))
def language_selective_features(table, n=10) -> dict(D: list[int], E: list[int])
def feature_direction(sae, idx) -> Tensor[d] (unit)
def sae_repo_for(model_name) -> tuple[repo, hookpoint_fmt]|None   # pythia-160m -> ("EleutherAI/sae-pythia-160m-32k","layers.{n}"), pythia-70m -> 32k, pythia-410m -> ("EleutherAI/sae-pythia-410m-65k","layers.{n}.mlp") (MLP only!)
```

## `lsne/patching.py`

```python
def position_classes(variant) -> dict[str, list[int]]   # 'ws','punct','boundary','word_start','word_cont','tail','all'
def align_tail(v_src, v_dst) -> list[tuple[int,int]]     # (src_pos, dst_pos) pairs for the shared tail, aligned from the end
def patch_run(mb, src_ids, dst_ids, site, pairs, target_id) -> dict(logp_patched, logp_src, logp_dst)
def counterfactual_attribution(mb, v_mono, v_mixed, sites: list[HookSite], classes: list[str], n_random=10, seed=0) -> DataFrame
    # recovery = (logp_patched - logp_mixed) / (logp_mono - logp_mixed); random control = same count of random tail positions
def grad_x_act(mb, ids, site, target_id) -> Tensor[T]
```

## `lsne/dynamics.py`

```python
@dataclass DamageMap: D: ndarray[L+1, T] (relative L2), cos: ndarray[L+1,T], layers: list[str], inject_layer:int, inject_pos:int, eps:float
def damage_spreading(mb, input_ids, site, position, direction: Tensor[d], eps, ref_cache: CacheOut|None=None) -> DamageMap
def growth_exponent(dm: DamageMap, positions: list[int], layer_from:int|None=None) -> dict(lam, intercept, r2, per_position: ndarray)   # OLS of ln D over layer index, layers > inject_layer; D clipped at 1e-12
def spreading_front(dm, threshold) -> ndarray[L+1]   # per layer: max distance (in positions) from inject_pos where D > threshold
def logit_lens_trajectory(mb, cache: CacheOut, positions=None) -> dict(kl [L+1,T], target_rank [L+1,T], target_logp [L+1,T])
def nonmonotonicity_index(kl_series: ndarray) -> float   # sum of positive increments / total variation (0 = monotone)
def update_norm_profile(cache) -> ndarray[L, T]
def ca_symbolize(mb, cache, lang_axis: Direction|None, lexicon_ids: dict, thresholds=(0.5,)) -> ndarray[L+1,T] int8  (0=N,1=E,2=D) using logit-lens mass on E vs D marker tokens
def ca_complexity(S) -> dict(spatial_entropy, temporal_entropy, activity, lz76, block_entropy_2x2)
def lz76(seq: Sequence[int]) -> int
```

## `lsne/generation.py`

```python
def schedule_ramp(eps_max, n_steps, tokens_per_step) -> Callable[[int], float]   # 0 -> eps_max -> 0, plateau per step
def schedule_quench(eps, t_on) -> Callable; schedule_const(eps)
def language_trajectory(tokenizer, tokens[B,n]) -> dict(labels [B,n], fD_window [B,n] (window 8), switches per sample)
def hysteresis(mb, input_ids, site, direction, eps_max, n_steps=8, tokens_per_step=8, n_samples=8, seed=0) -> dict(up: ndarray[n_steps], down: ndarray[n_steps], area: float, null_area: float (independent restarts at each eps), curves DataFrame)
def relaxation(mb, input_ids, site, direction, eps, t_on=16, n_new=64, n_samples=8) -> dict(fD_t, entropy_t, tau, tau_ci)
def switch_interval_stats(labels) -> dict(intervals, geometric_p, ks_stat, tail_index)
def susceptibility(fD_by_eps: dict[float, ndarray]) -> DataFrame(eps, mean, var, chi)
def fdt_ratio(m0_samples: ndarray, chi: float) -> float
```

## `lsne/tn_lattice.py`

```python
@dataclass LatticeTN: tensors: list[(inds, array)], output_inds, meta
def pca_basis(acts Tensor[N,d], r) -> Tensor[d,r]
def linear_response_jvp(mb, input_ids, site_from, position, direction, layers_to: list[int], window: slice) -> ndarray[len(layers_to), T_w, d]   # exact first-order via torch.func.jvp through the model (no TN)
def build_factorized_lattice(mb, cache, site_from, window: slice, basis: Tensor[d,r], layers: range) -> LatticeTN
    # per block l in layers: branch index b_l of size 2+H: skip (A=I_T, M=I_r), heads h (A=attn pattern [T_w,T_w], M = P^T W_O^h W_V^h gamma1/sigma1_mean P [r,r]),
    # MLP (A = delta_{t,t'} hyper-index, M_t = P^T W_out diag(gelu'(z_t)) W_in gamma2/sigma2_t P), plus ones[b_l]
def optimize_path(tn: LatticeTN, backend="tnco"|"opt_einsum", **kw) -> dict(path, cost_log10, runtime_s, backend)
def contract(tn, path) -> ndarray   # via opt_einsum.contract with explicit path
def compare_responses(measured: DamageMap, linear: ndarray, factorized: ndarray) -> DataFrame
```
TNCO usage (verified, tnco 0.4.0): `from tnco.app import Optimizer, Tensor, TensorNetwork`;
`tn = TensorNetwork([Tensor(inds=(...), dims=(...)), ...], output_inds=(...))`;
`new_tn, res = Optimizer().optimize(tn, betas=(0,1e4), n_steps=..., n_runs=...)`;
`res` is a list; `sorted(res, key=lambda r: r.cost)[0].path` is a list of index pairs
usable as `opt_einsum.contract(expr, *arrays, optimize=path)`; `r.cost.log10()`.
`new_tn.output_inds` is a frozenset — keep your own ordered output index tuple.
If `tnco` is not importable, fall back to `opt_einsum.contract_path(..., optimize='dp' if n<=12 else 'greedy')`.

## `lsne/geometry.py`  (Meaning-Contracts measures, DESIGN §4.4)

Implements the local-geometry measures of Godoy Ortiz 2026 (arXiv:2610.30715) on
residual snapshots. All inputs are torch or numpy arrays; outputs numpy / DataFrames.
```python
def pool_states(resid: Tensor[B,T,d] | ndarray, readout: str, positions: list[int]|None=None) -> ndarray[B,d]
    # readout "mean": mean over the given positions (default: all); "last": state at the last position
def deflate(X: ndarray[N,d], k: int, basis: ndarray[k,d]|None=None) -> tuple[ndarray[N,d], ndarray[k,d]]
    # center X, remove projection on top-k right singular vectors (fit on X if basis None); k=0 -> centered X
def layer_covariance(X: ndarray[N,d], r: int) -> tuple[ndarray[d,r] U_r, ndarray[r] lam_r]   # eigendecomposition of the covariance, truncated to rank r = min(r, N-1, d)
def mahalanobis(x: ndarray[d], y: ndarray[d], U_r, lam_r) -> float                          # eq. (1) of the paper
def log_volume(G: ndarray[n,d], eps: float=1e-8) -> float   # sum_i log max(lambda_i, eps) over ALL n eigenvalues of the centered n x d cloud (economy SVD; one is ~0)
def participation_ratio(G: ndarray[n,d]) -> float          # (sum lam)^2 / sum lam^2
def displacement(mean_a: ndarray[d], mean_b: ndarray[d], U_r=None, lam_r=None) -> dict(euclid, mahal)
def group_geometry(groups: dict[str, ndarray[n_g,d]], r: int=128, deflate_k: tuple=(0,5), ref_group: str|None=None) -> DataFrame
    # rows: group x k with n, log_volume, participation_ratio, displacement_euclid/mahal from ref_group mean (covariance from the pooled X of all groups)
def pair_ordering(X: ndarray[N,d], labels: DataFrame(family, formulation, lang), r=128, k=5) -> DataFrame
    # paper's categories on our bank: exact_translation (same family+formulation, E vs D), same_meaning_same_lang (A vs B, same lang), same_meaning_cross_lang (A_E vs B_D), other_family; mean Mahalanobis + cosine per category -> G0 gate
def per_position_log_volume(resid_group: ndarray[n,T_aligned,d], eps=1e-8) -> ndarray[T_aligned]   # cloud per aligned position
def enumerate_unit_masks(K: int, n_units: int=10) -> list[str]
    # all masks with unit1=E, unit10=E, unit9=D (if K>0), exactly K switches; K=0 -> ["E"*n_units]
def realization_groups(families, build_fn, K_levels, C_levels) -> dict[(family_id,K,C) -> list[Variant]]
    # build_fn(family, segment_mask) -> Variant; segment masks from lsne.stimuli.segment_masks_from_unit_mask if available, else a local re-implementation of DESIGN §2.1 (C1/C2/C3) — keep both consistent (test)
def geometry_table(mb, groups: dict, sites: list[HookSite], readouts=("mean","last"), r_mean=128, r_last=256, deflate_k=(0,5)) -> DataFrame
    # runs run_with_cache per variant once, pools states per site/readout, calls group_geometry with ref_group = the K=0 group of the same family
```
Tests: synthetic Gaussian clouds (log_volume decreases when the cloud shrinks; PR of an
isotropic cloud ≈ its dimension; Mahalanobis of x vs x is 0 and symmetric; deflate removes
the top direction exactly); enumerate_unit_masks(2) has 8 masks, all with exactly 2
switches and the required fixed units; pythia-14m: pair_ordering on 2 families × 2
formulations × E/D returns all four categories with finite values.

## `lsne/trajectory.py`  (DESIGN §4.5)

```python
def depth_trajectory(geom_table: DataFrame) -> DataFrame   # per (family,K,C,readout,k): log_volume(l), pr(l), displacement(l), delta_l log_volume, breathing_index = number of sign changes of delta_l, breathing_amplitude = mean |delta_l|, layer_of_max_contraction
def position_trajectory(mb, group_variants: list, site: HookSite, align: str="tail_from_end", n_prefix_bins: int=8) -> DataFrame   # per-position log-volume along the aligned tail + relative prefix bins; also position relative to last switch
def steering_tracks_switching(mb, variants_K0: list, variants_K: list, direction, site, eps_list, n_random=32, seed=0) -> DataFrame
    # per layer snapshot: v_K = mean(K) - mean(K0) (pooled 'last' readout); v_steer(eps) = mean over K0 variants of (h_steered - h) at the same snapshot after injecting eps*dir at the site; cosine(v_steer, v_K), Mahalanobis displacement of steered states relative to the K0 cloud; same for random directions (mean, sd)
def couple_with_dynamics(depth_traj: DataFrame, damage_lambda_by_layer: DataFrame, lens_kl_by_layer: DataFrame) -> DataFrame   # per-layer correlations (Spearman) between delta_l log_volume and lambda_l / KL_l, per condition
```
Tests: synthetic tables for depth_trajectory (breathing index of a monotone series is 0;
of an alternating series equals n-2 sign changes); steering_tracks_switching on pythia-14m
with two tiny hand-made groups returns cosines in [-1,1] with random controls.

## `lsne/lambda_dynamics.py`  (Biggio et al. Alg. 1 in activation space, DESIGN §4.6)

```python
@dataclass Discriminant: name:str, kind:str, fn: Callable[[Tensor[d]], Tensor[()]], grad: Callable[[Tensor[d]], Tensor[d]], meta: dict
def partial_forward(mb, input_ids:[1,T], site: HookSite, position:int) -> Callable[[Tensor[d]], Tensor[V]]
    # returns f(h) = last-position logits when the residual at (site, position) is REPLACED by h (Injector replace with a leaf tensor); differentiable wrt h
def make_discriminant(mb, kind:str, site, input_ids=None, position=None, w=None, b=0.0, target_id=None, sae=None, feature=None) -> Discriminant
    # kind "probe": g(h) = w·h + b (w from directions.probe_axis; positive = source class); "target_logit": g(h) = -logp(target | h) ; "entropy": g(h) = -H(next-token) ; "sae_feature": g(h) = -relu(pre-activation of feature) 
class KDEMimicry:
    def __init__(self, H_target: Tensor[n,d], bandwidth: float|str="median", metric: str="l2", U_r=None, lam_r=None, k_nearest: int|None=50)
    # metric "mahalanobis": whiten with z = lam_r^-1/2 U_r^T h (r-dim) and run the KDE in z; "l2": raw space. bandwidth "median" = median pairwise distance of H_target (in the metric). k_nearest: use only the k nearest reference points (paper used 50).
    def density(self, h) -> float            # (1/n) sum_i exp(-d(h,h_i)^2 / bw)   (RBF as in the paper; also "laplacian" via kernel="l1" option)
    def grad(self, h) -> Tensor[d]          # closed form (paper §3.2), mapped back to h-space for the Mahalanobis metric
def lambda_scale(g_values_range: float, n: int, bw: float) -> float    # lambda such that lambda/(n*bw) is comparable to the range of g (paper §3.2)
@dataclass Trajectory: points: Tensor[m+1,d], g: ndarray[m+1], density: ndarray[m+1], F: ndarray[m+1], dist_from_start: ndarray[m+1], lam: float, converged: bool, steps_to_cross: int|None
def evasion_trajectory(h0: Tensor[d], g: Discriminant, kde: KDEMimicry|None, lam: float, step: float, d_max: float, n_steps: int=200, eps_stop: float=1e-5, metric: str="l2") -> Trajectory
    # Alg. 1: direction = unit(∇g(h) - lam * ∇p̂(h)); h <- h - step*direction; if d(h,h0) > d_max project onto the ball boundary (l2); stop when F(h_m) - F(h_{m-1}) > -eps_stop (no decrease) or n_steps
def lambda_sweep(h0, g, kde, lambdas: Sequence[float], **kw) -> dict[float, Trajectory]
def trajectory_geometry(traj) -> dict(path_length, net_displacement, tortuosity, mean_turning_angle_deg, n_steps, steps_to_cross, final_g, final_density, final_F, dist_from_start)
def manifold_stats(points: Tensor[m,d], X_ref: Tensor[N,d], U_r=None, lam_r=None, sae=None, k: int=5) -> DataFrame per point: mahal_to_ref_mean, knn_mean_dist, local_log_volume (cloud of the k nearest ref points + the point; lsne.geometry.log_volume), sae_fvu, n_shared_features_with_ref (overlap of the point's top-k SAE features with the union of the k nearest refs' features)
def endpoint_effects(mb, input_ids, site, position, endpoints: dict[float, Tensor[d]], target_id, freq_ids=None) -> DataFrame(lam, logp_target, entropy, p_freq, B = logp_target - logp_target_unperturbed)
def feature_lambda_correlation(endpoints: dict[float, Tensor[d]], sae) -> DataFrame(feature, rho, pvalue, act_at_min_lambda, act_at_max_lambda, active_only_lambda0, active_only_lambda_pos)
def geometry_lambda_correlation(rows: DataFrame with column lam and numeric columns) -> DataFrame(quantity, rho, pvalue, n)
```
Tests (pythia-14m): probe discriminant gradient equals w; partial_forward reproduces the
model logits when h is the unperturbed residual; "entropy" discriminant gradient matches
lsne.model.entropy_gradient at the same site/position (allclose); KDE density is maximal at
a reference point and its gradient matches finite differences; evasion_trajectory with
lam=0 decreases g monotonically and respects d_max; with a large lam the endpoint is closer
(Mahalanobis / knn) to the target cloud than with lam=0 on a synthetic 2-cluster problem;
lambda_sweep returns one Trajectory per lambda; feature_lambda_correlation on a synthetic
SAE (TopKSAE.from_tensors) returns finite rho; endpoint_effects B=0 for the unperturbed state.

## `lsne/atlas.py`  (domain compass / susceptibility-atlas proxy, DESIGN §4.7)

Data: `data/lexicon/domains.json` = {"domains": {"law": {"E": [...], "D": [...]}, "politics": ..., "economy": ..., "culture_values": ..., "family": ..., "religion": ..., "wellbeing": ..., "science": ..., "everyday": ...}, "templates": {"E": ["The discussion was about {w}.", ...], "D": ["Die Diskussion handelte von {w}.", ...]}} with ≥ 40 words per domain and language.
```python
def load_domains(path=None) -> dict
def domain_token_ids(tokenizer, domains: dict, lang: str="E") -> dict[str, list[int]]   # single-token words with leading space only (report dropped words in meta)
def keyword_readout(mb, h: Tensor[d], direction: Tensor[d]|None=None, eps: float=0.05, top_k: int=50) -> dict(promoted: list[(token, delta_logit)], suppressed: ...)
    # logit lens of h + make_dose_delta(h, direction, eps) minus logit lens of h; direction None -> logit lens of h itself
def domain_profile(mb, h, direction=None, eps=0.05, domains_ids: dict=None, n_random: int=32, seed: int=0) -> DataFrame(domain, mass, mass_random_mean, mass_random_sd, z)
    # mass = sum of softmax probability mass (lens) on the domain's token ids after the step minus before; z against random directions of the same dose
def sae_feature_domain_profile(mb, sae, feature_idx: int, domains_ids) -> DataFrame   # unembedding projection of W_dec[feature] onto domain ids (mean logit per domain, z vs all tokens)
def domain_axis(mb, tokenizer, domain: str, site: HookSite, contrast: str|list[str]="everyday", lang: str="E", n_templates: int|None=None) -> Direction   # name "dom_<domain>": diff-of-means of last-token residuals on template sentences filled with domain words vs contrast domain words (lsne.directions.diff_of_means, denoise_pcs=1)
def counter_steer(h: Tensor[d], remove: Direction|Tensor, then: Direction|Tensor, eps_remove: float|None, eps_then: float) -> Tensor[d]
    # 1) project out the `remove` component of (h - mean) if eps_remove is None, else subtract eps_remove*RMS(h)*unit(remove); 2) add make_dose_delta(h, then, eps_then)
def load_atlas_extraction(path: str) -> list[dict(cluster_id, keywords: list[str], raw: str)]   # tolerant parser: blocks separated by blank lines; lines of the form "<id>: kw1, kw2, ..." or "- kw"; returns [] if the file does not exist
def atlas_domain_overlap(clusters, domains) -> DataFrame(cluster_id, domain, n_overlap, jaccard)
```

## `lsne/heads.py`  (per-head attribution and head-level steering, DESIGN §4.7)

```python
def head_slices(mb, layer: int) -> list[slice]                    # slices of the dense (W_O) input per head
def head_contributions(mb, input_ids, layer: int) -> Tensor[H, T, d]   # per-head direct contribution to the residual at this layer: x_h @ W_O[:, slice_h].T, captured with a forward pre-hook on layers[layer].attention.dense (verify against the attention output: sum over heads + bias == attention output)
def head_direct_logit_effect(mb, input_ids, layer: int, target_id: int, position: int=-1) -> Tensor[H]   # lens of each head's contribution at `position` projected on the target logit (logit lens through final LN linearised at the residual: use lsne.model.logit_lens on (h + c_h) - logit_lens(h))
def ablate_heads(mb, layer: int, heads: list[int], mode: str="mean", mean_source: Tensor|None=None) -> context manager   # zero or mean-ablate the chosen head slices at the dense input
def head_ablation_effects(mb, variants: list, layer_grid: list[int], target_id_attr: str="target_id", mode="mean") -> DataFrame(family_id, K, C, layer, head, logp_base, logp_ablated, delta)
def worst_head(effects: DataFrame, families_weak: list[str], families_stable: list[str]) -> dict(layer, head, score, table)   # score = mean delta on weak families - mean delta on stable families (largest loss when ablated on weak families)
class HeadInjector: context manager adding delta (Tensor[dh] in head space or Tensor[d] in residual space mapped through the pseudo-inverse of W_O slice) to head h's dense input at positions
def steer_at_head(mb, input_ids, layer: int, head: int, direction: Tensor[d], eps: float, position: int) -> Tensor[V]   # last-position logits with the residual-space direction applied through head h (least-squares pre-image through W_O slice), dose relative to RMS of the head's contribution
```
Tests (pythia-14m): head_contributions sum (+ dense bias) equals the attention module output (atol 1e-4); ablating all heads equals replacing the attention output by its bias (zero mode); head_direct_logit_effect returns H finite values; ablate_heads context restores the model afterwards; steer_at_head changes only the last-position logits; worst_head picks the head with the largest planted difference on a synthetic effects table.

## `lsne/pile_domains.py`  (empirical difficulty ranking of Pile subfields, DESIGN §4.8b)

```python
PILE_MARKERS: dict[str, list[str]]   # keyword/marker lists per subfield: pubmed, protein, protein_misfolding, code, math, law, news, web, books, wikipedia, ... (substring + regex markers)
def classify_sequence(tokenizer, token_ids: ndarray, markers=None) -> str   # best-matching subfield or "other" (keyword hit counts on the decoded text)
def sample_sequence_difficulty(mb, n_sequences=2000, seed=0, markers=None, cache_dir="cache/pile_nll", max_len=512) -> DataFrame(seq_index, subfield, n_tokens, mean_nll, mean_nll_tail)
    # uses lsne.pile_fetch.fetch_sequence; truncates to max_len for speed; token NLL under mb
def difficulty_ranking(df) -> DataFrame(subfield, n_seqs, mean_nll, sd, rank)   # sorted hardest first
def hardest_subfield(ranking, exclude=("other",)) -> str
```

## `lsne/rdx.py`  (Representational Difference Explanation, DESIGN §4.8a)

```python
def difference_vector(acts_a: ndarray[na,d], acts_b: ndarray[nb,d]) -> ndarray[d]   # mean(a) - mean(b)
def explained_fraction(delta: ndarray[d], basis: ndarray[k,d]) -> float   # ||proj_basis(delta)||^2 / ||delta||^2 (basis rows need not be orthonormal; use lstsq)
def rdx_score(delta: ndarray[d], basis: ndarray[k,d], n_random: int=32, seed: int=0) -> dict(explained, random_mean, random_sd, z, k)
    # random bases of the same k drawn in the delta's ambient space (gaussian, row-normalised)
def rdx_table(mb, contrasts: dict[str, tuple[ndarray,ndarray]], bases: dict[str, ndarray], sites=None, n_random=32, seed=0) -> DataFrame(contrast, basis, site, explained, z, k)
    # contrasts e.g. {"culture_vs_misfold": (acts_culture, acts_misfold), ...}; bases e.g. {"domain_axes": D_dom, "sae": W_dec_subset, "heads": head_dirs, "pca": top_pcs}
def stability(rdx_rows_by_family: dict[str, DataFrame]) -> DataFrame   # mean and across-family sd of explained/z per (contrast, basis, site)
```

## `lsne/sharing.py`  (representational sharing index and network activity, DESIGN §4.8c)

```python
def active_features(sae, acts: ndarray[n,d], threshold: float=0.0, topk_only: bool=True) -> set[int]   # union of active latent indices over the n states
def sharing_index(sets: dict[str, set[int]]) -> DataFrame(a, b, jaccard, n_a, n_b, n_shared)   # pairwise Jaccard between conditions; plus a 'global' row (shared across all / union)
def head_profile_cosines(mb, variants_by_condition: dict[str, list], layer_grid) -> DataFrame(a, b, layer, cosine)   # cosine of mean per-head contribution-norm profiles between conditions
def network_activity(sae, acts, threshold=0.0) -> dict(frac_active, mean_mass, participation_ratio, n_active_mean)
def sharing_report(mb, sae, groups: dict[str, ndarray], site, threshold=0.0) -> dict(sharing: DataFrame, activity: DataFrame)   # groups keyed by condition/domain -> pooled states
```

## `lsne/ttt.py`  (test-time training as structural injection, DESIGN §4.8d)

```python
def load_corpus(path: str, tokenizer, max_docs=None, seq_len=512) -> list[Tensor[T]]   # reads data/ttt/*.txt|*.jsonl; returns tokenised snippets; a small built-in public-domain placeholder when the path is empty/missing
def permute_corpus(snippets, seed=0) -> list[Tensor]   # token-permuted control
@dataclass TTTResult: nll_curve: ndarray[steps+1], steps_to_threshold: int|None, final_nll: float, meta: dict
def memorisation_speed(mb, snippet: Tensor[T], lr=1e-4, steps=50, threshold=0.5, optimizer="adamw", weight_decay=0.0, seed=0, train_params="all") -> TTTResult
    # continued training on ONE snippet (or a small batch); records the snippet's own NLL after each step; does NOT mutate the passed model unless inplace=True (works on a deep copy / state-dict snapshot and restores)
def test_time_train(mb, corpus: list[Tensor], lr=1e-4, steps=200, batch=4, seed=0, control: str|None=None, inplace=False) -> ModelBundle   # returns a bundle with trained weights (copy by default); control in {None,'neutral','nofork','permuted'} selects the corpus variant
def before_after(mb, mb_trained, measure_fn: Callable[[ModelBundle], DataFrame], label="") -> DataFrame   # runs a measurement on both and tags phase=before/after
def ttt_memorisation_table(mb, corpus, controls: dict[str, list], checkpoints: list[str]=None, **kw) -> DataFrame(checkpoint, arm, snippet, steps_to_threshold, final_nll)
```
Tests: pile_domains.classify_sequence on hand-written protein/law/code strings; difficulty_ranking orders a synthetic NLL table; rdx.explained_fraction is 1 when delta lies in the basis span and ~0 for an orthogonal basis; rdx_score z is high for a planted-in-basis delta; sharing_index Jaccard of identical sets is 1 and of disjoint sets 0; network_activity on a synthetic SAE; ttt.memorisation_speed on pythia-14m with one short snippet LOWERS its NLL over a few steps and restores the original weights afterwards (the passed bundle's logits are unchanged); test_time_train(copy) leaves the original model's logits unchanged; load_corpus returns the placeholder when the path is missing. Mark any test that trains for more than a few steps or needs a real corpus as slow.

## `lsne/transplant.py`  (cross-model activation transplant, DESIGN §4.9)

```python
class SourceModel(Protocol):            # pluggable source adapter
    name: str; d_model: int; n_layers: int
    def encode(self, item) -> Tensor[1,T]
    def residuals(self, ids: Tensor[1,T], layers: list[int]|None=None) -> dict[int, Tensor[T,d_S]]   # block inputs per layer (plus n_layers = last block output)
class HFSourceModel(SourceModel): def __init__(self, spec: ModelSpec)   # wraps lsne.model.load_model/run_with_cache; works for any GPT-NeoX/HF causal LM with hidden_states
class TraceSourceStub(SourceModel): def __init__(self, d_model, n_layers, vocab, seed)   # deterministic random-weight stand-in for a BlockGPT-like trace model (documents the interface; NOT a trained model)
@dataclass Alignment: W: Tensor[d_S,d_T], b: Tensor[d_T], method: str, rank: int|None, layer_src: int, layer_tgt: int, r2_train: float, r2_holdout: float, r2_permuted: float, n_pairs: int
def fit_alignment(acts_src: Tensor[N,d_S], acts_tgt: Tensor[N,d_T], method: str="ridge", rank: int|None=None, alpha: float=1.0, holdout: float=0.2, seed: int=0) -> Alignment
    # ridge: closed form on centered data; procrustes: PCA to rank r on both sides then orthogonal Procrustes; r2_permuted from a seeded permutation of the pairs
def layer_matching(src: SourceModel, mb_tgt, pairs: list[tuple[item_src, str]], layers_src=None, layers_tgt=None, readout: str="mean", **fit_kw) -> DataFrame(layer_src, layer_tgt, r2_holdout, r2_permuted, gap)
def transfer_direction(align: Alignment, v_src: Tensor[d_S]) -> Tensor[d_T]      # W^T v (unit-normalised), no bias
def transfer_states(align: Alignment, H_src: Tensor[n,d_S]) -> Tensor[n,d_T]      # H W + b
def substitute_residual(mb, input_ids, site: HookSite, positions: list[int], H_new: Tensor[len,d_T]) -> Tensor[V]   # last-position logits with Injector(mode="replace"); returns logits and (via return_info) the audit
def plant_kv_prefix(mb, input_ids: Tensor[1,T], H_by_layer: dict[int, Tensor[n,d_T]], placeholder_id: int=0, layers: list[int]|None=None) -> dict(logits_last: Tensor[V], logits: Tensor[T,V], n_prefix: int)
    # runs [placeholder*n] + input_ids with Injector(mode="replace") on the block_input of every layer in H_by_layer (default all L) at prefix positions 0..n-1; the prefix's keys/values are thus computed from the transplanted states; returns logits of the real tokens only (positions n..n+T-1)
def plant_kv_cache(mb, input_ids, H_by_layer, **kw) -> dict   # same effect implemented through transformers' cache API (DynamicCache prefill with a hooked prefix forward, then the real tokens with past_key_values); must agree with plant_kv_prefix (test)
def transplant_experiment(mb, src: SourceModel, align: Alignment, v_src: Tensor[d_S], prompts: list[str], site: HookSite, eps: float, n_prefix: int=8, modes=("residual","kv","both"), controls=("random","identity","permuted_alignment"), gen_kw=None) -> DataFrame
    # per prompt × mode × control: logp_target (if target given), entropy, language fraction of a short steered continuation (lsne.generation.language_trajectory), cosine of the transferred direction with the native lang axis if provided
def functional_transfer_test(mb_tgt, src: SourceModel, texts_E: list[str], texts_D: list[str], site_src: int, site_tgt: HookSite, native_axis: Tensor[d_T]|None=None, n_gen: int=16, n_samples: int=4, seed: int=0) -> dict(cos_transferred_vs_native, delta_fD_transferred, delta_fD_native, sign_agreement: bool, r2_holdout, r2_permuted)
```
Tests (CPU): HFSourceModel on pythia-70m (download ~160 MB once; mark slow+network) and pythia-14m as target: fit_alignment r2_holdout > r2_permuted on paired states of ~40 short sentences; procrustes W has orthonormal columns in the PCA bases; transfer_states shape; substitute_residual with the target's own states reproduces its logits (identity control, atol 1e-4); plant_kv_prefix with n=0 equals the plain forward; plant_kv_prefix and plant_kv_cache agree (atol 1e-3); TraceSourceStub implements the protocol and layer_matching returns a table; functional_transfer_test with pythia-14m as BOTH source and target (same model, d_S = d_T: alignment is near-identity, r2 ~ 1) yields sign_agreement True and cos > 0.9 — this is the self-consistency test; a cross-size test (pythia-70m → pythia-14m) is marked slow and only asserts r2_holdout > r2_permuted and that the outputs are finite.

## `lsne/stats.py`

`cluster_bootstrap(df, value_col, cluster_col, n_boot=10000, seed=0, stat=np.mean) -> dict(mean, ci_low, ci_high, n_clusters)`;
`paired_contrast(df, value_col, cond_col, a, b, pair_cols, cluster_col) -> dict`; `bh_correct(pvals) -> ndarray`; `cohen_d`.

## `lsne/io.py`

`RunDir(root, stage, model, revision)` → paths; `write_manifest(run_dir, **meta)` (git sha,
package version, model spec, dtype, hook, anchor hashes, timestamp); `save_df(df, run_dir, name)`
(csv + parquet if available); `drive_root()` (Colab: `/content/drive/MyDrive/LSNE`, else `./runs`).

## `lsne/experiments/`

One module per stage with `run(cfg: dict, run_dir, mb, ...) -> dict[str, DataFrame]`;
`lsne/cli.py` dispatches `python -m lsne.cli run --stage {A,B,C,D,E,F} --model ... --revision ... --families F01,F02 --out ...` and `merge`.
All stage functions must accept `limit` parameters so a CPU smoke run with pythia-14m finishes in < 3 min.

## Tests (`tests/`)

pytest, CPU only, default model `EleutherAI/pythia-14m` revision `step98000` (fixture in
`conftest.py`, session-scoped). Network tests marked `network`; anything downloading
SAEs marked `slow`. SAE unit tests use `TopKSAE.from_tensors` with random weights.
