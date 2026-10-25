"""Shared constants and dataclasses. Signatures here are a contract for all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------------------
# State 2143 anchor (verified 2026-10-25 by HTTP range request + SHA256, see docs/DESIGN.md §1)
# --------------------------------------------------------------------------------------
STATE2143 = dict(
    model="EleutherAI/pythia-160m",
    dataset_repo="EleutherAI/pile-standard-pythia-preshuffled",
    dataset_revision_sha="bac79b6820adb34e451f9a02cc1dc7cd920febf0",  # dossier D1 (not enforced)
    seq_len=2049,
    batch_size=1024,
    sequence_index=5_025_183,
    batch_index0=4907,
    position_in_batch=415,
    context_len=208,
    target_id=22905,          # "bey"
    target_text="bey",
    sha256_sequence="bdd4bbdd2b8138437e052b2f0e3aeb212a563c25fe447c920b0e9c6ee76f55d4",
    sha256_context="84c8f1b6a7a933f53f3b21e353c0c3af219d3ccfbcb7092584a9c6ad1035a750",
    sha256_context_prefix_dossier="84c8f1b6a7a933f5",
    shard_bytes=30_000_000_000,
    n_shards=21,
    shard_fmt="document-{i:05d}-of-00020.bin",
    hook_layer=11,
    hook_site="block_input",
    hook_position=207,
    epsilon=0.01,
    checkpoints_intervention=("step98000", "step99000"),
    checkpoints_exposure=("step4000", "step5000"),
    checkpoint_final="step143000",
)

# Dossier checkpoint trace (27 checkpoints, 4k..143k) used for the training-time sweep.
CHECKPOINT_TRACE = (
    "step1000", "step2000", "step3000", "step4000", "step5000", "step6000", "step8000",
    "step10000", "step20000", "step24000", "step26000", "step27000", "step28000",
    "step30000", "step31000", "step32000", "step98000", "step99000", "step125000",
    "step128000", "step130000", "step131000", "step132000", "step133000", "step135000",
    "step140000", "step143000",
)

# Dose grids (relative RMS dose epsilon)
DOSES = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
DOSES_STEER = (0.2, 0.5, 1.0)

# Switching grid
K_LEVELS = (0, 2, 4, 8)
K_LEVELS_EXTREME = (16, 28)
C_LEVELS = ("C1", "C2", "C3")

# Unit-level language masks (dossier p.23): unit 1 and 10 always E, unit 9 always D if K>0.
MASKS_UNIT_LEVEL = {
    0: "EEEEEEEEEE",
    2: "EEEEDDDDDE",
    4: "EEDDEEDDDE",
    8: "EDEDEDEDDE",
}

# Layer grid for patching / lens analyses (160M: 12 blocks). For other depths use
# config.layer_grid(n_layers).
LAYER_GRID_160M = (2, 5, 8, 11)


def layer_grid(n_layers: int) -> tuple[int, ...]:
    """Four roughly equally spaced block indices including the last one."""
    if n_layers <= 4:
        return tuple(range(n_layers))
    step = (n_layers - 1) / 3.0
    return tuple(sorted({int(round(i * step)) for i in range(4)}))


SAE_REPOS = {
    "EleutherAI/pythia-70m": ("EleutherAI/sae-pythia-70m-32k", "layers.{n}"),
    "EleutherAI/pythia-160m": ("EleutherAI/sae-pythia-160m-32k", "layers.{n}"),
    "EleutherAI/pythia-160m-deduped": ("EleutherAI/sae-pythia-160m-deduped-32k", "layers.{n}"),
    "EleutherAI/pythia-410m": ("EleutherAI/sae-pythia-410m-65k", "layers.{n}.mlp"),  # MLP outputs only
}


@dataclass
class ModelSpec:
    name: str = STATE2143["model"]
    revision: str = "step98000"
    dtype: str = "float32"            # "float32" | "bfloat16" | "float16"
    device: str = "auto"              # "auto" | "cpu" | "cuda" | "cuda:0"
    attn_implementation: str = "eager"

    def tag(self) -> str:
        return f"{self.name.split('/')[-1]}@{self.revision}:{self.dtype}"


@dataclass(frozen=True)
class HookSite:
    layer: int
    site: str = "block_input"   # block_input | block_output | mlp_output | attn_output

    def __post_init__(self):
        if self.site not in ("block_input", "block_output", "mlp_output", "attn_output"):
            raise ValueError(f"unknown site {self.site}")

    @property
    def key(self) -> str:
        return f"L{self.layer}.{self.site}"

    @property
    def sae_hookpoint(self) -> Optional[str]:
        """EleutherAI sparsify hookpoint name for this site (residual-stream SAEs)."""
        if self.site == "block_output":
            return f"layers.{self.layer}"
        if self.site == "block_input":
            return f"layers.{self.layer - 1}" if self.layer > 0 else None
        if self.site == "mlp_output":
            return f"layers.{self.layer}.mlp"
        return None


def state2143_site() -> HookSite:
    return HookSite(layer=STATE2143["hook_layer"], site=STATE2143["hook_site"])


@dataclass
class RunConfig:
    """Top-level configuration shared by all stages (serialised into the manifest)."""
    model: ModelSpec = field(default_factory=ModelSpec)
    families_dir: str = "data/stimuli/families"
    families: Optional[list[str]] = None      # None = all
    k_levels: tuple = K_LEVELS
    k_extreme: tuple = K_LEVELS_EXTREME
    c_levels: tuple = C_LEVELS
    doses: tuple = DOSES
    doses_steer: tuple = DOSES_STEER
    n_random_dirs: int = 32
    seed: int = 0
    out_root: str = "runs"
    limit_variants: Optional[int] = None      # smoke-test limiter
    limit_families: Optional[int] = None
    gen_n_new: int = 64
    gen_n_samples: int = 8
    gen_temperature: float = 1.0
    tn_rank: int = 64
    tn_window: int = 32
