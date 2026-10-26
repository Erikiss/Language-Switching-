#!/usr/bin/env python
"""Generate the Colab notebooks ``notebooks/0*.ipynb`` with nbformat.

The notebooks are *generated*, never edited by hand: every cell source lives in this
script, the notebooks are rebuilt with ::

    python scripts/build_notebooks.py            # writes notebooks/*.ipynb
    python scripts/build_notebooks.py --check    # exit 1 if the committed files drift
    python scripts/build_notebooks.py --out DIR  # write somewhere else (tests)

and ``tests/test_notebooks.py`` asserts that the committed notebooks equal the
regenerated ones.  The build is deterministic (fixed cell ids, no outputs, no
timestamps) so that ``git diff`` on the notebooks only shows real changes.

Every notebook has the same skeleton (German prose, English code comments):

1. markdown header: purpose, inputs, outputs, dossier stage,
2. setup cell: clone / update the repository from ``GIT_URL`` into ``/content/lsne-repo``
   (skipped when the notebook already runs inside a checkout), ``pip install -r
   requirements.txt`` and ``pip install -e .``,
3. optional TNCO cell (``apt-get install libboost-dev`` + ``pip install tnco``), with the
   ``opt_einsum`` fallback explained,
4. Drive mount cell (``google.colab.drive``, guarded by ``try/except`` so that the notebook
   also runs outside Colab) defining ``OUT_ROOT = lsne.io.drive_root()``,
5. configuration cell (``MODEL``, ``REVISION``, ``DTYPE``, ``FAMILIES``, ``SMOKE`` ...),
6. model loading and a small runner (``run_stage``) that calls the stage ``run()``
   functions directly, keeps the tables in memory, prints a traceback on failure and
   continues (the CLI equivalent is shown as a comment),
7. plotting cells (matplotlib) for the key tables of the stage,
8. a final cell writing a short run summary JSON into ``OUT_ROOT``.

Every code cell is compiled at build time so that a syntax error fails the build (and
the test), not the Colab session.

Notebooks 00-05 cover the dossier stages A-F, notebooks 06-11 the extension stages G-L
(docs/STAGES.md: local geometry / trajectory, lambda dynamics, domain compass + heads,
RDX + Pile difficulty + sharing, test-time training, cross-model transplant).  The G-L
notebooks add a parameter cell for the stage's user-supplied inputs (atlas extraction,
TTT corpus / checkpoints, transplant source), a substage report read from the manifest
(``substages``, ``prereg``) and a header paragraph naming the pre-registered contrasts
and the tables that carry them.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

GIT_URL = "https://github.com/Erikiss/Language-Switching-.git"
GIT_BRANCH = "claude/language-switching-non-equilibrium-ejpz0q"
REPO_DIR = "/content/lsne-repo"

#: notebook file stem -> (stage letter, title)
NOTEBOOKS: dict[str, tuple[str, str]] = {
    "00_setup_and_anchor": ("A", "Setup, Anker und Stimuli"),
    "01_baseline_and_directions": ("B", "Baselines und Richtungen"),
    "02_switching_grid": ("C", "Sprachwechsel-Raster und Counterfactual-Attribution"),
    "03_nonequilibrium": ("D", "Nicht-Gleichgewichts-Diagnostiken"),
    "04_tn_linear_response": ("E", "Tensornetzwerk und lineare Antwort"),
    "05_scale_and_checkpoints": ("F", "Skalierung und Checkpoint-Sweep"),
    "06_geometry_trajectory": ("G", "Lokale Geometrie und Trajektorie"),
    "07_lambda_dynamics": ("H", "λ-Dynamik im Aktivierungsraum"),
    "08_compass_heads": ("I", "Domänen-Kompass, Domänenachsen und Heads"),
    "09_rdx_sharing": ("J", "RDX, Pile-Schwierigkeit und Sharing"),
    "10_ttt_injection": ("K", "Test-Time-Training als strukturelle Injektion"),
    "11_transplant": ("L", "Cross-Modell-Transplantation"),
}


# ======================================================================================
# helpers
# ======================================================================================
def _src(text: str, **subst: str) -> str:
    """Dedent a cell source and substitute ``{{NAME}}`` placeholders."""
    out = textwrap.dedent(text).strip("\n") + "\n"
    for k, v in subst.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def md(text: str, **subst: str):
    return ("markdown", _src(text, **subst))


def code(text: str, **subst: str):
    src = _src(text, **subst)
    compile(src, "<notebook cell>", "exec")        # syntax check at build time
    return ("code", src)


# ======================================================================================
# shared cells
# ======================================================================================
def header_cell(stem: str, stage: str, title: str, purpose: str, inputs: str, outputs: str,
                steps: str, prereg: str = "") -> tuple[str, str]:
    """Markdown header; ``prereg`` (stages G-L) adds the paragraph naming the pre-registered
    contrasts of the stage and the tables that carry them (docs/STAGES.md)."""
    docs = "siehe `docs/DESIGN.md` §8" + (" und `docs/STAGES.md`" if stage in "GHIJKL" else "")
    head = """
        # LS-NEQ {{STEM}} — {{TITLE}}

        **Dossier-Stufe {{STAGE}}** (`lsne.experiments.STAGES["{{STAGE}}"]`, {{DOCS}}).
        Dieses Notebook wird von `scripts/build_notebooks.py` erzeugt; Änderungen bitte dort
        vornehmen und neu generieren (`python scripts/build_notebooks.py`).

        **Zweck.** {{PURPOSE}}

        **Eingaben.** {{INPUTS}}

        **Ausgaben.** {{OUTPUTS}}
        Alle Tabellen liegen als CSV (+ Parquet) im Laufverzeichnis
        `<RUNS_ROOT>/{{STAGE}}/<modell>@<revision>/` zusammen mit `manifest.json`
        (Git-SHA, Modell, dtype, Hook, Anker-Hashes, Zeitstempel); Abbildungen unter
        `<OUT_ROOT>/figures/{{STEM}}/`; am Ende eine kurze `summary_{{STEM}}_*.json` in `OUT_ROOT`.

        **Ablauf.** {{STEPS}}
"""
    prereg_block = """
        **Präregistrierung.** {{PREREG}}
""" if prereg else ""
    tail = """
        Hinweise: Die Zellen sind gegen Fehler abgesichert (ein fehlgeschlagener Schritt druckt
        den Traceback und das Notebook läuft weiter). Mit `SMOKE = True` läuft alles in
        < 3 min auf CPU mit `pythia-14m` (Rauchtest, keine wissenschaftliche Aussage).
        Mit `REUSE = True` werden vorhandene Ergebnisse (manifest.json mit `status: ok`)
        geladen statt neu gerechnet — so lässt sich nach einem Colab-Abbruch fortsetzen.
        """
    return md(
        head + prereg_block + tail,
        STEM=stem, TITLE=title, STAGE=stage, DOCS=docs, PURPOSE=purpose, INPUTS=inputs, OUTPUTS=outputs, STEPS=steps,
        PREREG=prereg,
    )


def setup_cell() -> tuple[str, str]:
    return code(
        """
        # --- Setup: Repository holen / aktualisieren und installieren -------------------------
        # Runs inside Colab (clone into /content/lsne-repo) and inside an existing checkout
        # (local Jupyter, tests): a checkout is detected by lsne/__init__.py + pyproject.toml.
        import importlib.util
        import os
        import pathlib
        import subprocess
        import sys

        GIT_URL = os.environ.get("LSNE_GIT_URL", "{{GIT_URL}}")
        GIT_BRANCH = os.environ.get("LSNE_GIT_BRANCH", "{{GIT_BRANCH}}")
        REPO_DIR = os.environ.get("LSNE_REPO_DIR", "{{REPO_DIR}}")


        def _sh(cmd: str) -> int:
            print("$", cmd, flush=True)
            return subprocess.run(cmd, shell=True).returncode


        _here = pathlib.Path.cwd().resolve()
        _local = next((p for p in [_here, *_here.parents]
                       if (p / "lsne" / "__init__.py").exists() and (p / "pyproject.toml").exists()), None)
        if _local is not None:
            REPO_DIR = str(_local)
            print("Vorhandener Checkout:", REPO_DIR)
        elif pathlib.Path(REPO_DIR, ".git").exists():
            _sh(f"git -C {REPO_DIR} fetch -q --depth 1 origin {GIT_BRANCH} && "
                f"git -C {REPO_DIR} checkout -q {GIT_BRANCH} && git -C {REPO_DIR} pull -q --ff-only origin {GIT_BRANCH}")
        else:
            _sh(f"git clone -q --depth 1 --branch {GIT_BRANCH} {GIT_URL} {REPO_DIR}")
        os.chdir(REPO_DIR)
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)

        if _local is None or importlib.util.find_spec("lsne") is None:
            _sh(f"{sys.executable} -m pip install -q -r requirements.txt")
            _sh(f"{sys.executable} -m pip install -q -e .")
        import lsne
        print("lsne", lsne.__version__, "aus", REPO_DIR, "| Branch", GIT_BRANCH)
        print("Git-SHA:", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
        """,
        GIT_URL=GIT_URL, GIT_BRANCH=GIT_BRANCH, REPO_DIR=REPO_DIR,
    )


def tnco_cell(default_install: bool) -> tuple[str, str]:
    return code(
        """
        # --- Optional: TNCO (Google Research) ---------------------------------------------------
        # TNCO optimises the contraction ORDER of the linearised lattice network (stage E).
        # It needs the Boost headers and a C++ build (a few minutes on Colab).  Without TNCO
        # lsne.tn_lattice.optimize_path falls back to opt_einsum ("dp" for <= 12 tensors,
        # else "greedy"); the contracted response is identical, only the path cost / runtime
        # columns of tn_paths differ ("fallback_reason" says why).  Stages A-D never need it.
        INSTALL_TNCO = {{DEFAULT}}   # True: bauen (nur für Notebook 04 / Stufe E sinnvoll)

        try:
            import tnco
            print("tnco", getattr(tnco, "__version__", "?"), "ist bereits installiert")
        except ImportError:
            if INSTALL_TNCO:
                _sh("apt-get install -y -q libboost-dev > /tmp/apt_boost.log 2>&1 || cat /tmp/apt_boost.log | tail -5")
                _sh(f'{sys.executable} -m pip install -q "tnco @ git+https://github.com/google-research/tnco"')
                try:
                    import tnco
                    print("tnco", getattr(tnco, "__version__", "?"), "installiert")
                except ImportError as e:
                    print("TNCO konnte nicht gebaut werden:", repr(e), "-> Fallback opt_einsum (siehe docs/RUNBOOK_COLAB.md)")
            else:
                print("TNCO nicht installiert -> lsne.tn_lattice nutzt opt_einsum (dp/greedy) als Fallback.")
        """,
        DEFAULT="True" if default_install else "False",
    )


def drive_cell() -> tuple[str, str]:
    return code(
        """
        # --- Google Drive einbinden (außerhalb von Colab wird lokal geschrieben) ---------------
        import os

        try:
            from google.colab import drive            # type: ignore
            drive.mount("/content/drive")
        except Exception as e:                        # not on Colab, or the mount was refused
            print("Kein Colab-Drive:", repr(e).split("(")[0], "-> lokale Ausgabe")

        from lsne.io import drive_root

        OUT_ROOT = drive_root()                       # Colab: /content/drive/MyDrive/LSNE, sonst $LSNE_RUNS oder ./runs
        RUNS_ROOT = os.path.join(OUT_ROOT, "runs")    # Laufverzeichnisse wie in jobs/shards.yaml (--out)
        os.makedirs(RUNS_ROOT, exist_ok=True)
        if OUT_ROOT.startswith("/content/drive"):
            # share the anchor / unigram caches between sessions (see jobs/shards.yaml defaults.env)
            os.environ.setdefault("LSNE_CACHE", os.path.join(OUT_ROOT, "cache"))
        print("OUT_ROOT  =", OUT_ROOT)
        print("RUNS_ROOT =", RUNS_ROOT)
        print("LSNE_CACHE =", os.environ.get("LSNE_CACHE", "./cache"))
        """
    )


def config_cell(stem: str, stage: str, extra_lines: str = "") -> tuple[str, str]:
    return code(
        """
        # --- Konfiguration (hier anpassen) ------------------------------------------------------
        NOTEBOOK = "{{STEM}}"
        STAGE = "{{STAGE}}"
        MODEL = "EleutherAI/pythia-160m"   # HF-Modellname (State 2143: pythia-160m)
        REVISION = "step98000"             # HF-Revision: step98000 / step99000 (Eingriff), step143000 (SAE)
        DTYPE = "float32"                  # "float32" (Standard) | "bfloat16" (eigener Lauf, eigene Null-Dosis-Baseline)
        DEVICE = "auto"                    # "auto" | "cuda" | "cpu"
        FAMILIES = None                    # z. B. ["F01", "F02"] oder ["DEV01A"]; None = alle Familien (F01-F12, DEV01-DEV04)
        SMOKE = False                      # True: Mini-Lauf mit pythia-14m, 1 Familie, 3 Varianten (< 3 min CPU)
        SEED = 0
        REUSE = True                       # vorhandene Ergebnisse (manifest.json, status ok) laden statt neu rechnen
        USE_SAE = False                    # SAE-Feature-Richtungen (EleutherAI/sae-pythia-160m-32k; nur bei step143000 zertifizierbar)
        {{EXTRA}}
        from lsne.config import ModelSpec, RunConfig
        from lsne.experiments import common as C

        if SMOKE:
            MODEL, REVISION, DTYPE = "EleutherAI/pythia-14m", "step98000", "float32"
            FAMILIES = FAMILIES or ["DEV01"]
        spec = ModelSpec(name=MODEL, revision=REVISION, dtype=DTYPE, device=DEVICE)
        cfg = RunConfig(model=spec, families=FAMILIES, seed=SEED, out_root=RUNS_ROOT)
        limits = C.Limits()
        if SMOKE:
            cfg, limits = C.smoke_config(cfg), C.Limits.smoke()
        limits.use_sae = bool(USE_SAE) and not SMOKE
        limits.progress = True

        _fam = f" --families {','.join(FAMILIES)}" if FAMILIES else ""
        CLI_COMMAND = (f"python -m lsne.cli run --stage {STAGE} --model {MODEL} --revision {REVISION} --dtype {DTYPE}"
                       f" --device {DEVICE} --out {RUNS_ROOT} --seed {SEED}{_fam}{' --smoke' if SMOKE else ''}"
                       f"{' --use-sae' if limits.use_sae else ''}")
        print("Modell:", spec.tag(), "| Familien:", FAMILIES or "alle", "| SMOKE:", SMOKE)
        print("CLI-Äquivalent:\\n  " + CLI_COMMAND)
        """,
        STEM=stem, STAGE=stage, EXTRA=extra_lines,
    )


def model_cell() -> tuple[str, str]:
    return code(
        """
        # --- Modell laden -------------------------------------------------------------------
        import torch
        from lsne.model import load_model

        mb = load_model(spec)
        print(f"{spec.tag()}: {mb.n_layers} Blöcke, d_model={mb.d_model}, {mb.n_heads} Heads, "
              f"Vokabular={mb.vocab}, device={mb.device}, dtype={mb.dtype}")
        print("Dossier-Eingriffsstelle (verallgemeinert):", C.dossier_site(mb).key,
              "| Tiefen-Injektion (D1b / Stufe E):", C.depth_site(mb).key)
        """
    )


def runner_cell() -> tuple[str, str]:
    return code(
        """
        # --- Runner: Stufen direkt über run() ausführen (Ergebnisse bleiben im Speicher) --------
        import contextlib
        import json
        import time
        import traceback

        import numpy as np
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt

        from lsne.experiments import STAGES, STAGE_TITLES
        from lsne.io import RunDir, load_df, read_manifest

        ctx: dict = {}        # shared objects between stages (variants, original, direction bank)
        results: dict = {}    # stage -> {table name: DataFrame}
        report: dict = {}     # stage -> status / runtime / error
        FIG_DIR = os.path.join(OUT_ROOT, "figures", NOTEBOOK)
        os.makedirs(FIG_DIR, exist_ok=True)


        def run_stage(stage: str, reuse=None, **kw):
            \"\"\"Run one stage; on failure print the traceback, write error.txt and continue.\"\"\"
            reuse = REUSE if reuse is None else reuse
            rd = RunDir(cfg.out_root, stage, cfg.model)
            t0 = time.perf_counter()
            if reuse and rd.manifest_path.exists():
                try:
                    m = read_manifest(rd)
                    if m.get("status") == "ok" and m.get("tables"):
                        results[stage] = {name: load_df(rd, name) for name in m["tables"]}
                        report[stage] = dict(status="reused", run_dir=str(rd.path), seconds=0.0, error="")
                        print(f"Stufe {stage}: vorhandene Ergebnisse geladen aus {rd.path} "
                              f"(REUSE=False erzwingt Neuberechnung)")
                        return results[stage]
                except Exception as e:  # unreadable previous run: recompute
                    print("Wiederverwendung nicht möglich:", repr(e))
            print(f"Stufe {stage} ({STAGE_TITLES[stage]}) -> {rd.path}", flush=True)
            try:
                results[stage] = STAGES[stage].run(cfg, mb, rd, ctx=ctx, limits=limits, **kw)
                report[stage] = dict(status="ok", run_dir=str(rd.path), seconds=time.perf_counter() - t0, error="")
                print(f"Stufe {stage} ok in {report[stage]['seconds']:.0f} s: {', '.join(sorted(results[stage]))}")
            except Exception as e:
                traceback.print_exc()
                C.write_error(rd, stage, e, cfg, mb)
                report[stage] = dict(status="error", run_dir=str(rd.path), seconds=time.perf_counter() - t0, error=repr(e))
                results[stage] = {}
                print(f"Stufe {stage} FEHLGESCHLAGEN: {e!r} (error.txt in {rd.path})")
            return results[stage]


        def table(stage: str, name: str) -> pd.DataFrame:
            \"\"\"Table of a stage result or an empty DataFrame (keeps the plot cells robust).\"\"\"
            df = results.get(stage, {}).get(name)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


        def show(df, n: int = 40):
            \"\"\"display() in Jupyter / Colab, print() elsewhere.\"\"\"
            if isinstance(df, pd.DataFrame):
                df = df.head(n)
            try:
                display(df)          # noqa: F821 - provided by IPython
            except NameError:
                print(df.to_string() if hasattr(df, "to_string") else df)


        def savefig(name: str):
            path = os.path.join(FIG_DIR, f"{name}_{spec.tag().replace(':', '_').replace('@', '_')}.png")
            plt.gcf().savefig(path, dpi=120, bbox_inches="tight")
            print("Abbildung:", path)


        @contextlib.contextmanager
        def guarded(title: str):
            \"\"\"Print the traceback of a failing cell section and carry on.\"\"\"
            print("==", title)
            try:
                yield
            except Exception:
                traceback.print_exc()
                print(f"[{title}] übersprungen (siehe Traceback)")
        """
    )


def summary_cell() -> tuple[str, str]:
    return code(
        """
        # --- Kurze Laufzusammenfassung nach OUT_ROOT schreiben ---------------------------------
        with guarded("Zusammenfassung schreiben"):
            import datetime

            summary = dict(
                notebook=NOTEBOOK, stage=STAGE, model=spec.name, revision=spec.revision, dtype=spec.dtype,
                device=str(mb.device), n_layers=mb.n_layers, d_model=mb.d_model, smoke=bool(SMOKE),
                families=FAMILIES, seed=SEED, out_root=OUT_ROOT, runs_root=RUNS_ROOT, cli_command=CLI_COMMAND,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                report=report,
                tables={st: {n: int(len(df)) for n, df in r.items()} for st, r in results.items()},
                figures=sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png")),
            )
            path = os.path.join(OUT_ROOT, f"summary_{NOTEBOOK}_{spec.tag().replace(':', '_').replace('@', '_')}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(C.to_jsonable(summary), f, indent=1, ensure_ascii=False)
            print("Zusammenfassung:", path)
            print(json.dumps(C.to_jsonable(summary), indent=1, ensure_ascii=False)[:3000])
        """
    )


def substages_cell() -> tuple[str, str]:
    """Stages G-L: substage report from the manifest + a helper for boolean csv columns."""
    return code(
        """
        # --- Substufen-Bericht: die Stufen G-L fangen Fehler pro Substufe ab (docs/STAGES.md) ----
        # A failed substage is recorded in manifest.json -> substages.errors (traceback in
        # substage_errors.txt), its table stays empty and the stage goes on; the manifest's
        # `prereg` map names the table / rows that carry every pre-registered contrast.
        def substage_report(stage: str) -> dict:
            \"\"\"Print the substage report and the pre-registration map of a G-L manifest.\"\"\"
            rd = RunDir(cfg.out_root, stage, cfg.model)
            if not rd.manifest_path.exists():
                print(f"Stufe {stage}: kein manifest.json in {rd.path}")
                return {}
            m = read_manifest(rd)
            sub = m.get("substages") or {}
            rt = m.get("runtime_s")
            print(f"Stufe {stage}: Status {m.get('status')}" + (f", {rt:.0f} s" if isinstance(rt, (int, float)) else "")
                  + f", Substufen fehlgeschlagen: {sub.get('n_failed', '?')}"
                  + (" | STUB: lsne.transplant nicht importierbar" if m.get("stub") else ""))
            for name, err in (sub.get("errors") or {}).items():
                print(f"  FEHLER in Substufe {name}: {err}")
            if sub.get("n_failed"):
                print(f"  Tracebacks: {rd.path / 'substage_errors.txt'}")
            for k, v in (m.get("prereg") or {}).items():
                print(f"  Präregistrierung {k}: {v}")
            return m


        def truthy(s: pd.Series) -> pd.Series:
            \"\"\"Boolean view of a flag column (bool in memory, 'True'/'False' strings after a csv reload).\"\"\"
            return s.astype(str).str.lower().eq("true")
        """
    )


# ======================================================================================
# 00 setup and anchor (stage A)
# ======================================================================================
def nb_00() -> list[tuple[str, str]]:
    cells = [
        header_cell(
            "00_setup_and_anchor", "A", "Setup, Anker und Stimuli",
            purpose=("Installiert das Paket `lsne`, bindet Drive ein, prüft die Identität von State 2143 "
                     "(Sequenz 5 025 183 per HTTP-Range-Request, SHA256 der 2049 Tokens und der 208 "
                     "Kontext-Tokens, Ziel 22905 „bey“) und baut die Stimulusbank: Familien × Formulierung × "
                     "Wechselzahl K × Strukturstufe C, mit Konsistenzprüfungen (Ziel-ID, Tail-IDs, Ziel-Leak) "
                     "und Token-Annotation."),
            inputs="`data/stimuli/families/*.json`, `data/stimuli/original_2143.json`, der Anker (Netzwerk, gecacht unter `LSNE_CACHE`).",
            outputs="Tabellen `families`, `variants`, `checks`, `exclusions`, `token_classes`, `tokens`.",
            steps=("Setup → Drive → Konfiguration → Modell → Anker-Verifikation (Hashes, dekodierter Kontext) → "
                   "Stufe A → Variantentabelle, Ausschlüsse, Tokenklassen-Zählung → Zusammenfassung."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("00_setup_and_anchor", "A"),
        model_cell(),
        runner_cell(),
        code(
            """
            # --- Anker-Verifikation: State 2143 ----------------------------------------------------
            # The sequence is fetched by HTTP range request (4098 bytes of one shard of
            # EleutherAI/pile-standard-pythia-preshuffled), cached under LSNE_CACHE/state2143 and
            # compared with the dossier hashes.  The text is never written into the repository.
            with guarded("State-2143-Anker"):
                from lsne.anchor import load_anchor, tokenizer_roundtrip_check
                from lsne.config import STATE2143

                anchor = load_anchor(cache_dir=C.cache_dir("state2143"), tokenizer=mb.tokenizer, strict=False)
                print("Sequenzindex      :", STATE2143["sequence_index"], "=", STATE2143["batch_index0"], "*",
                      STATE2143["batch_size"], "+", STATE2143["position_in_batch"])
                print("SHA256 Sequenz    :", anchor.sha256_sequence)
                print("   erwartet       :", STATE2143["sha256_sequence"], "| ok:", anchor.sha256_sequence == STATE2143["sha256_sequence"])
                print("SHA256 Kontext    :", anchor.sha256_context)
                print("   erwartet       :", STATE2143["sha256_context"], "| ok:", anchor.sha256_context == STATE2143["sha256_context"],
                      "| Dossier-Präfix:", anchor.sha256_context.startswith(STATE2143["sha256_context_prefix_dossier"]))
                print("Ziel-ID           :", anchor.target_id, "| erwartet:", STATE2143["target_id"], "| Zieltext:", repr(anchor.target_text))
                print("Anker verifiziert :", anchor.verified)
                print("Tokenizer-Roundtrip:", tokenizer_roundtrip_check(mb.tokenizer, anchor))
                print("\\n--- Dekodierter Kontext (208 Tokens) ---")
                print(anchor.text)
                print("\\n--- Fortsetzung im Datensatz (32 Tokens, beginnt mit dem Ziel) ---")
                print(anchor.continuation_text)
            """
        ),
        code(
            """
            # --- Stufe A ausführen ----------------------------------------------------------------
            # CLI-Äquivalent (siehe CLI_COMMAND oben):
            #   python -m lsne.cli run --stage A --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT>
            res_A = run_stage("A")
            print({k: len(v) for k, v in res_A.items()})
            """
        ),
        code(
            """
            # --- Familien und Varianten -----------------------------------------------------------
            with guarded("Familien"):
                fam = table("A", "families")
                show(fam[[c for c in ["tag", "kind", "title", "n_units", "target", "target_id", "dev"] if c in fam.columns]], n=60)

            with guarded("Variantentabelle"):
                var = table("A", "variants")
                cols = [c for c in ["tag", "kind", "K", "C", "mask", "n_tokens", "n_words", "n_tokens_D", "F100", "n_switches",
                                    "last_switch_pos", "tail_start", "target", "target_id"] if c in var.columns]
                print(f"{len(var)} Varianten; davon original: {int((var['kind'] == 'original').sum()) if 'kind' in var else 0}")
                show(var[cols], n=60)
                std = var[var["kind"] == "standard"] if "kind" in var else var
                print("\\nVarianten pro (K, C):")
                show(std.groupby(["K", "C"]).size().rename("n").reset_index())

            with guarded("Prüfungen und Ausschlüsse"):
                chk, exc = table("A", "checks"), table("A", "exclusions")
                print(f"{len(chk)} geprüfte Varianten, {len(exc)} ausgeschlossen")
                if len(exc):
                    show(exc, n=100)
                else:
                    print("Keine Ausschlüsse (Ziel-ID, Tail-IDs und Ziel-Leak in allen Varianten konsistent).")
            """
        ),
        code(
            """
            # --- Tokenklassen: Anteile pro K und Sprache (Längen-Konfundierung sichtbar machen) -----
            with guarded("Tokenklassen"):
                tc = table("A", "token_classes")
                std = tc[tc["kind"] == "standard"] if "kind" in tc else tc
                piv = std.groupby(["K", "klass"])["n"].sum().unstack("klass").fillna(0)
                share = piv.div(piv.sum(axis=1), axis=0)
                ax = share.plot(kind="bar", stacked=True, figsize=(8, 4), width=0.75)
                ax.set_ylabel("Anteil der Tokens")
                ax.set_xlabel("K (Wechselzahl)")
                ax.set_title("Tokenklassen pro K (ws, punct, boundary, word_start, word_cont)")
                ax.legend(title="Klasse", bbox_to_anchor=(1.02, 1), loc="upper left")
                savefig("token_classes_by_K")
                plt.show()
                print("Absolute Zählungen (Klasse × Sprache):")
                show(std.groupby(["lang", "klass"])["n"].sum().unstack("klass").fillna(0).astype(int))

            with guarded("Tokens pro Wort nach Sprache"):
                var = table("A", "variants")
                std = var[var["kind"] == "standard"] if "kind" in var else var
                if {"n_tokens_E", "n_words_E", "n_tokens_D", "n_words_D"} <= set(std.columns):
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    g = std.groupby("K")[["n_tokens", "n_tokens_D", "n_words_D"]].mean()
                    ax.plot(g.index, g["n_tokens"], "o-", label="Tokens gesamt")
                    ax.plot(g.index, g["n_tokens_D"], "s-", label="Tokens deutsch")
                    ax.plot(g.index, g["n_words_D"], "^-", label="Wörter deutsch")
                    ax.set_xlabel("K")
                    ax.set_ylabel("Mittel pro Variante")
                    ax.set_title("Deutsche Segmente erzeugen mehr Subwort-Tokens (Kovariate, DESIGN §2.4)")
                    ax.legend()
                    savefig("length_covariate")
                    plt.show()
            """
        ),
        summary_cell(),
    ]
    return cells


# ======================================================================================
# 01 baseline and directions (stage B)
# ======================================================================================
def nb_01() -> list[tuple[str, str]]:
    return [
        header_cell(
            "01_baseline_and_directions", "B", "Baselines und Richtungen",
            purpose=("Misst für jede Variante die unveränderte Baseline (log p(Ziel), Entropie, Masse auf den 100 "
                     "häufigsten Tokens) und baut die Richtungsbank an der Dossier-Stelle (Block-Eingang des "
                     "letzten Blocks, letzte Kontextposition): `ent` (∂H/∂h, State-2143-Konstruktion), `lang_dom` "
                     "(Difference-of-Means D−E, denoised), `lang_probe`, `freq` (Häufigkeitsachse), 32 Zufallsrichtungen, "
                     "optional SAE-Feature-Richtungen. Geometriebericht: Kosinusmatrix, Unembedding-Projektion "
                     "(beförderte/unterdrückte Tokens), P_freq vs. Entropie."),
            inputs="Stimulusbank (Stufe A wird bei Bedarf intern neu aufgebaut), Anker, Unigramm-Zählung (Range-Requests, gecacht).",
            outputs=("Tabellen `baseline`, `direction_meta`, `cosines`, `unembed`, `pfreq_entropy_tokens`, "
                     "`pfreq_entropy_by_class` [+ `sae_features`, `sae_summary`]; Richtungsbank `directions.npz` + "
                     "`directions_meta.json` (von den Stufen C–E wiederverwendet)."),
            steps="Setup → Drive → Konfiguration → Modell → Stufe B → Kosinus-Heatmap → Unembedding-Tokens → P_freq-vs-Entropie-Streudiagramm → Zusammenfassung.",
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("01_baseline_and_directions", "B",
                    extra_lines='AXIS_NPZ = None                    # optional: axes.npz mit ARI-Achse, gegen die "ent" orthogonalisiert wird'),
        model_cell(),
        runner_cell(),
        code(
            """
            # --- Stufe B ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage B --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> [--use-sae] [--axis-npz axes.npz]
            res_B = run_stage("B", axis_npz=AXIS_NPZ)
            print({k: len(v) for k, v in res_B.items()})
            if "bank" in ctx:
                bank = ctx["bank"]
                print("Richtungen:", list(bank.keys()))
                print("Übersprungen:", bank.meta.get("skipped"), "| Unigramm-Quelle:", bank.meta.get("unigram"))
                if "sae" in bank.meta:
                    print("SAE:", {k: v for k, v in bank.meta["sae"].items() if k not in ("features_D", "features_E")})
            """
        ),
        code(
            """
            # --- Baseline pro Bedingung -----------------------------------------------------------
            with guarded("Baseline log p(Ziel) nach K und C"):
                base = table("B", "baseline")
                std = base[base["kind"] == "standard"] if "kind" in base else base
                g = std.groupby(["K", "C"])[["logp_target", "p_target", "entropy", "p_freq", "target_rank"]].agg(["mean", "std"])
                show(g)
                fig, ax = plt.subplots(figsize=(7, 3.6))
                for Cc, gg in std.groupby("C"):
                    m = gg.groupby("K")["logp_target"].agg(["mean", "std"])
                    ax.errorbar(m.index, m["mean"], yerr=m["std"].fillna(0), marker="o", capsize=3, label=Cc)
                ax.set_xlabel("K")
                ax.set_ylabel("ln p₀(Ziel)")
                ax.set_title("Reiner Texteffekt des Sprachwechsels auf die Zielvorhersage")
                ax.legend(title="C")
                savefig("baseline_logp_by_K")
                plt.show()
            """
        ),
        code(
            """
            # --- Kosinus-Heatmap der Richtungen ---------------------------------------------------
            with guarded("Kosinus-Heatmap"):
                cos = table("B", "cosines").set_index("name")
                keep = [n for n in cos.index if not n.startswith("rand_") or n in ("rand_00", "rand_01")]
                M = cos.loc[keep, keep].astype(float)
                n = len(keep)
                fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
                im = ax.imshow(M.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
                ax.set_xticks(range(n))
                ax.set_xticklabels(keep, rotation=90)
                ax.set_yticks(range(n))
                ax.set_yticklabels(keep)
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f"{M.iat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
                meta = table("B", "direction_meta")
                site_key = str(meta["site"].iloc[0]) if len(meta) else C.dossier_site(mb).key
                ax.set_title(f"Kosinus der Richtungen an {site_key} (Zufallsrichtungen: nur rand_00/01 gezeigt)")
                fig.colorbar(im, ax=ax, label="Kosinus", shrink=0.8)
                savefig("direction_cosines")
                plt.show()
                if len(meta):
                    cols = [c for c in ["name", "dir_family", "site", "train_acc", "entropy", "source", "fvu", "certified", "selectivity"] if c in meta.columns]
                    show(meta[cols], n=60)
            """
        ),
        code(
            """
            # --- Unembedding: welche Tokens befördert / unterdrückt jede Richtung? ------------------
            with guarded("Unembedding-Projektion"):
                un = table("B", "unembed")
                names = [n for n in ["ent", "lang_dom", "lang_probe", "freq", "sae_D0", "sae_E0", "rand_00"] if n in set(un["direction"])]
                for name in names:
                    sub = un[un["direction"] == name]
                    pro = sub[sub["kind"] == "promoted"].sort_values("rank")["token"].head(15).tolist()
                    sup = sub[sub["kind"] == "suppressed"].sort_values("rank")["token"].head(15).tolist()
                    print(f"{name:11s} befördert : {pro}")
                    print(f"{'':11s} unterdrückt: {sup}")
                # bar chart of the top-10 scores of the two main axes
                fig, axes = plt.subplots(1, min(2, len(names)) or 1, figsize=(11, 3.8), squeeze=False)
                for ax, name in zip(axes[0], [n for n in ["lang_dom", "ent"] if n in names] or names[:2]):
                    sub = un[(un["direction"] == name)].sort_values(["kind", "rank"])
                    pro = sub[sub["kind"] == "promoted"].head(10)
                    sup = sub[sub["kind"] == "suppressed"].head(10)
                    y = np.arange(10)
                    ax.barh(y + 0.2, pro["score"].to_numpy(), 0.4, color="C3", label="befördert")
                    ax.barh(y - 0.2, sup["score"].to_numpy(), 0.4, color="C0", label="unterdrückt")
                    ax.set_yticks(y)
                    ax.set_yticklabels([f"{repr(p)} / {repr(s)}" for p, s in zip(pro["token"], sup["token"])], fontsize=7)
                    ax.invert_yaxis()
                    ax.set_title(f"{name}: Unembedding-Score (LN-Gain · W_U)")
                    ax.legend(fontsize=7)
                savefig("unembed_top_tokens")
                plt.show()
            """
        ),
        code(
            """
            # --- P_freq vs. Entropie pro Token, gefärbt nach K -------------------------------------
            # Operationalised core of the "frequent-token suppression under uncertainty" idea
            # (DESIGN §9): mass on the 100 most frequent Pile tokens against the next-token entropy.
            with guarded("P_freq vs. Entropie"):
                tok = table("B", "pfreq_entropy_tokens")
                tok = tok[np.isfinite(tok["p_freq"]) & np.isfinite(tok["entropy"])]
                if len(tok) > 8000:
                    tok = tok.sample(8000, random_state=SEED)
                fig, ax = plt.subplots(figsize=(7.5, 5))
                for K, g in sorted(tok.groupby("K")):
                    r = np.corrcoef(g["entropy"], g["p_freq"])[0, 1] if len(g) > 2 else float("nan")
                    ax.scatter(g["entropy"], g["p_freq"], s=6, alpha=0.45, label=f"K={K} (n={len(g)}, r={r:+.2f})")
                ax.set_xlabel("Entropie der Nächstes-Token-Verteilung [nat]")
                ax.set_ylabel("P_freq: Masse auf den 100 häufigsten Tokens")
                ax.set_title("Unterdrückung häufiger Tokens bei Unsicherheit? (pro Position)")
                ax.legend(fontsize=8)
                savefig("pfreq_vs_entropy")
                plt.show()
                agg = table("B", "pfreq_entropy_by_class")
                if len(agg):
                    print("Aggregiert nach (K, C, Tokenklasse, Tail):")
                    show(agg.sort_values(["K", "C", "is_tail", "klass"]), n=60)
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 02 switching grid (stage C)
# ======================================================================================
def nb_02() -> list[tuple[str, str]]:
    return [
        header_cell(
            "02_switching_grid", "C", "Sprachwechsel-Raster und Counterfactual-Attribution",
            purpose=("Dossier-Kernkontraste auf dem Raster K × C: Texteffekt T(K, C) = ln p₀(gemischt) − ln p₀(K=0), "
                     "Interventionsnutzen B(x) = ln p_δ − ln p₀ als Dosiskurve pro Richtung (ent, lang_dom, freq, "
                     "Zufallsrichtungen, jeweils ±) und Wechselwirkung I = B(gemischt) − B(K=0); dazu die "
                     "Counterfactual-Attribution nach Im et al.: Patching einzelner Tokenklassen (Whitespace, "
                     "Satzzeichen, Grenz-Tokens, …) des gemeinsamen englischen Schlussabschnitts von x_mono in x_mixed "
                     "mit Zufallspositions-Null, sowie Gradient × Aktivierung."),
            inputs="Stimulusbank, Richtungsbank (aus dem Stufe-B-Lauf auf Drive, sonst neu gebaut).",
            outputs=("Tabellen `grid`, `grid_summary`, `contrasts`, `contrasts_boot`, `attribution`, "
                     "`attribution_summary`, `grad_x_act`, `grad_x_act_by_class`."),
            steps="Setup → Drive → Konfiguration → Modell → Stufe C → T(K,C) mit Familien-Bootstrap → B(x)-Dosiskurven → Attribution nach Tokenklasse vs. Zufall → Zusammenfassung.",
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("02_switching_grid", "C"),
        model_cell(),
        runner_cell(),
        code(
            """
            # --- Stufe C ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage C --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT>
            res_C = run_stage("C")
            print({k: len(v) for k, v in res_C.items()})
            """
        ),
        code(
            """
            # --- T(K, C): Texteffekt mit Familien-Cluster-Bootstrap -------------------------------
            # Families are the statistical unit (DESIGN §9): resample whole families with replacement.
            with guarded("T(K, C) mit Familien-Bootstrap"):
                from lsne.stats import cluster_bootstrap

                con = table("C", "contrasts")
                std = con[con["kind"] == "standard"].drop_duplicates("tag")     # T does not depend on direction / dose
                rows = []
                for (K, Cc), g in std.groupby(["K", "C"], sort=True):
                    r = cluster_bootstrap(g, "T", "family", n_boot=limits.n_boot, seed=SEED)
                    rows.append(dict(K=int(K), C=Cc, T=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                                     n_families=r["n_clusters"], n_variants=r["n_rows"]))
                tb = pd.DataFrame(rows)
                show(tb)
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(len(tb))
                colors = {"C1": "C0", "C2": "C1", "C3": "C2", "seg": "C7"}
                ax.bar(x, tb["T"], yerr=[(tb["T"] - tb["ci_low"]).clip(lower=0), (tb["ci_high"] - tb["T"]).clip(lower=0)],
                       capsize=4, color=[colors.get(c, "C4") for c in tb["C"]])
                ax.set_xticks(x)
                ax.set_xticklabels([f"K={k}\\n{c}" for k, c in zip(tb["K"], tb["C"])])
                ax.axhline(0, color="k", lw=0.8)
                ax.set_ylabel("T = ln p₀(gemischt) − ln p₀(K=0)  [nat]")
                ax.set_title(f"Texteffekt T(K, C), 95 %-CI aus {limits.n_boot} Familien-Bootstraps")
                savefig("T_by_K_C")
                plt.show()
                boot = table("C", "contrasts_boot")
                if len(boot):
                    print("Vorab berechnete Kontraste (Stufe C, inkl. I pro Richtung/Dosis):")
                    show(boot[boot["quantity"] == "T"])
            """
        ),
        code(
            """
            # --- B(x): Dosiskurven pro Richtung (Mittel über Varianten, Linien pro K) --------------
            with guarded("B(x) Dosiskurven"):
                gs = table("C", "grid_summary")
                gs = gs[gs["kind"] == "standard"] if "kind" in gs else gs
                fams = [f for f in ["ent", "lang_dom", "freq", "rand"] if f in set(gs["dir_family"])]
                fig, axes = plt.subplots(1, len(fams), figsize=(4.3 * len(fams), 3.9), sharey=True, squeeze=False)
                for ax, fam in zip(axes[0], fams):
                    sub = gs[gs["dir_family"] == fam]
                    for i, (K, g) in enumerate(sorted(sub.groupby("K"))):
                        for sign, ls in ((1, "-"), (-1, "--")):
                            m = g[g["sign"] == sign].groupby("eps")["B"].mean()
                            ax.plot(m.index, m.values, ls, color=f"C{i}", marker="o", ms=3,
                                    label=f"K={K} ({'+' if sign > 0 else '−'})")
                    ax.set_xscale("symlog", linthresh=1e-3)
                    ax.axhline(0, color="k", lw=0.8)
                    ax.set_title(f"Richtung {fam}" + (" (Mittel über Zufallsrichtungen)" if fam == "rand" else ""), fontsize=9)
                    ax.set_xlabel("Dosis ε (relativ zur RMS des Residuums)")
                axes[0, 0].set_ylabel("B = ln p_δ(Ziel) − ln p₀(Ziel)")
                axes[0, -1].legend(fontsize=7, ncol=2)
                savefig("B_dose_curves")
                plt.show()
                # interaction I = B(mixed) - B(K=0) at the dossier dose 0.01
                con = table("C", "contrasts")
                sub = con[(con["kind"] == "standard") & (con["eps"] == 0.01) & (con["sign"] == 1)]
                if len(sub):
                    print("I = B(gemischt) − B(K=0) bei ε = 0.01, Vorzeichen +, Mittel über Familien:")
                    show(sub.groupby(["dir_family", "K", "C"])["I"].agg(["mean", "std", "count"]))
            """
        ),
        code(
            """
            # --- Counterfactual-Attribution: Wiederherstellung R nach Tokenklasse vs. Zufall ---------
            # R_c = (ln p_patched - ln p_mixed) / (ln p_mono - ln p_mixed); random control = the same
            # number of random tail positions (Im et al. 2026: delimiters carry a disproportionate share).
            with guarded("Attribution nach Tokenklasse"):
                att = table("C", "attribution_summary")
                for pair, g in att.groupby("pair"):
                    layers = sorted(g["layer"].unique())
                    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.8), sharey=True, squeeze=False)
                    for ax, l in zip(axes[0], layers):
                        s = g[g["layer"] == l].set_index("class").reindex(list(C.PATCH_CLASSES))
                        x = np.arange(len(s))
                        ax.bar(x - 0.2, s["recovery_mean"], 0.4, yerr=s["recovery_sd"].fillna(0), capsize=2, label="Tokenklasse")
                        ax.bar(x + 0.2, s["recovery_random_mean"], 0.4, color="grey", label="Zufallspositionen (gleiche Anzahl)")
                        ax.set_xticks(x)
                        ax.set_xticklabels(s.index, rotation=70, fontsize=8)
                        ax.axhline(0, color="k", lw=0.8)
                        ax.axhline(1, color="k", lw=0.5, ls=":")
                        ax.set_title(f"{pair}: Block-Eingang {l}", fontsize=9)
                    axes[0, 0].set_ylabel("Wiederherstellung R")
                    axes[0, 0].legend(fontsize=7)
                    savefig(f"attribution_{pair}")
                    plt.show()
                if len(att):
                    print("Überschuss über Zufall (recovery_mean − recovery_random_mean):")
                    show(att.pivot_table(index=["pair", "class"], columns="layer", values="excess_over_random"), n=60)

            with guarded("Gradient × Aktivierung: Anteil pro Tokenklasse"):
                gx = table("C", "grad_x_act_by_class")
                std = gx[gx["kind"] == "standard"] if "kind" in gx else gx
                piv = std.groupby(["layer", "class"])["abs_share"].mean().unstack("class")
                ax = piv.plot(kind="bar", figsize=(9, 3.8), width=0.8)
                ax.set_ylabel("mittlerer Anteil an Σ|grad × act|")
                ax.set_xlabel("Block-Eingang")
                ax.set_title("Gradient × Aktivierung von ln p(Ziel) nach Tokenklasse")
                ax.legend(fontsize=7, ncol=4)
                savefig("grad_x_act_share")
                plt.show()
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 03 non-equilibrium (stage D)
# ======================================================================================
def nb_03() -> list[tuple[str, str]]:
    return [
        header_cell(
            "03_nonequilibrium", "D", "Nicht-Gleichgewichts-Diagnostiken",
            purpose=("Die Gitterdiagnostiken D1–D9 aus `docs/DESIGN.md` §4.2: Störungsausbreitung (Damage-Maps, "
                     "Wachstumsexponent λ als präregistrierter primärer Endpunkt), Logit-Lens-Nichtmonotonie, "
                     "Hysterese unter Dosisrampe mit Neustart-Null, Relaxation nach Dosissprung (τ(ε)), "
                     "Wechselstatistik, Suszeptibilität, Fluktuation–Antwort und die CA-Komplexität des "
                     "Symbolfelds N/E/D. Phasendiagramme über (ε, K) pro Richtung."),
            inputs="Stimulusbank, Richtungsbank (Stufe B auf Drive oder neu), Generierungsbudget `cfg.gen_n_new` × `cfg.gen_n_samples`.",
            outputs=("Tabellen `damage`, `damage_profile`, `damage_phase`, `lens`, `lens_summary`, `update_norms`, "
                     "`ca`, `ca_layers`, `hysteresis`, `hysteresis_curves`, `relaxation`, `relaxation_curves`, "
                     "`switch_stats`, `susceptibility`, `fdt`."),
            steps=("Setup → Drive → Konfiguration → Modell → Stufe D → Damage-Maps K=0/8/28 → λ(ε, K)-Phasendiagramm → "
                   "Lens-Nichtmonotonie → Hysterese → Relaxation τ(ε) → CA-Diagramme → Suszeptibilität/FDT → Zusammenfassung."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("03_nonequilibrium", "D"),
        model_cell(),
        runner_cell(),
        code(
            """
            # --- Stufe D ausführen (längste Stufe: Generierung D3-D7) ------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage D --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> [--gen-n-new 64 --gen-n-samples 8]
            res_D = run_stage("D")
            print({k: len(v) for k, v in res_D.items()})
            """
        ),
        code(
            """
            # --- Kernvarianten (K=0, K=8/C1, K=28/seg der ersten Formulierung) und Richtungsbank ----
            with guarded("Kernvarianten und Richtungsbank"):
                variants, _checks = C.get_variants(cfg, mb, ctx)
                original = C.get_original(cfg, mb, ctx, limits)
                site = C.dossier_site(mb)
                bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original)
                std = [v for v in variants if not C.is_original(v)]
                fam0, form0 = (std[0].family_id, std[0].formulation) if std else (None, None)
                picks = []
                for (K, Cc) in C.CORE_CONDITIONS:
                    m = [v for v in std if v.family_id == fam0 and v.formulation == form0 and (int(v.K), str(v.C)) == (K, Cc)]
                    if m:
                        picks.append((K, Cc, m[0]))
                DIR_MAP = "lang_dom" if "lang_dom" in bank else list(bank.keys())[0]
                print("Kernvarianten:", [v.tag for _, _, v in picks], "| Richtung:", DIR_MAP, "| Bankquelle:", bank.meta.get("source"))
            """
        ),
        code(
            """
            # --- D1: Damage-Maps (Layer × Position) für K=0 vs K=8 vs K=28 ------------------------
            # Injection at the depth site (block input n_layers//3) at the last token of unit 9,
            # D(l, t) = ||h' - h|| / ||h||; the dashed line marks the shared English tail.
            with guarded("Damage-Maps"):
                from lsne.dynamics import damage_spreading, growth_exponent

                EPS_MAP = float(max(limits.damage_doses))
                fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4.2), squeeze=False)
                im = None
                for ax, (K, Cc, v) in zip(axes[0], picks):
                    pos = C.unit_last_position(v)
                    dm = damage_spreading(mb, C.ids_tensor(mb, v), C.depth_site(mb), pos, bank[DIR_MAP].vec, EPS_MAP,
                                          target_id=int(v.target_id))
                    lam = growth_exponent(dm, [p for p in C.tail_positions(v) if p >= pos] or [v.n_tokens - 1])["lam"]
                    im = ax.imshow(np.log10(np.clip(dm.D, 1e-6, None)), aspect="auto", origin="lower", cmap="magma", vmin=-6, vmax=0,
                                   interpolation="nearest")
                    ax.axvline(v.tail_start - 0.5, color="cyan", lw=0.9, ls="--")
                    ax.plot([pos], [dm.inject_layer], marker="^", color="white", ms=8)
                    ax.set_xlim(max(-0.5, pos - 24.5), v.n_tokens - 0.5)      # sites before t0 are unperturbed (causality)
                    ax.set_title(f"K={K} {Cc} ({v.tag})\\nInjektion @ t={pos}, λ={lam:+.3f}", fontsize=9)
                    ax.set_xlabel("Position t")
                axes[0, 0].set_ylabel("Snapshot ℓ (Block-Eingänge; oberster = Ausgang)")
                fig.suptitle(f"Störungsausbreitung D(ℓ, t): Richtung {DIR_MAP}, ε={EPS_MAP:g}, Injektion {C.depth_site(mb).key} "
                             "(gestrichelt: Beginn des englischen Schlussabschnitts)", fontsize=10, y=1.04)
                if im is not None:
                    fig.colorbar(im, ax=axes[0].tolist(), label="log10 D(ℓ, t)", shrink=0.9)
                savefig("damage_maps")
                plt.show()
            """
        ),
        code(
            """
            # --- D1: Phasendiagramm λ(ε, K) pro Richtung (Injektion 'depth', Mittel über C) ---------
            with guarded("Phasendiagramm λ(ε, K)"):
                ph = table("D", "damage_phase")
                ph = ph[(ph["inject"] == "depth") & (ph["kind"] == "standard")]
                fams = [f for f in ["lang_dom", "ent", "rand", "sae"] if f in set(ph["dir_family"])]
                vmax = float(np.nanmax(np.abs(ph["lam_mean"]))) if len(ph) else 1.0
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                fig, axes = plt.subplots(1, len(fams), figsize=(4.6 * len(fams), 3.9), squeeze=False)
                im = None
                for ax, fam in zip(axes[0], fams):
                    piv = ph[ph["dir_family"] == fam].groupby(["K", "eps"])["lam_mean"].mean().unstack("eps")
                    im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                    ax.set_xticks(range(piv.shape[1]))
                    ax.set_xticklabels([f"{e:g}" for e in piv.columns])
                    ax.set_yticks(range(piv.shape[0]))
                    ax.set_yticklabels([str(k) for k in piv.index])
                    for i in range(piv.shape[0]):
                        for j in range(piv.shape[1]):
                            ax.text(j, i, f"{piv.iat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
                    ax.set_title(f"λ — Richtung {fam}", fontsize=10)
                    ax.set_xlabel("ε")
                    ax.set_ylabel("K")
                if im is not None:
                    fig.colorbar(im, ax=axes[0].tolist(), label="λ = Steigung von ln D über ℓ (λ > 0: Wachstum)", shrink=0.9)
                savefig("lambda_phase_eps_K")
                plt.show()
                # structure level at fixed K: lambda over (eps, C)
                k_max = int(ph[ph["C"] != "seg"]["K"].max()) if len(ph[ph["C"] != "seg"]) else None
                if k_max:
                    sub = ph[(ph["K"] == k_max) & (ph["dir_family"] == fams[0])]
                    print(f"λ(ε, C) bei K={k_max}, Richtung {fams[0]}:")
                    show(sub.pivot_table(index="C", columns="eps", values="lam_mean"))
                print("Primärer Endpunkt (Präregistrierung): Δλ = λ(K>0, lang_dom) − λ(K=0, lang_dom) vs. derselbe Kontrast mit rand;")
                print("das Familien-Bootstrap-Intervall wird aus der Tabelle `damage` (eine Zeile pro Variante) gebildet:")
                dmg = table("D", "damage")
                if len(dmg):
                    from lsne.stats import cluster_bootstrap
                    d = dmg[(dmg["inject"] == "depth") & (dmg["kind"] == "standard")]
                    e_max = float(d["eps"].max())
                    rows = []
                    for fam in [f for f in ["lang_dom", "rand"] if f in set(d["dir_family"])]:
                        dd = d[(d["dir_family"] == fam) & (d["eps"] == e_max)]
                        ref = dd[dd["K"] == 0].groupby("family")["lam"].mean()
                        for K, g in dd[dd["K"] > 0].groupby("K"):
                            diff = g.groupby("family")["lam"].mean()
                            both = pd.DataFrame(dict(dlam=diff - ref.reindex(diff.index))).dropna().reset_index()
                            r = cluster_bootstrap(both, "dlam", "family", n_boot=limits.n_boot, seed=SEED)
                            rows.append(dict(dir_family=fam, K=int(K), eps=e_max, dlam=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], n_families=r["n_clusters"]))
                    show(pd.DataFrame(rows))
            """
        ),
        code(
            """
            # --- D2: Logit-Lens — KL zur Endvorhersage über die Tiefe und Nichtmonotonie-Index ------
            with guarded("Logit-Lens Nichtmonotonie"):
                lens = table("D", "lens")
                ls = table("D", "lens_summary")
                lens = lens[lens["C"] != "orig"]
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
                for K, g in sorted(lens.groupby("K")):
                    m = g.groupby("layer")["kl_last"].mean()
                    a1.plot(m.index, m.values, marker="o", ms=3, label=f"K={K}")
                a1.set_yscale("log")
                a1.set_xlabel("Snapshot ℓ")
                a1.set_ylabel("KL(p_final ‖ p_ℓ) [nat], letzte Position")
                a1.set_title("Konvergenz der Zwischenschicht-Vorhersage")
                a1.legend(fontsize=8)
                lss = ls[ls["kind"] == "standard"] if "kind" in ls else ls
                rng = np.random.default_rng(SEED)
                for i, (K, g) in enumerate(sorted(lss.groupby("K"))):
                    a2.scatter(np.full(len(g), float(K)) + rng.uniform(-0.3, 0.3, len(g)), g["nmi_last"], s=14, alpha=0.6, color=f"C{i}")
                    a2.plot([K - 0.4, K + 0.4], [g["nmi_last"].mean()] * 2, color=f"C{i}", lw=2)
                a2.set_xlabel("K")
                a2.set_ylabel("Nichtmonotonie-Index (letzte Position)")
                a2.set_title("Σ positiver KL-Zuwächse / Gesamtvariation (0 = monoton)")
                savefig("lens_nonmonotonicity")
                plt.show()
                if len(lss):
                    show(lss.groupby(["K", "C"])[["nmi_last", "nmi_tail", "kl_last_max", "layer_of_max_kl_last", "first_layer_target_top10"]].mean())
            """
        ),
        code(
            """
            # --- D3: Hysterese — Aufwärts- vs. Abwärtsast der Dosisrampe mit Neustart-Null ---------
            with guarded("Hysterese"):
                hc = table("D", "hysteresis_curves")
                hy = table("D", "hysteresis")
                DIR_GEN = "lang_dom" if "lang_dom" in set(hc["direction"]) else str(hc["direction"].iloc[0])
                sub = hc[hc["direction"] == DIR_GEN]
                tags = list(dict.fromkeys(sub["tag"]))[:6]
                fig, axes = plt.subplots(1, len(tags), figsize=(4 * len(tags), 3.8), sharey=True, squeeze=False)
                for ax, tag in zip(axes[0], tags):
                    g = sub[sub["tag"] == tag]
                    for br, st, lab in (("up", "o-", "aufwärts 0→ε_max"), ("down", "s-", "abwärts ε_max→0"),
                                        ("restart_a", "^:", "Neustart a"), ("restart_b", "v:", "Neustart b")):
                        b = g[g["branch"] == br].sort_values("level")
                        ax.plot(b["eps"], b["fD"], st, ms=4, label=lab)
                    a = hy[(hy["tag"] == tag) & (hy["direction"] == DIR_GEN)]
                    if len(a):
                        ax.set_title(f"{tag}\\nA_hyst={a['area'].iloc[0]:+.3f}  Null={a['null_area'].iloc[0]:+.3f}", fontsize=9)
                    ax.set_xlabel(f"Dosis ε ({DIR_GEN})")
                axes[0, 0].set_ylabel("f_D: Anteil deutscher Tokens pro Plateau")
                axes[0, 0].legend(fontsize=7)
                savefig("hysteresis_curves")
                plt.show()
                if len(hy):
                    show(hy[["tag", "K", "C", "direction", "area", "null_area", "area_minus_null", "up_mean", "down_mean", "n_switches_total"]])
            """
        ),
        code(
            """
            # --- D4: Relaxation nach Dosissprung — τ(ε) und f_D(t) ---------------------------------
            with guarded("Relaxation τ(ε)"):
                rx = table("D", "relaxation")
                rc = table("D", "relaxation_curves")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
                for (tag, d), g in rx.groupby(["tag", "direction"]):
                    g = g.sort_values("eps")
                    yerr = [(g["tau"] - g["tau_lo"]).clip(lower=0).fillna(0), (g["tau_hi"] - g["tau"]).clip(lower=0).fillna(0)]
                    a1.errorbar(g["eps"], g["tau"], yerr=yerr, marker="o", capsize=2, ls="-" if d == "lang_dom" else "--", label=f"{tag} · {d}")
                a1.set_xlabel("ε (Quench bei t_on)")
                a1.set_ylabel("τ [Tokens] (Exponentialfit, Bootstrap-CI)")
                a1.set_yscale("log")
                a1.set_title("Kritische Verlangsamung? τ sollte im Gleichgewicht dosisunabhängig sein")
                a1.legend(fontsize=6)
                DIR_GEN = "lang_dom" if "lang_dom" in set(rc["direction"]) else str(rc["direction"].iloc[0])
                for (tag, eps), g in rc[rc["direction"] == DIR_GEN].groupby(["tag", "eps"]):
                    a2.plot(g["t"], g["fD_t"], label=f"{tag} ε={eps:g}")
                a2.axvline(limits.gen_t_on, color="k", ls="--", lw=0.8)
                a2.set_xlabel("Generierungsschritt t")
                a2.set_ylabel("f_D(t) (Fenster 8)")
                a2.set_title(f"Antwort auf den Dosissprung ({DIR_GEN})")
                a2.legend(fontsize=6)
                savefig("relaxation_tau")
                plt.show()
            """
        ),
        code(
            """
            # --- D9: CA-Diagramme — Symbolfeld S(ℓ, t) ∈ {N, E, D} aus der Logit-Lens-Sprachmasse ----
            with guarded("CA-Diagramme"):
                from matplotlib.colors import ListedColormap
                from lsne.dynamics import ca_symbolize
                from lsne.model import run_with_cache

                lex = C.marker_lexicon(mb.tokenizer)
                cmap = ListedColormap(["#d9d9d9", "#1f77b4", "#d62728"])
                fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.4 * len(picks)), squeeze=False)
                for ax, (K, Cc, v) in zip(axes[:, 0], picks):
                    cache = run_with_cache(mb, C.ids_tensor(mb, v), "all_block_inputs")
                    S = ca_symbolize(mb, cache, lex, lang_axis=bank.get("lang_dom"), thresholds=(limits.ca_threshold,))
                    ax.imshow(S, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
                    # input language of every token as a strip above the field
                    lang = np.array([2 if l == "D" else 1 for l in v.annot["lang"]])[None, :]
                    ax.imshow(lang, aspect="auto", extent=(-0.5, v.n_tokens - 0.5, S.shape[0] - 0.5, S.shape[0] + 0.5),
                              cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
                    ax.set_ylim(-0.5, S.shape[0] + 0.5)
                    ax.set_xlim(-0.5, v.n_tokens - 0.5)
                    ax.axvline(v.tail_start - 0.5, color="k", lw=0.8, ls="--")
                    ax.set_title(f"K={K} {Cc} ({v.tag}); oberste Zeile = Eingabesprache", fontsize=9)
                    ax.set_ylabel("ℓ")
                axes[-1, 0].set_xlabel("Position t   (grau N, blau E, rot D)")
                savefig("ca_fields")
                plt.show()
                ca = table("D", "ca")
                if len(ca):
                    cols = [c for c in ["tag", "K", "C", "spatial_entropy", "temporal_entropy", "activity", "lz76", "block_entropy_2x2", "frac_D", "frac_D_final_tail"] if c in ca.columns]
                    show(ca[cols])
            """
        ),
        code(
            """
            # --- D5-D7: Wechselstatistik, Suszeptibilität χ(ε), Fluktuation-Antwort T_eff ----------
            with guarded("Suszeptibilität und FDT"):
                su = table("D", "susceptibility")
                fdt = table("D", "fdt")
                sw = table("D", "switch_stats")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
                for (tag, d, op), g in su.groupby(["tag", "direction", "order_parameter"]):
                    if op != "fD":
                        continue
                    g = g.sort_values("eps")
                    a1.plot(g["eps"], g["mean"], marker="o", ls="-" if d == "lang_dom" else "--", label=f"{tag} · {d}")
                    a2.plot(g["eps"], g["chi"], marker="o", ls="-" if d == "lang_dom" else "--", label=f"{tag} · {d}")
                a1.set_xlabel("ε (konstant)")
                a1.set_ylabel("⟨f_D⟩ über Samples")
                a1.set_title("Ordnungsparameter f_D(ε)")
                a2.set_xlabel("ε")
                a2.set_ylabel("χ = d f_D / dε")
                a2.set_title("Suszeptibilität")
                a1.legend(fontsize=6)
                savefig("susceptibility")
                plt.show()
                if len(fdt):
                    cols = [c for c in ["tag", "K", "C", "direction", "var0_fD", "chi_fD_1", "T_eff_fD_1", "var0_proj", "chi_proj_0", "T_eff_proj_0"] if c in fdt.columns]
                    print("Fluktuation-Antwort (explorativ): T_eff = Var₀(m)/χ")
                    show(fdt[cols])
                if len(sw):
                    cols = [c for c in ["tag", "K", "direction", "source", "eps", "n_intervals", "geometric_p", "ks_stat", "tail_index"] if c in sw.columns]
                    print("Wechselstatistik (geometrisch = Markov; schwere Ränder = Lawinen):")
                    show(sw[cols], n=30)
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 04 tensor network (stage E)
# ======================================================================================
def nb_04() -> list[tuple[str, str]]:
    return [
        header_cell(
            "04_tn_linear_response", "E", "Tensornetzwerk und lineare Antwort",
            purpose=("Vergleich dreier Antwortebenen auf eine Störung an einer Gitterstelle (`docs/DESIGN.md` §5, D8): "
                     "(a) gemessene nichtlineare Störung, (b) exakte lineare Antwort per Vorwärts-Autodiff (JVP), "
                     "(c) faktorisiertes Tensornetzwerk des linearisierten Transformers (eingefrorene Attention-Muster und "
                     "LN-Skalen, PCA-Unterraum r, Positionsfenster) mit einer von TNCO optimierten Kontraktionsreihenfolge "
                     "(Fallback opt_einsum). (a)−(b) misst Nichtlinearität, (b)−(c) den Fehler der Faktorisierung. "
                     "TNCO ist hier ein Optimierer für Kontraktionspfade, kein Interpretierbarkeitsverfahren "
                     "(siehe `docs/TNCO_INTEGRATION.md`)."),
            inputs="K=0- und K=8/C1-Varianten der ersten `limits.n_families_tn` Familien, Richtungsbank (lang_dom, ent), `cfg.tn_rank`, `cfg.tn_window`.",
            outputs="Tabellen `tn_compare`, `tn_summary`, `tn_paths`, `tn_basis`.",
            steps="Setup → TNCO (optional) → Drive → Konfiguration → Modell → Stufe E → Antwortnormen pro Position → Nichtlinearität/Faktorisierungsfehler → TNCO-vs-opt_einsum-Tabelle → PCA-Basis → Zusammenfassung.",
        ),
        setup_cell(),
        tnco_cell(True),
        drive_cell(),
        config_cell("04_tn_linear_response", "E",
                    extra_lines=("TN_RANK = None                     # PCA-Rang r des Residualraums (Standard cfg.tn_rank = 64; A100: bis d_model)\n"
                                 "TN_WINDOW = None                   # Positionsfenster T_w (Standard 32)")),
        code(
            """
            # --- Stufe-E-Parameter ----------------------------------------------------------------
            if TN_RANK is not None:
                cfg.tn_rank = int(TN_RANK)
            if TN_WINDOW is not None:
                cfg.tn_window = int(TN_WINDOW)
            print(f"Rang r = {cfg.tn_rank}, Fenster T_w = {cfg.tn_window}, Familien für Stufe E: {limits.n_families_tn}, "
                  f"TNCO-Schritte: {limits.tn_n_steps} × {limits.tn_n_runs} Läufe, Dosen {limits.tn_doses}")
            """
        ),
        model_cell(),
        runner_cell(),
        code(
            """
            # --- Stufe E ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage E --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> [--tn-rank 64 --tn-window 32]
            res_E = run_stage("E")
            print({k: len(v) for k, v in res_E.items()})
            """
        ),
        code(
            """
            # --- Antwortnormen pro Fensterposition: gemessen vs. linear (JVP) vs. faktorisiert (TN) --
            with guarded("Antwortebenen"):
                tc = table("E", "tn_compare")
                combos = list(dict.fromkeys(zip(tc["tag"], tc["direction"])))[:6]
                epss = sorted(tc["eps"].unique())
                fig, axes = plt.subplots(len(combos), len(epss), figsize=(5.2 * len(epss), 3.1 * len(combos)), squeeze=False, sharex=True)
                for i, (tag, d) in enumerate(combos):
                    for j, eps in enumerate(epss):
                        g = tc[(tc["tag"] == tag) & (tc["direction"] == d) & (tc["eps"] == eps)].sort_values("rel_pos")
                        ax = axes[i, j]
                        ax.plot(g["rel_pos"], g["measured_norm"], "o-", ms=3, label="(a) gemessen, nichtlinear")
                        ax.plot(g["rel_pos"], g["linear_norm"], "s--", ms=3, label="(b) exakt linear (JVP)")
                        ax.plot(g["rel_pos"], g["factorized_norm"], "^:", ms=3, label="(c) faktorisiertes TN")
                        ax.set_yscale("log")
                        ax.axvline(0, color="k", lw=0.6, ls=":")
                        ax.set_title(f"{tag} · {d} · ε={eps:g}", fontsize=9)
                        if i == len(combos) - 1:
                            ax.set_xlabel("Position relativ zur Injektion")
                        if j == 0:
                            ax.set_ylabel("‖Antwort‖")
                fig.suptitle("Antwort am Ausgang des letzten Blocks: (a) gemessen, (b) exakt linear, (c) faktorisiertes TN", fontsize=10)
                axes[0, 0].legend(fontsize=7)
                savefig("tn_response_norms")
                plt.show()
            """
        ),
        code(
            """
            # --- Nichtlinearität |a−b|/|b| und Faktorisierungsfehler |b−c|/|b| -----------------------
            with guarded("Nichtlinearität und Faktorisierungsfehler"):
                tc = table("E", "tn_compare")
                ts = table("E", "tn_summary")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
                for (tag, d, eps), g in tc.groupby(["tag", "direction", "eps"]):
                    g = g[g["rel_pos"] >= 0].sort_values("rel_pos")
                    a1.plot(g["rel_pos"], g["nonlinearity"], marker=".", label=f"{tag} {d} ε={eps:g}")
                    a2.plot(g["rel_pos"], g["factorization_error"], marker=".", label=f"{tag} {d} ε={eps:g}")
                for ax, t in ((a1, "Nichtlinearität ‖a−b‖/‖b‖ (D8; ∝ ε im linearen Regime)"), (a2, "Faktorisierungsfehler ‖b−c‖/‖b‖")):
                    ax.set_yscale("log")
                    ax.set_xlabel("Position relativ zur Injektion")
                    ax.set_title(t, fontsize=10)
                a1.legend(fontsize=6)
                savefig("tn_errors")
                plt.show()
                if len(ts):
                    cols = [c for c in ["tag", "direction", "eps", "r", "T_w", "source_captured", "nonlinearity_wmean", "factorization_error_wmean",
                                        "nonlinearity_last", "factorization_error_last", "cos_measured_linear_last", "cos_linear_factorized_last", "path_backend"] if c in ts.columns]
                    show(ts[cols])
            """
        ),
        code(
            """
            # --- Kontraktionspfade: TNCO vs. opt_einsum (Kosten log10 FLOPs, Laufzeit) --------------
            with guarded("TNCO vs. opt_einsum"):
                tp = table("E", "tn_paths")
                agg = tp.groupby("backend").agg(backend_used=("backend_used", "first"), n_tensors=("n_tensors", "first"),
                                                T_w=("T_w", "first"), r=("r", "first"), n_blocks=("n_blocks", "first"),
                                                cost_log10_opt_einsum=("cost_log10_opt_einsum", "mean"),
                                                cost_log10_backend=("cost_log10", "mean"), runtime_s=("runtime_s", "mean"),
                                                largest_intermediate_log10=("largest_intermediate_log10", "mean"),
                                                fallback_reason=("fallback_reason", "first"), nbytes=("nbytes", "first"))
                order = [b for b in ["tnco", "opt_einsum", "reference_greedy", "reference_dp"] if b in agg.index]
                agg = agg.reindex(order)
                show(agg)
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
                a1.bar(order, agg["cost_log10_opt_einsum"], color=["C3" if b == "tnco" else "C0" for b in order])
                a1.set_ylabel("log10 FLOPs (opt_einsum-Zählung, pro Pfad)")
                a1.set_title("Kontraktionskosten des besten Pfads")
                a2.bar(order, agg["runtime_s"], color=["C3" if b == "tnco" else "C0" for b in order])
                a2.set_ylabel("Optimierer-Laufzeit [s]")
                a2.set_yscale("log")
                a2.set_title("Laufzeit der Pfadsuche")
                for ax in (a1, a2):
                    ax.tick_params(axis="x", rotation=20)
                savefig("tn_paths")
                plt.show()
                if "tnco" in agg.index and str(agg.loc["tnco", "fallback_reason"]):
                    print("Hinweis: TNCO nicht verwendet ->", agg.loc["tnco", "fallback_reason"])
            """
        ),
        code(
            """
            # --- PCA-Basis: kumulierte erklärte Varianz des Residualraums an der Injektionsstelle ------
            with guarded("PCA-Basis"):
                tb = table("E", "tn_basis")
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.plot(tb["rank"], tb["explained_variance_cum"], "o-", ms=3)
                ax.set_xlabel("Rang r")
                ax.set_ylabel("kumulierte erklärte Varianz")
                ax.set_title(f"Trunkierung auf r={cfg.tn_rank} von d={mb.d_model}")
                ax.axhline(1.0, color="k", lw=0.5, ls=":")
                savefig("tn_basis")
                plt.show()
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 05 scale and checkpoints (stage F)
# ======================================================================================
def nb_05() -> list[tuple[str, str]]:
    return [
        header_cell(
            "05_scale_and_checkpoints", "F", "Skalierung und Checkpoint-Sweep",
            purpose=("Führt eine Shard aus `jobs/shards.yaml` aus: entweder die Stufen A–E eines Modells/Checkpoints "
                     "(Shards 1, 2, 3, 5) oder den Stufe-F-Sweep über (Modell, Revision)-Paare (Shard 4: 27 Dossier-"
                     "Checkpoints 4k…143k von pythia-160m mit B+D; Shard 6: pythia-1b und pythia-1.4b mit B/C/D, nur C1). "
                     "Jedes Paar wird mit frisch geladenem Modell gerechnet, die Ergebnisse landen pro Stufe sofort auf "
                     "Drive (inkrementeller Export); ein Wiederaufnahme-Schutz überspringt Paare/Stufen, deren "
                     "`manifest.json` bereits `status: ok` meldet."),
            inputs="`jobs/shards.yaml` (SHARD = 1…6), Stimulusbank; optional eigene Paare über `FORCE_PAIRS`.",
            outputs=("Laufverzeichnisse `<RUNS_ROOT>/<stufe>/<modell>@<rev>/` (Shards 1–3, 5) bzw. "
                     "`<RUNS_ROOT>/F/<modell>@<rev>/shard<i>of<n>/<stufe>/<modell>@<rev>/` (Shards 4, 6) mit `plan`/`runs`-Tabellen; "
                     "zusammengeführte CSVs unter `.../merged/`; Verlaufsplots über Checkpoints."),
            steps="Setup → Drive → Shard wählen → Plan (Paare × Familien, Shard-Zuordnung) → Schleife mit Wiederaufnahme-Schutz → Merge (`lsne.cli merge`) → Verlauf über Checkpoints → Zusammenfassung.",
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        code(
            """
            # --- Shard wählen (jobs/shards.yaml) ---------------------------------------------------
            NOTEBOOK = "05_scale_and_checkpoints"
            STAGE = "F"
            SHARD = 4                 # 1..6 aus jobs/shards.yaml (4 = Checkpoint-Trace, 6 = 1b/1.4b); 0 = eigene Angaben
            SMOKE = False             # True: pythia-14m, 1 Familie, Stufen B+D limitiert (< 3 min CPU)
            REUSE = True              # Paare/Stufen mit manifest.json (status ok) überspringen
            DTYPE = "float32"
            DEVICE = "auto"
            FAMILIES = None           # z. B. ["F01", "F02"]; None = alle
            SEED = 0
            FORCE_PAIRS = None        # z. B. "step4000,step5000" | "EleutherAI/pythia-410m@step143000" | "trace" | None = laut Shard
            FORCE_F_STAGES = None     # z. B. ["B", "D"]; None = laut Shard
            SHARD_INDEX, SHARD_COUNT = None, None   # Feinaufteilung einer Shard auf mehrere GPUs (None = laut Shard)

            import subprocess
            import sys

            try:
                import yaml
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])
                import yaml

            with open("jobs/shards.yaml", encoding="utf-8") as f:
                PLAN = yaml.safe_load(f)
            shards = {int(s["id"]): s for s in PLAN["shards"]}
            shard = shards.get(int(SHARD), dict(id=0, account="-", gpu="-", purpose="manuell (FORCE_PAIRS / FORCE_F_STAGES)",
                                                model="EleutherAI/pythia-160m", revision="step98000", stages=["A", "B", "C", "D", "E"]))
            print(f"Shard {shard['id']} ({shard.get('account', '?')} / {shard.get('gpu', '?')}): {shard.get('purpose', '')}")
            print("Stufen:", shard.get("stages"), "| F-Stufen:", shard.get("f_stages"), "| C-Stufen:", shard.get("c_levels"), "| SAE:", shard.get("sae"))
            print("CLI-Befehl der Shard:\\n  " + " ".join(str(shard.get("command", "-")).split()))
            print("Alle Shards:")
            for s in PLAN["shards"]:
                print(f"  {s['id']}: {s.get('account')}/{s.get('gpu')} — {s.get('purpose')}")
            """
        ),
        code(
            """
            # --- Plan: Paare (Modell, Revision) × Familien mit deterministischer Shard-Zuordnung ------
            from dataclasses import replace

            import numpy as np
            import pandas as pd

            from lsne.config import ModelSpec, RunConfig
            from lsne.experiments import common as C
            from lsne.experiments.stage_f_scale import family_filter, parse_pairs, plan_triples

            base_model = shard.get("model") or (shard.get("models") or ["EleutherAI/pythia-160m"])[0]
            base_rev = str(shard.get("revision", "step143000"))
            spec = ModelSpec(name=base_model, revision=base_rev, dtype=DTYPE, device=DEVICE)
            cfg = RunConfig(model=spec, families=FAMILIES, seed=SEED, out_root=RUNS_ROOT)
            if shard.get("c_levels"):
                cfg.c_levels = tuple(str(c) for c in shard["c_levels"])
            limits = C.Limits()
            limits.use_sae = bool(shard.get("sae"))
            IS_F = "F" in list(shard.get("stages") or [])
            if IS_F:
                if shard.get("revisions") == "config.CHECKPOINT_TRACE":
                    pairs = parse_pairs("trace", cfg)
                elif shard.get("models"):
                    pairs = [(str(m), base_rev) for m in shard["models"]]
                else:
                    pairs = parse_pairs(None, cfg)
                f_stages = [str(s) for s in (shard.get("f_stages") or limits.f_stages)]
                shard_index, shard_count = int(shard.get("shard_index", 0)), int(shard.get("shard_count", 1))
            else:
                pairs = [(base_model, base_rev)]
                f_stages = [str(s) for s in (shard.get("stages") or ["A", "B", "C", "D", "E"])]
                shard_index, shard_count = 0, 1
            if FORCE_PAIRS:
                pairs = parse_pairs(FORCE_PAIRS, cfg)
                IS_F = True
            if FORCE_F_STAGES:
                f_stages = [str(s).upper() for s in FORCE_F_STAGES]
            if SHARD_INDEX is not None and SHARD_COUNT is not None:
                shard_index, shard_count = int(SHARD_INDEX), int(SHARD_COUNT)
            if SMOKE:
                spec = ModelSpec(name="EleutherAI/pythia-14m", revision="step98000", dtype="float32", device=DEVICE)
                cfg = C.smoke_config(replace(cfg, model=spec, families=FAMILIES or ["DEV01"]))
                limits = C.Limits.smoke()
                pairs, f_stages, IS_F = [(spec.name, spec.revision)], ["B", "D"], True
            # the per-pair stages of the sweep run with the reduced ("limited") settings like CLI stage F
            sub_limits = limits.as_limited() if IS_F else limits
            families = sorted({f.family_id for f in C.select_families(cfg)})
            plan = plan_triples(pairs, families, shard_index, shard_count)
            mine = plan[plan["mine"]]
            CLI_COMMAND = " ".join(str(shard.get("command", "")).split()) or "python -m lsne.cli run ..."
            print(f"{len(pairs)} Paare × {len(families)} Familien = {len(plan)} Tripel, davon {len(mine)} in Shard {shard_index}/{shard_count}; "
                  f"Stufen pro Paar: {f_stages}; Layout: {'F-Sweep' if IS_F else 'Standard'}; SMOKE={SMOKE}")
            print("Paare:", pairs[:8], "..." if len(pairs) > 8 else "")
            """
        ),
        code(
            """
            # --- Schleife über die Paare: frisches Modell je Paar, inkrementeller Drive-Export, Resume --
            import contextlib
            import datetime
            import gc
            import json
            import time
            import traceback
            from pathlib import Path

            import matplotlib.pyplot as plt
            import torch

            from lsne.experiments import STAGES, STAGE_TITLES
            from lsne.io import RunDir, read_manifest, save_df, write_manifest
            from lsne.model import load_model

            FIG_DIR = os.path.join(OUT_ROOT, "figures", NOTEBOOK)
            os.makedirs(FIG_DIR, exist_ok=True)


            def show(df, n: int = 40):
                df = df.head(n) if isinstance(df, pd.DataFrame) else df
                try:
                    display(df)          # noqa: F821
                except NameError:
                    print(df.to_string() if hasattr(df, "to_string") else df)


            def savefig(name: str):
                path = os.path.join(FIG_DIR, f"{name}_shard{shard['id']}.png")
                plt.gcf().savefig(path, dpi=120, bbox_inches="tight")
                print("Abbildung:", path)


            @contextlib.contextmanager
            def guarded(title: str):
                print("==", title)
                try:
                    yield
                except Exception:
                    traceback.print_exc()
                    print(f"[{title}] übersprungen (siehe Traceback)")


            def manifest_ok(rd) -> bool:
                try:
                    return rd.manifest_path.exists() and read_manifest(rd).get("status") == "ok"
                except Exception:
                    return False


            if IS_F:
                f_dir = RunDir(cfg.out_root, "F", cfg.model)                      # same layout as `lsne.cli run --stage F`
                sub_root = f_dir.path / f"shard{shard_index}of{shard_count}"
            else:
                f_dir, sub_root = None, Path(cfg.out_root)
            progress_path = os.path.join(OUT_ROOT, f"progress_{NOTEBOOK}_shard{shard['id']}.json")
            rows = []
            t_all = time.perf_counter()


            def flush_progress():
                # the stage tables are already on Drive (finish_stage writes them per stage); this
                # small file records what finished so that a new session can show the state at once
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(dict(shard=shard["id"], pairs=pairs, stages=f_stages, rows=rows,
                                   timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()), f, indent=1)


            done_pairs = list(dict.fromkeys(zip(mine["model"], mine["revision"])))
            for n_pair, (model, revision) in enumerate(done_pairs, 1):
                fams = sorted(mine[(mine["model"] == model) & (mine["revision"] == revision)]["family"].tolist())
                pspec = replace(cfg.model, name=model, revision=revision)
                sub_cfg = replace(cfg, model=pspec, families=family_filter(cfg.families, fams))
                todo = [st for st in f_stages if not (REUSE and manifest_ok(RunDir(sub_root, st, pspec, mkdir=False)))]
                print(f"\\n[{n_pair}/{len(done_pairs)}] {model}@{revision}: Familien {len(fams)}, Stufen {todo or 'alle fertig (Resume)'}", flush=True)
                for st in f_stages:
                    if st not in todo:
                        rows.append(dict(model=model, revision=revision, stage=st, status="skipped_done", seconds=0.0, n_tables=0,
                                         families=",".join(fams), error="", run_dir=str(RunDir(sub_root, st, pspec, mkdir=False).path)))
                if not todo:
                    continue
                t_load = time.perf_counter()
                try:
                    mb2 = load_model(pspec)
                    rows.append(dict(model=model, revision=revision, stage="load", status="ok", seconds=time.perf_counter() - t_load,
                                     n_tables=0, families=",".join(fams), error="", run_dir=""))
                except Exception as e:                      # e.g. a checkpoint that does not exist
                    traceback.print_exc()
                    rows.append(dict(model=model, revision=revision, stage="load", status="error", seconds=time.perf_counter() - t_load,
                                     n_tables=0, families=",".join(fams), error=repr(e), run_dir=""))
                    flush_progress()
                    continue
                ctx: dict = {}
                for st in todo:
                    rd = RunDir(sub_root, st, pspec)
                    t_st = time.perf_counter()
                    print(f"  Stufe {st} ({STAGE_TITLES[st]}) -> {rd.path}", flush=True)
                    try:
                        tables = STAGES[st].run(sub_cfg, mb2, rd, ctx=ctx, limits=sub_limits)
                        rows.append(dict(model=model, revision=revision, stage=st, status="ok", seconds=time.perf_counter() - t_st,
                                         n_tables=len(tables), families=",".join(fams), error="", run_dir=str(rd.path)))
                        print(f"  Stufe {st} ok in {time.perf_counter() - t_st:.0f} s", flush=True)
                    except Exception as e:
                        traceback.print_exc()
                        C.write_error(rd, st, e, sub_cfg, mb2)
                        rows.append(dict(model=model, revision=revision, stage=st, status="error", seconds=time.perf_counter() - t_st,
                                         n_tables=0, families=",".join(fams), error=repr(e), run_dir=str(rd.path)))
                    flush_progress()
                del mb2, ctx
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            runs = pd.DataFrame(rows, columns=["model", "revision", "stage", "status", "seconds", "n_tables", "families", "error", "run_dir"])
            if f_dir is not None:
                save_df(plan, f_dir, "plan")
                save_df(runs, f_dir, "runs")
                write_manifest(f_dir, stage="F", status="ok", cfg=cfg, model=cfg.model, pairs=pairs, stages=f_stages,
                               shard_index=shard_index, shard_count=shard_count, n_triples=int(len(plan)), n_mine=int(len(mine)),
                               sub_root=str(sub_root), notebook=NOTEBOOK, shard_id=shard["id"],
                               runtime_s=time.perf_counter() - t_all)
            flush_progress()
            print(f"\\nFertig in {time.perf_counter() - t_all:.0f} s; Fortschritt: {progress_path}")
            show(runs, n=200)
            """
        ),
        code(
            """
            # --- Zusammenführen (lsne.cli merge) und Verlauf über Checkpoints / Modelle --------------
            with guarded("Merge und Verlauf"):
                from lsne.cli import merge_runs

                merged_dir = Path(sub_root) / "merged"
                counts = merge_runs(sub_root, merged_dir)
                print({k: v for k, v in sorted(counts.items())})


                def step_of(rev):
                    s = str(rev)
                    return int(s[4:]) if s.startswith("step") and s[4:].isdigit() else np.nan


                def load_merged(name):
                    p = merged_dir / f"{name}.csv"
                    if not p.exists():
                        return pd.DataFrame()
                    df = pd.read_csv(p)
                    df["step"] = df["run_revision"].map(step_of)
                    return df


                n_rev = len(set(r for _, r in pairs))
                if IS_F and n_rev > 1:
                    # checkpoint trace: lambda (depth injection, lang_dom, largest dose) and baseline log p(target)
                    dp = load_merged("damage_phase")
                    base = load_merged("baseline")
                    hy = load_merged("hysteresis")
                    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                    if len(dp):
                        sub = dp[(dp["inject"] == "depth") & (dp["dir_family"] == "lang_dom") & (dp["kind"] == "standard")]
                        sub = sub[sub["eps"] == sub["eps"].max()]
                        for K, g in sorted(sub.groupby("K")):
                            m = g.groupby("step")["lam_mean"].mean().sort_index()
                            axes[0].plot(m.index, m.values, "o-", ms=3, label=f"K={K}")
                        axes[0].set_xscale("log")
                        axes[0].set_xlabel("Trainingsschritt")
                        axes[0].set_ylabel("λ (lang_dom, Injektion depth, ε max)")
                        axes[0].set_title("Störungswachstum über das Training")
                        axes[0].legend(fontsize=7)
                    if len(base):
                        bs = base[base["kind"] == "standard"] if "kind" in base else base
                        for K, g in sorted(bs.groupby("K")):
                            m = g.groupby("step")["logp_target"].mean().sort_index()
                            axes[1].plot(m.index, m.values, "o-", ms=3, label=f"K={K}")
                        axes[1].set_xscale("log")
                        axes[1].set_xlabel("Trainingsschritt")
                        axes[1].set_ylabel("ln p₀(Ziel)")
                        axes[1].set_title("Baseline über das Training")
                        axes[1].legend(fontsize=7)
                    if len(hy):
                        for (K, d), g in sorted(hy.groupby(["K", "direction"])):
                            m = g.groupby("step")["area_minus_null"].mean().sort_index()
                            axes[2].plot(m.index, m.values, "o-", ms=3, label=f"K={K} {d}")
                        axes[2].set_xscale("log")
                        axes[2].axhline(0, color="k", lw=0.6)
                        axes[2].set_xlabel("Trainingsschritt")
                        axes[2].set_ylabel("A_hyst − Null")
                        axes[2].set_title("Hysterese über das Training")
                        axes[2].legend(fontsize=6)
                    savefig("checkpoint_trace")
                    plt.show()
                elif IS_F:
                    # scale arm: compare models side by side on the phase table
                    dp = load_merged("damage_phase")
                    if len(dp):
                        sub = dp[(dp["inject"] == "depth") & (dp["dir_family"] == "lang_dom") & (dp["kind"] == "standard")]
                        show(sub.pivot_table(index=["run_model", "K"], columns="eps", values="lam_mean"), n=60)
                else:
                    print("Stufen A-E eines Modells: Abbildungen in den Notebooks 00-04 (REUSE=True lädt diese Ergebnisse).")
            """
        ),
        code(
            """
            # --- Kurze Laufzusammenfassung nach OUT_ROOT schreiben ---------------------------------
            with guarded("Zusammenfassung schreiben"):
                summary = dict(
                    notebook=NOTEBOOK, stage=STAGE, shard=shard["id"], purpose=shard.get("purpose", ""), smoke=bool(SMOKE),
                    model=spec.name, revision=spec.revision, dtype=spec.dtype, pairs=pairs, stages=f_stages, layout=("F" if IS_F else "standard"),
                    shard_index=shard_index, shard_count=shard_count, families=families, n_triples=int(len(plan)), n_mine=int(len(mine)),
                    out_root=OUT_ROOT, runs_root=RUNS_ROOT, sub_root=str(sub_root), cli_command=CLI_COMMAND,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    runs=runs.to_dict("records"),
                    n_ok=int((runs["status"] == "ok").sum()), n_error=int((runs["status"] == "error").sum()),
                    n_skipped=int((runs["status"] == "skipped_done").sum()),
                    figures=sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png")),
                )
                path = os.path.join(OUT_ROOT, f"summary_{NOTEBOOK}_shard{shard['id']}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(C.to_jsonable(summary), f, indent=1, ensure_ascii=False)
                print("Zusammenfassung:", path)
                print(json.dumps({k: v for k, v in C.to_jsonable(summary).items() if k != "runs"}, indent=1, ensure_ascii=False)[:3000])
            """
        ),
    ]


# ======================================================================================
# 06 local geometry and trajectory (stage G)
# ======================================================================================
def nb_06() -> list[tuple[str, str]]:
    return [
        header_cell(
            "06_geometry_trajectory", "G", "Lokale Geometrie und Trajektorie",
            purpose=("Der Zwischenschritt aus `docs/DESIGN.md` §4.4/§4.5 (Godoy Ortiz 2026, *Meaning Contracts*): "
                     "Realisierungswolken pro Bedingung (Familie, K, C) aus allen Formulierungen × zulässigen "
                     "Einheitsmasken, lokale Geometrie an jedem Residual-Snapshot (Top-k-Deflation, trunkierte "
                     "Mahalanobis-Distanz, Log-Volumen, Participation Ratio, Verschiebung zur K=0-Wolke), G0-Gate "
                     "(Kern-und-Streuung-Ordnung exakter Übersetzungen), G1-Kontraktionskontrast (n-angepasst, "
                     "Familien-Bootstrap) und die Trajektorien T1 (Tiefe: „atmet“ der Raum?), T2 (Position relativ zur "
                     "Wechselstelle), T3 (Steering-Validität: reproduziert `lang_dom` die diskrete K-Verschiebung?) "
                     "und T5 (Kopplung von Δ log-Vol mit λ_ℓ und Lens-KL aus Stufe D)."),
            inputs=("Stimulusbank (alle Formulierungen A–D einer Familie), Richtungsbank (Stufe B auf Drive oder neu); "
                    "für T5 der Stufe-D-Lauf desselben Modells unter `RUNS_ROOT` (Notebook 03)."),
            outputs=("Tabellen `clouds`, `geometry`, `pair_ordering`, `g0_gate`, `depth_trajectory`, `contraction`, "
                     "`contraction_boot`, `position_traj`, `steering`, `steering_boot`, `coupling`."),
            steps=("Setup → Drive → Konfiguration → Modell → Stufe G → G0-Gate → T1 Tiefenkurven → G1 Kontraktion mit "
                   "Bootstrap → T2 Positionsinkremente → T3 Steering-Kosinus → T5 Kopplung → Zusammenfassung."),
            prereg=("Kontrast 7 (G0-Gate: `g0_gate.pass_all` / `pass_middle`; `degenerate = True` heißt: zu wenige "
                    "Zustände für den Rang, Gate nicht aussagekräftig), Kontrast 8 (G1: `contraction_boot`-Zeilen mit "
                    "`preregistered = True`, K=8/C1, Deflation k=5, Mean-Pool für Log-Vol, Last-Token für PR, Vorzeichen "
                    "offen) und Kontrast 9 (T3: `steering_boot`, `quantity = cos_at_max_disp_layer`, ε = 0,05, "
                    "Mindesteffekt cos ≥ 0,3 gegen 32 Zufallsrichtungen). Fällt G0 durch, werden 8–9 nicht interpretiert; "
                    "fällt T3 negativ aus, gelten Dosis-Phasendiagramme nur als Eingriffs-, nicht als Sprachwechselbefunde."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("06_geometry_trajectory", "G"),
        model_cell(),
        runner_cell(),
        substages_cell(),
        code(
            """
            # --- Stufe G ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage G --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT>
            # T5 (coupling) reads damage_profile / lens of the stage-D run of the same model below
            # RUNS_ROOT (empty until notebook 03 has run); the clouds are shared with H/K via ctx.
            print(f"Wolken: K {limits.g_k_levels} × C {limits.g_c_levels}, bis {limits.g_max_masks} Masken je Bedingung, "
                  f"Deflation k {limits.g_deflate_k}, Rang r_mean={limits.g_r_mean} / r_last={limits.g_r_last}, "
                  f"T3-Dosen {limits.g_steer_eps}, n-Matching: {limits.g_n_subsample} Ziehungen")
            res_G = run_stage("G")
            print({k: len(v) for k, v in res_G.items()})
            manifest_G = substage_report("G")
            cl = table("G", "clouds")
            if len(cl):
                print("Realisierungen pro Bedingung (excluded = Konsistenzprüfung nicht bestanden):")
                show(cl.groupby(["group", "excluded"]).size().rename("n").reset_index(), n=60)
            """
        ),
        code(
            """
            # --- G0-Gate: Kern-und-Streuung-Ordnung exakter Übersetzungen (Kontrast 7) ---------------
            # d(A_E, A_D) < d(A_E, B_E) < d(other family) under the truncated Mahalanobis distance of the
            # prefix-mean states (deflation k=5); the gate passes when the order holds in >= 50 % of the
            # snapshots (all / middle third).  `degenerate`: r_eff >= N-1, the distance is uninformative.
            with guarded("G0-Gate"):
                gate = table("G", "g0_gate")
                show(gate)
                if len(gate) and truthy(gate["degenerate"]).iloc[0]:
                    print("ACHTUNG: degenerate = True (r_eff ≥ N−1: zu wenige gepoolte Zustände für den Rang) -> "
                          "Gate nicht aussagekräftig; mehr Familien/Formulierungen nötig (FAMILIES = None).")
                po = table("G", "pair_ordering")
                cats = [c for c in ["exact_translation", "same_meaning_same_lang", "other_family"] if c in set(po["category"])]
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
                for c in cats:
                    g = po[po["category"] == c].sort_values("layer")
                    a1.errorbar(g["layer"], g["mean_mahal"], yerr=g["sd_mahal"].fillna(0), marker="o", ms=4, capsize=2, label=c)
                    a2.plot(g["layer"], g["mean_cosine"], marker="o", ms=4, label=c)
                for _, r in po[truthy(po["core_order_ok"])].drop_duplicates("layer").iterrows():
                    a1.axvspan(r["layer"] - 0.4, r["layer"] + 0.4, color="green", alpha=0.08)
                a1.set_xlabel("Snapshot ℓ")
                a1.set_ylabel("mittlere Mahalanobis-Distanz (trunkiert, k=5)")
                a1.set_title("Kern (exakte Übersetzung) < Streuung (freie Realisierung) < Fremdfamilie?\\n(grün: Ordnung hält)", fontsize=9)
                a1.legend(fontsize=7)
                a2.set_xlabel("Snapshot ℓ")
                a2.set_ylabel("mittlerer Kosinus")
                a2.set_title("Kosinus derselben Paare (Mean-Pool des Präfixes)", fontsize=9)
                a2.legend(fontsize=7)
                savefig("g0_pair_ordering")
                plt.show()
            """
        ),
        code(
            """
            # --- T1: Tiefen-Trajektorie der lokalen Geometrie (Log-Volumen, Inkremente, PR) ----------
            with guarded("T1 Tiefen-Trajektorie"):
                dt = table("G", "depth_trajectory")
                k_def = 5 if 5 in set(dt["k"]) else int(dt["k"].iloc[0])
                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                for (K, Cc), g in sorted(dt[(dt["readout"] == "mean") & (dt["k"] == k_def)].groupby(["K", "C"])):
                    m = g.groupby("layer")[["log_volume", "delta_log_volume"]].mean()
                    axes[0].plot(m.index, m["log_volume"], marker="o", ms=3, label=f"K={K} {Cc}")
                    axes[1].plot(m.index, m["delta_log_volume"], marker="o", ms=3, label=f"K={K} {Cc}")
                for (K, Cc), g in sorted(dt[(dt["readout"] == "last") & (dt["k"] == k_def)].groupby(["K", "C"])):
                    m = g.groupby("layer")["participation_ratio"].mean()
                    axes[2].plot(m.index, m.values, marker="s", ms=3, label=f"K={K} {Cc}")
                axes[0].set_title(f"Log-Volumen der Realisierungswolke (Mean-Pool, k={k_def})", fontsize=9)
                axes[1].set_title("Inkrement Δ_ℓ log-Vol (negativ = Kontraktion; Vorzeichenwechsel = „Atmen“)", fontsize=9)
                axes[1].axhline(0, color="k", lw=0.6)
                axes[2].set_title(f"Participation Ratio (Last-Token, k={k_def})", fontsize=9)
                for ax in axes:
                    ax.set_xlabel("Snapshot ℓ")
                    ax.legend(fontsize=7)
                savefig("t1_depth_trajectory")
                plt.show()
                cols = [c for c in ["family_id", "K", "C", "readout", "k", "breathing_index", "breathing_amplitude",
                                    "layer_of_max_contraction", "layer_of_max_expansion", "net_change"] if c in dt.columns]
                print("Atem-Index (Vorzeichenwechsel von Δ_ℓ log-Vol), Amplitude und Phasenlage pro Bedingung:")
                show(dt[dt["k"] == k_def].drop_duplicates(["family_id", "K", "C", "readout"])[cols], n=40)
            """
        ),
        code(
            """
            # --- G1: Kontraktion unter Sprachwechsel, n-angepasst, mit Familien-Bootstrap (Kontrast 8) --
            # d_log_volume = log-Vol(K, C) - log-Vol(K=0) per family and snapshot; the mixed cloud is
            # re-sampled to the K=0 cloud's size because the log-volume sums min(n, d) eigenvalues
            # (raw values of clouds with different n are not comparable; both are reported).
            with guarded("G1 Kontraktion"):
                cb = table("G", "contraction_boot")
                k_def = 5 if 5 in set(cb["k"]) else int(cb["k"].iloc[0])
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, (q, ro) in zip(axes, [("d_log_volume", "mean"), ("d_participation_ratio", "last")]):
                    sub = cb[(cb["quantity"] == q) & (cb["readout"] == ro) & (cb["k"] == k_def)]
                    for (K, Cc), g in sorted(sub.groupby(["K", "C"])):
                        g = g.sort_values("layer")
                        pre = bool(truthy(g["preregistered"]).any())
                        ax.errorbar(g["layer"], g["mean"],
                                    yerr=[(g["mean"] - g["ci_low"]).clip(lower=0).fillna(0), (g["ci_high"] - g["mean"]).clip(lower=0).fillna(0)],
                                    marker="o", ms=4, capsize=2, lw=2 if pre else 1, label=f"K={K} {Cc}" + (" (präregistriert)" if pre else ""))
                    ax.axhline(0, color="k", lw=0.6)
                    ax.set_xlabel("Snapshot ℓ")
                    ax.set_title(f"{q} ({ro}-Readout, k={k_def}); 95 %-CI aus {limits.n_boot} Familien-Bootstraps", fontsize=9)
                    ax.legend(fontsize=7)
                savefig("g1_contraction")
                plt.show()
                con = table("G", "contraction")
                pre = con[truthy(con["preregistered"])]
                if len(pre):
                    print("Roh vs. n-angepasst (präregistrierte Zeilen K=8/C1/k=5, Mittel über Familien):")
                    show(pre.groupby(["readout", "layer"])[["n_mixed", "n_ref", "d_log_volume_raw", "d_log_volume",
                                                              "d_participation_ratio_raw", "d_participation_ratio"]].mean())
            """
        ),
        code(
            """
            # --- T2: Positions-Trajektorie — Log-Volumen pro Position (Präfix-Bins, ab Wechselstelle, Tail) --
            with guarded("T2 Positions-Trajektorie"):
                pt = table("G", "position_traj")
                site0 = C.dossier_site(mb).key if C.dossier_site(mb).key in set(pt["site"]) else str(pt["site"].iloc[0])
                sub = pt[pt["site"] == site0]
                regions = [r for r in ["prefix_bin", "from_switch", "tail"] if r in set(sub["region"])]
                titles = {"prefix_bin": "Präfix (relative Positionsbins)", "from_switch": "Position relativ zur letzten Wechselstelle",
                          "tail": "gemeinsamer Schlussabschnitt (vom Ende her)"}
                fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 3.8), squeeze=False)
                for ax, region in zip(axes[0], regions):
                    for grp, g in sorted(sub[sub["region"] == region].groupby("group")):
                        g = g.sort_values("index")
                        ax.plot(g["index"], g["log_volume"], marker="o", ms=3, label=grp)
                    ax.set_title(titles[region], fontsize=9)
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Log-Volumen (Last-Token je Position)")
                    ax.legend(fontsize=6)
                fig.suptitle(f"T2 an {site0}: wandert die Verdichtung mit den Wechselstellen?", fontsize=10)
                savefig("t2_position_trajectory")
                plt.show()
            """
        ),
        code(
            """
            # --- T3: Steering-Validität — cos(Steuerverschiebung, mean(K=8) − mean(K=0)) (Kontrast 9) --
            with guarded("T3 Steering"):
                st = table("G", "steering")
                sb = table("G", "steering_boot")
                eps0 = 0.05 if np.isclose(st["eps"], 0.05).any() else float(st["eps"].iloc[0])
                sites = list(dict.fromkeys(st["inject_site"]))
                fig, axes = plt.subplots(1, len(sites), figsize=(6 * len(sites), 4), squeeze=False, sharey=True)
                for ax, isk in zip(axes[0], sites):
                    g = st[(st["inject_site"] == isk) & np.isclose(st["eps"], eps0)]
                    m = g.groupby("layer")[["cos_steer_vs_K", "cos_rand_mean", "cos_rand_sd"]].mean()
                    ax.plot(m.index, m["cos_steer_vs_K"], "o-", label="lang_dom (Mittel über Familien)")
                    ax.fill_between(m.index, m["cos_rand_mean"] - 2 * m["cos_rand_sd"], m["cos_rand_mean"] + 2 * m["cos_rand_sd"],
                                    color="grey", alpha=0.25, label="Zufallsrichtungen ± 2 SD")
                    ax.axhline(0.3, color="r", lw=0.8, ls="--", label="Mindesteffekt cos ≥ 0,3")
                    ax.axhline(0, color="k", lw=0.6)
                    for b in sorted(set(g[truthy(g["at_max_disp_layer"])]["layer"])):
                        ax.axvline(b, color="C1", lw=0.8, ls=":")
                    ax.set_title(f"Injektion {isk}, ε={eps0:g}", fontsize=9)
                    ax.set_xlabel("Snapshot ℓ")
                    ax.legend(fontsize=7)
                axes[0, 0].set_ylabel("cos(Steuerverschiebung, mean(K=8) − mean(K=0))")
                fig.suptitle("T3: reproduziert kontinuierliches Steuern die Geometrie echter Wechsel? "
                             "(gepunktet: Layer maximaler diskreter Verschiebung)", fontsize=10)
                savefig("t3_steering_cosine")
                plt.show()
                top = sb[sb["quantity"] == "cos_at_max_disp_layer"]
                if len(top):
                    print("Präregistrierter Kontrast 9: Kosinus im Layer maximaler Verschiebung (Familien-Bootstrap):")
                    show(top[["inject_site", "eps", "preregistered", "min_effect_met", "mean", "ci_low", "ci_high", "n_clusters"]])
            """
        ),
        code(
            """
            # --- T5: Kopplung von Δ log-Vol mit λ_ℓ (D1) und Lens-KL (D2) über die Layer --------------
            with guarded("T5 Kopplung"):
                cp = table("G", "coupling")
                if cp.empty:
                    print("Keine Kopplung: Stufe D ist für dieses Modell unter RUNS_ROOT noch nicht gelaufen (Notebook 03) "
                          "oder die Tabellen damage_profile / lens fehlen.")
                else:
                    show(cp.sort_values(["partner", "K", "C"]), n=60)
                    fig, ax = plt.subplots(figsize=(7, 3.6))
                    for partner, g in cp.groupby("partner"):
                        lab = {"lambda": "λ_ℓ (Störungswachstum, Stufe D)", "kl": "Lens-KL (Stufe D)"}.get(partner, partner)
                        ax.scatter([f"K={k} {c}" for k, c in zip(g["K"], g["C"])], g["rho"], label=lab, s=30)
                    ax.axhline(0, color="k", lw=0.6)
                    ax.set_ylabel("Spearman ρ über Layer")
                    ax.set_title("Ist Verdichtung der Ordnungsparameter der Nicht-Gleichgewichts-Signaturen?", fontsize=9)
                    ax.legend(fontsize=7)
                    savefig("t5_coupling")
                    plt.show()
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 07 lambda dynamics (stage H)
# ======================================================================================
def nb_07() -> list[tuple[str, str]]:
    return [
        header_cell(
            "07_lambda_dynamics", "H", "λ-Dynamik im Aktivierungsraum",
            purpose=("Biggio et al. (2013/2017), Algorithmus 1 im Aktivierungsraum (`docs/DESIGN.md` §4.6): von der "
                     "Dossier-Stelle (Block-Eingang des letzten Blocks, letzte Position) eines K=0-Starts aus wird "
                     "F(h) = ĝ(h) − λ·p̂(h) per Einheitsschritten minimiert — ĝ eine Diskriminante (`target_logit`: "
                     "−log p(Ziel); `probe`: lineare Sprach-Probe; `entropy`; `sae_feature`), p̂ eine KDE über echte "
                     "K=8/C1-Zustände (Mahalanobis-gewhitet). λ = 0 überquert die Grenze schnellstmöglich, aber ohne "
                     "Datenstützung; λ > 0 erzwingt Mimikry. Pro λ werden Pfadgeometrie, Datenstützung des Endpunkts "
                     "(Mahalanobis zur Zielwolke, k-NN, SAE-FVU), Wirkung (B, Entropie, p_freq) und die SAE-Features "
                     "gemessen; Spearman-ρ mit λ über den Sweep, Familien-Bootstrap; zusätzlich B(x)(λ) für State 2143."),
            inputs=("Stimulusbank, Richtungsbank (für `probe`: `lang_probe`), K=8/C1-Realisierungswolke (Stufe G aus `ctx`, "
                    "sonst neu gebaut), optional SAE (`USE_SAE=True` auf step143000; Rauchtest: zufälliger Platzhalter-SAE, "
                    "markiert `sae_source = synthetic_random`), Anker für State 2143."),
            outputs=("Tabellen `lambda_paths`, `lambda_steps`, `lambda_corr`, `lambda_boot`, `lambda_features`, "
                     "`lambda_features_summary`, `lambda_original`."),
            steps=("Setup → Drive → Konfiguration → Stufe-H-Parameter → Modell → Stufe H → Pfade (g, Dichte, F) → "
                   "Endpunkt-Größen vs. λ → Spearman mit Bootstrap → SAE-Feature-Kandidaten → State 2143 → Zusammenfassung."),
            prereg=("Kontrast 10 (`lambda_boot`-Zeilen mit `preregistered = True`: Diskriminante `target_logit`, ρ(λ, SAE-FVU) < 0, "
                    "ρ(λ, Schritte bis zur Grenze) offen, ρ(λ, Mahalanobis zur Zielwolke) < 0; Feature-Kandidaten "
                    "|ρ| ≥ 0,7 in `lambda_features`) und Kontrast 11 (`lambda_original`: Vorzeichen von B bei λ_rel = 3 "
                    "gegenüber λ = 0, `same_sign`). Kandidaten werden auf Entwicklungsfamilien eingefroren und auf "
                    "Testfamilien geprüft."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("07_lambda_dynamics", "H"),
        code(
            """
            # --- Stufe-H-Parameter (Budget ≈ Starts × Diskriminanten × λ-Stufen × Schritte Forward+Backward) --
            H_N_STEPS = None       # Schritte pro Pfad (Standard limits.h_n_steps = 100; kleine GPU: 30)
            H_MAX_STARTS = None    # Startzustände (K=0-Formulierungen) pro Familie (Standard 4)
            if H_N_STEPS is not None:
                limits.h_n_steps = int(H_N_STEPS)
            if H_MAX_STARTS is not None:
                limits.h_max_starts = int(H_MAX_STARTS)
            print(f"λ_rel-Sweep {limits.h_lambdas_rel}; Pfad: {limits.h_n_steps} Schritte × {limits.h_step_rel}·‖h₀‖ in der "
                  f"ℓ₂-Kugel {limits.h_dmax_rel}·‖h₀‖; Starts/Familie {limits.h_max_starts}; Diskriminanten {limits.h_discriminants}; "
                  f"KDE k_nearest={limits.h_knn}, Rang {limits.h_cov_rank}; Grenzabstand {limits.h_cross_nats} nats; "
                  f"SAE-Platzhalter bei Bedarf: {limits.synthetic_sae_if_missing}")
            """
        ),
        model_cell(),
        runner_cell(),
        substages_cell(),
        code(
            """
            # --- Stufe H ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage H --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> [--use-sae]
            # (--stage G,H shares the realization clouds; with --use-sae on step143000 the sae_feature
            # discriminant and the feature correlations use the published SAE instead of a placeholder)
            res_H = run_stage("H")
            print({k: len(v) for k, v in res_H.items()})
            manifest_H = substage_report("H")
            if manifest_H:
                print("Starts:", manifest_H.get("n_starts"), "| Zielzustände:", manifest_H.get("n_target"), "| KDE:", manifest_H.get("kde"),
                      "| SAE:", manifest_H.get("sae_source"))
            """
        ),
        code(
            """
            # --- Pfade im Aktivierungsraum: g, Dichte und F = g − λ·Dichte pro Schritt (erster Start) -------
            with guarded("λ-Pfade"):
                steps = table("H", "lambda_steps")
                tag0 = str(steps["tag"].iloc[0])
                discs = list(dict.fromkeys(steps["discriminant"]))
                fig, axes = plt.subplots(len(discs), 3, figsize=(15, 3.2 * len(discs)), squeeze=False)
                for i, disc in enumerate(discs):
                    g = steps[(steps["tag"] == tag0) & (steps["discriminant"] == disc)]
                    for lam_rel, p in sorted(g.groupby("lam_rel")):
                        p = p.sort_values("step")
                        for ax, col in zip(axes[i], ["g", "density", "F"]):
                            ax.plot(p["step"], p[col], marker=".", ms=3, label=f"λ_rel={lam_rel:g}")
                    axes[i, 0].set_ylabel(disc)
                    axes[i, 0].set_title("Diskriminante g (g < 0: Grenze überquert)", fontsize=9)
                    axes[i, 1].set_title("Zielklassen-Dichte p̂ (KDE, Mahalanobis)", fontsize=9)
                    axes[i, 2].set_title("F = g − λ·p̂ (Alg. 1 minimiert F)", fontsize=9)
                    axes[i, 0].axhline(0, color="k", lw=0.6)
                for ax in axes[-1]:
                    ax.set_xlabel("Schritt")
                axes[0, 2].legend(fontsize=7)
                fig.suptitle(f"Start {tag0}: λ = 0 steigt schnellstmöglich ab, λ > 0 bleibt in der Datenstützung (Mimikry)", fontsize=10, y=1.01)
                savefig("lambda_paths")
                plt.show()
            """
        ),
        code(
            """
            # --- Endpunkt-Größen als Funktion von λ: Schritte bis zur Grenze, Mahalanobis, SAE-FVU, B -------
            with guarded("Endpunkte vs. λ"):
                paths = table("H", "lambda_paths")
                quants = [q for q in ["steps_to_cross", "mahal_to_ref_mean", "knn_mean_dist", "sae_fvu", "B", "path_length"] if q in paths.columns]
                fig, axes = plt.subplots(1, len(quants), figsize=(3.4 * len(quants), 3.6), squeeze=False)
                for ax, q in zip(axes[0], quants):
                    for disc, g in paths.groupby("discriminant"):
                        m = g.groupby("lam_rel")[q].agg(["mean", "std"])
                        ax.errorbar(m.index, m["mean"], yerr=m["std"].fillna(0), marker="o", ms=4, capsize=2, label=disc)
                    ax.set_xscale("symlog", linthresh=0.1)
                    ax.set_xlabel("λ_rel")
                    ax.set_title(q, fontsize=9)
                axes[0, 0].legend(fontsize=7)
                fig.suptitle("Mittel ± SD über Starts (steps_to_cross: NaN = Grenze im Budget nicht erreicht)", fontsize=10)
                savefig("lambda_endpoints")
                plt.show()
                print("Anteil überquerter Grenzen pro (Diskriminante, λ_rel):")
                show(paths.assign(crossed=truthy(paths["crossed"]).astype(float)).pivot_table(index="discriminant", columns="lam_rel", values="crossed", aggfunc="mean"))
            """
        ),
        code(
            """
            # --- Spearman-ρ(λ, ·) pro Diskriminante mit Familien-Bootstrap (Kontrast 10) -----------------
            with guarded("Spearman-Korrelationen"):
                lb = table("H", "lambda_boot")
                pre = lb[truthy(lb["preregistered"])]
                print("Präregistrierte Zeilen (target_logit; erwartete Vorzeichen: sae_fvu < 0, mahal_to_ref_mean < 0, steps_to_cross offen):")
                show(pre[["discriminant", "quantity", "expected_sign", "sign_as_expected", "mean", "ci_low", "ci_high", "n_clusters"]])
                quants = [q for q in ["steps_to_cross", "mahal_to_ref_mean", "knn_mean_dist", "sae_fvu", "local_log_volume", "B", "entropy",
                                      "kl_from_base", "path_length", "tortuosity"] if q in set(lb["quantity"])]
                sub = lb[lb["quantity"].isin(quants)]
                discs = list(dict.fromkeys(sub["discriminant"]))
                fig, ax = plt.subplots(figsize=(1.1 * len(quants) + 3, 4))
                x = np.arange(len(quants))
                w = 0.8 / max(1, len(discs))
                for i, disc in enumerate(discs):
                    g = sub[sub["discriminant"] == disc].set_index("quantity").reindex(quants)
                    ax.bar(x + (i - (len(discs) - 1) / 2) * w, g["mean"], w,
                           yerr=[(g["mean"] - g["ci_low"]).clip(lower=0).fillna(0), (g["ci_high"] - g["mean"]).clip(lower=0).fillna(0)],
                           capsize=2, label=disc)
                ax.set_xticks(x)
                ax.set_xticklabels(quants, rotation=40, ha="right", fontsize=8)
                ax.axhline(0, color="k", lw=0.6)
                ax.set_ylabel("Spearman ρ(λ, Größe), Familienmittel")
                ax.set_title("Welche Pfadgrößen korrelieren monoton mit λ?", fontsize=10)
                ax.legend(fontsize=7)
                savefig("lambda_spearman")
                plt.show()
            """
        ),
        code(
            """
            # --- SAE-Features vs. λ: Kandidaten |ρ| ≥ 0,7; nur bei λ=0 bzw. nur bei λ>0 aktive Features ----
            with guarded("SAE-Feature-Korrelationen"):
                fs = table("H", "lambda_features_summary")
                paths = table("H", "lambda_paths")
                if fs.empty:
                    print("Keine SAE-Feature-Tabelle (kein SAE: USE_SAE=True auf step143000; im Rauchtest ein Platzhalter-SAE).")
                else:
                    src = str(paths["sae_source"].iloc[0]) if len(paths) else "?"
                    print("SAE-Quelle:", src, "(synthetic_random = Platzhalter: Funktionstest, kein Befund)")
                    cand = fs[truthy(fs["candidate_all"])]
                    print(f"{len(cand)} Kandidaten-Features (|ρ| ≥ 0,7 bei allen Starts):")
                    show(cand.sort_values("rho_mean"), n=40)
                    fig, ax = plt.subplots(figsize=(7, 3.6))
                    for disc, g in fs.groupby("discriminant"):
                        ax.scatter(g["feature"], g["rho_mean"], s=12, label=disc, alpha=0.7)
                    ax.axhline(0.7, color="r", lw=0.7, ls="--")
                    ax.axhline(-0.7, color="r", lw=0.7, ls="--")
                    ax.axhline(0, color="k", lw=0.5)
                    ax.set_xlabel("SAE-Feature (Latent-Index)")
                    ax.set_ylabel("mittleres ρ(λ, Aktivierung am Endpunkt)")
                    ax.set_title("Feature-Aktivierung vs. λ (gestrichelt: Kandidatenschwelle |ρ| = 0,7)", fontsize=9)
                    ax.legend(fontsize=7)
                    savefig("lambda_features")
                    plt.show()
                    print("Nur bei λ=0 aktiv (Artefakt-Kandidaten) bzw. nur bei λ>0 aktiv (Repräsentations-Kandidaten):")
                    show(fs[(fs["n_active_only_lambda0"] > 0) | (fs["n_active_only_lambda_pos"] > 0)]
                         .sort_values(["n_active_only_lambda_pos", "n_active_only_lambda0"], ascending=False), n=30)
            """
        ),
        code(
            """
            # --- Kontrast 11: B(x)(λ) für State 2143 — bleibt der 98k/99k-Nutzen auf der Datenmannigfaltigkeit bestehen? --
            with guarded("State 2143: B(λ)"):
                orig = table("H", "lambda_original")
                if orig.empty:
                    print("Keine Original-Tabelle (Anker nicht verfügbar oder zu wenige Zielzustände).")
                else:
                    r = orig.iloc[0]
                    print(f"B(λ=0) = {r['B_lambda0']:+.4f} nats, B(λ_rel=3) = {r['B_lambda3']:+.4f} nats, gleiches Vorzeichen: {r['same_sign']}; "
                          f"Zielwolke: {r['target_cloud']} (n_orig={r['n_target_orig']}, n={r['n_target']})")
                    g = orig.sort_values("lam_rel")
                    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
                    a1.plot(g["lam_rel"], g["B"], "o-", label="B = ln p_δ(bey) − ln p₀(bey)")
                    a1.axhline(0, color="k", lw=0.6)
                    a1.set_xscale("symlog", linthresh=0.1)
                    a1.set_xlabel("λ_rel")
                    a1.set_ylabel("B [nats]")
                    a1.set_title("State 2143 (Ziel 22905 „bey“), Diskriminante target_logit", fontsize=9)
                    a1.legend(fontsize=7)
                    a2.plot(g["lam_rel"], g["mahal_to_ref_mean"], "s-", color="C1", label="Mahalanobis zur Zielwolke")
                    a2.plot(g["lam_rel"], g["sae_fvu"], "^-", color="C2", label="SAE-FVU am Endpunkt")
                    a2.set_xscale("symlog", linthresh=0.1)
                    a2.set_xlabel("λ_rel")
                    a2.set_title("Datenstützung des Endpunkts", fontsize=9)
                    a2.legend(fontsize=7)
                    savefig("lambda_original_2143")
                    plt.show()
                    show(g[["lam_rel", "lam", "n_steps", "steps_to_cross", "crossed", "B", "entropy", "p_freq", "mahal_to_ref_mean", "sae_fvu"]])
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 08 domain compass and heads (stage I)
# ======================================================================================
def nb_08() -> list[tuple[str, str]]:
    return [
        header_cell(
            "08_compass_heads", "I", "Domänen-Kompass, Domänenachsen und Heads",
            purpose=("Wohin steuern? (`docs/DESIGN.md` §4.7) Domänenprofil des State-2143-Zustands h₂₀₇ und der Richtungen "
                     "`ent`, `lang_dom`, `freq` (Logit-Lens-Proxy des Suszeptibilitätsatlas: beförderte Masse pro Domäne, "
                     "z gegen Zufallsrichtungen gleicher Dosis), Import der gespeicherten Atlas-Extraktion (optional), "
                     "Domänenachsen `dom_law`, `dom_culture_politics`, `dom_protein_misfolding` (Difference-of-Means auf "
                     "Domänensätzen), **Gegensteuern** (Jura-Anteil herausprojizieren, dann Kultur/Politik-Dosis, als "
                     "Substitution geschrieben) gegen `dom_law`, `lang_dom` und Zufall nach Familiendomäne, Head-Attribution "
                     "(Mean-Ablation, direkter Logit-Beitrag) und Steering am **schlechtesten Head** vs. am Residualstrom."),
            inputs=("Stimulusbank (für den Domänenvergleich die Familien F13–F16 Kultur/Politik, F17–F18 Jura, F19–F20 "
                    "Proteinfehlfaltung, Rest Alltag), Richtungsbank, Domänenlexikon `data/lexicon/domains.json`, optional "
                    "`data/atlas/*.txt` (ATLAS_DIR), Anker für h₂₀₇."),
            outputs=("Tabellen `compass`, `compass_summary`, `atlas_overlap`, `domain_axes`, `direction_domain`, "
                     "`direction_domain_T`, `direction_domain_boot`, `head_effects`, `head_direct`, `worst_head`, "
                     "`head_steering`, `head_steering_boot`; Achsen in `directions_domain.npz`."),
            steps=("Setup → Drive → Konfiguration → Stufe-I-Parameter → Modell → Stufe I → Domänenprofile → Domänenachsen → "
                   "Richtung × Domäne → Head-Attribution und schlechtester Head → Head-Arm → Zusammenfassung."),
            prereg=("Kontrast 12 (`compass_summary`, `direction = ent`, `lang = E`: `law_is_top` / `law_rank` — Dokumentation, "
                    "kein Test), Kontrast 13 (`direction_domain_boot`, `quantity = B_cs_minus_lang_dom`, "
                    "`family_domain = culture_politics`, K=8/C1, Mindesteffekt 0,05 nats) und Kontrast 14 "
                    "(`head_steering_boot`, `quantity = B_head_minus_residual`, zweiseitig; Head-Auswahl auf "
                    "Entwicklungsfamilien eingefroren)."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("08_compass_heads", "I"),
        code(
            """
            # --- Stufe-I-Parameter ---------------------------------------------------------------
            ATLAS_DIR = None   # Verzeichnis der gespeicherten Atlas-Extraktion (*.txt / *.md); None = data/atlas; z. B. "/content/drive/MyDrive/LSNE/inputs/atlas"
            _atlas = ATLAS_DIR or str(C._REPO_ROOT / "data" / "atlas")
            _n_atlas = len([f for f in (os.listdir(_atlas) if os.path.isdir(_atlas) else []) if f.endswith((".txt", ".md"))])
            print("Domänen-Lexikon:", C._REPO_ROOT / "data" / "lexicon" / "domains.json")
            print(f"Atlas-Verzeichnis: {_atlas} ({_n_atlas} Datei(en); ohne Dateien läuft nur der Logit-Lens-Proxy)")
            print(f"Dosen B(x): {limits.i_doses}; Head-Arm-Dosen (relativ zur Head-RMS): {limits.i_head_eps}; "
                  f"Zufallsrichtungen des Domänenprofils: {limits.i_n_random_profile}")
            if ATLAS_DIR:
                CLI_COMMAND += f" --atlas-dir {ATLAS_DIR}"
            """
        ),
        model_cell(),
        runner_cell(),
        substages_cell(),
        code(
            """
            # --- Stufe I ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage I --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> [--atlas-dir data/atlas]
            res_I = run_stage("I", atlas_dir=ATLAS_DIR)
            print({k: len(v) for k, v in res_I.items()})
            manifest_I = substage_report("I")
            if manifest_I:
                print("Zustand h₂₀₇ aus:", manifest_I.get("state_source"), "| schlechtester Head:", manifest_I.get("worst_head"),
                      "| verworfene Lexikonwörter (mehr als ein Token):", manifest_I.get("domain_ids_dropped"))
            """
        ),
        code(
            """
            # --- Kontrast 12: Domänenprofil von h₂₀₇ und der Richtungen (z gegen Zufallsrichtungen) ----------
            with guarded("Domänen-Kompass"):
                comp = table("I", "compass")
                summ = table("I", "compass_summary")
                print("Zustand:", comp["state_source"].iloc[0] if len(comp) else "?", "| Atlas-Erwartung für ent: law (Jura) ganz oben")
                show(summ[summ["state_source"] != "sae_decoder"], n=40)
                lang0 = "E"
                sub = comp[(comp["lang"] == lang0) & (comp["state_source"] != "sae_decoder")]
                dirs = [d for d in ["state", "ent", "lang_dom", "freq", "dom_law", "dom_culture_politics", "dom_protein_misfolding"]
                        if d in set(sub["direction"])]
                doms = sorted(set(sub["domain"]))
                piv = sub.pivot_table(index="direction", columns="domain", values="z").reindex(dirs)[doms]
                fig, ax = plt.subplots(figsize=(0.8 * len(doms) + 3, 0.5 * len(dirs) + 2))
                vmax = float(np.nanmax(np.abs(piv.to_numpy(dtype=float)))) or 1.0
                im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_xticks(range(len(doms)))
                ax.set_xticklabels(doms, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(dirs)))
                ax.set_yticklabels(dirs)
                for i in range(len(dirs)):
                    for j in range(len(doms)):
                        val = piv.iat[i, j]
                        if np.isfinite(val):
                            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7)
                ax.set_title(f"Domänenprofil (Lexikon {lang0}): z der beförderten Masse pro Domäne, ε=0,05, "
                             f"gegen {limits.i_n_random_profile} Zufallsrichtungen", fontsize=9)
                fig.colorbar(im, ax=ax, label="z", shrink=0.8)
                savefig("compass_domain_profile")
                plt.show()
                atl = table("I", "atlas_overlap")
                if len(atl):
                    print("Überlappung der Atlas-Cluster (Schlüsselwörter) mit den Domänenlexika:")
                    show(atl.sort_values("jaccard", ascending=False), n=30)
                else:
                    print("Keine Atlas-Extraktion gefunden (data/atlas bzw. ATLAS_DIR ohne *.txt) -> nur der lokale Proxy.")
            """
        ),
        code(
            """
            # --- Domänenachsen: Kosinus zur Richtungsbank und untereinander --------------------------
            with guarded("Domänenachsen"):
                ax_t = table("I", "domain_axes")
                show(ax_t)
                cos_cols = [c for c in ax_t.columns if c.startswith("cos_") and c != "cos_denoised_plain"]
                M = ax_t.set_index("name")[cos_cols].astype(float)
                fig, ax = plt.subplots(figsize=(0.9 * len(cos_cols) + 2, 2.6))
                im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks(range(len(cos_cols)))
                ax.set_xticklabels([c[4:] for c in cos_cols], rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(M)))
                ax.set_yticklabels(M.index, fontsize=8)
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        if np.isfinite(M.iat[i, j]):
                            ax.text(j, i, f"{M.iat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
                ax.set_title("Kosinus der Domänenachsen zu ent / lang_dom / freq / lang_probe und untereinander", fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.8)
                savefig("domain_axes_cosines")
                plt.show()
            """
        ),
        code(
            """
            # --- Kontrast 13: B(x) pro Richtung × Familiendomäne (Gegensteuern gegen dom_law, dann dom_culture_politics) --
            with guarded("Richtung × Domäne"):
                dd = table("I", "direction_domain")
                db = table("I", "direction_domain_boot")
                std = dd[dd["kind"] == "standard"] if "kind" in dd else dd
                eps0 = 0.05 if np.isclose(std["eps"], 0.05).any() else float(sorted(set(std["eps"]))[-1])
                sub = std[np.isclose(std["eps"], eps0)]
                sub = sub[sub["K"] == 8] if (sub["K"] == 8).any() else sub
                order = [d for d in ["dom_culture_politics_cs", "dom_culture_politics", "dom_law", "lang_dom", "rand"] if d in set(sub["direction"])]
                groups = list(dict.fromkeys(sub["family_domain"]))
                piv = sub.groupby(["family_domain", "direction"])["B"].mean().unstack("direction").reindex(groups)[order]
                ax = piv.plot(kind="bar", figsize=(8, 4), width=0.8)
                ax.axhline(0, color="k", lw=0.6)
                ax.set_ylabel(f"B = ln p_δ − ln p₀ [nats], ε={eps0:g}, K=8/C1")
                ax.set_xlabel("Familiendomäne")
                ax.set_title("Ist Steuern in der Schwachdomäne (Kultur/Politik) treffsicherer? (cs = nach Gegensteuern)", fontsize=10)
                ax.legend(fontsize=7, ncol=2)
                savefig("direction_x_domain")
                plt.show()
                q = db[db["quantity"] == "B_cs_minus_lang_dom"]
                if len(q):
                    print("Präregistrierter Kontrast 13: B(dom_culture_politics nach Gegensteuern) − B(lang_dom), Familien-Bootstrap, Mindesteffekt 0,05 nats:")
                    show(q[["family_domain", "K", "C", "eps", "preregistered", "min_effect_met", "mean", "ci_low", "ci_high", "n_clusters"]])
                    if not truthy(q["preregistered"]).any():
                        print("Hinweis: die präregistrierte Zeile braucht Kultur/Politik-Familien (F13–F16) im Lauf (FAMILIES = None).")
                T = table("I", "direction_domain_T")
                if len(T):
                    print("Texteffekt T(K=8, C1) pro Familiendomäne:")
                    show(T.groupby("family_domain")["T"].agg(["mean", "std", "count"]))
            """
        ),
        code(
            """
            # --- Head-Attribution: Mean-Ablation und direkter Logit-Beitrag pro Head, schlechtester Head ----
            with guarded("Head-Attribution"):
                eff = table("I", "head_effects")
                wh = table("I", "worst_head")
                hd = table("I", "head_direct")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8))
                piv = eff.groupby(["layer", "head"])["delta"].mean().unstack("head")
                vmax = float(np.nanmax(np.abs(piv.to_numpy(dtype=float)))) or 1.0
                im = a1.imshow(piv.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                a1.set_xticks(range(piv.shape[1]))
                a1.set_xticklabels(piv.columns)
                a1.set_yticks(range(piv.shape[0]))
                a1.set_yticklabels([f"L{l}" for l in piv.index])
                a1.set_xlabel("Head")
                a1.set_title("Δ ln p(Ziel) bei Mean-Ablation (Mittel über Varianten; Stern = schlechtester Head)", fontsize=9)
                fig.colorbar(im, ax=a1, shrink=0.8)
                if len(wh):
                    sel = wh[truthy(wh["selected"])].iloc[0]
                    a1.plot([list(piv.columns).index(sel["head"])], [list(piv.index).index(sel["layer"])], marker="*", color="k", ms=14)
                    print(f"Schlechtester Head: L{sel['layer']}H{sel['head']} (Auswahl: {sel['selection']}; schwache Familien: "
                          f"{sel['families_weak']}; stabile Familien: {sel['families_stable'] or '-'})")
                    show(wh.head(8))
                pv2 = hd.groupby(["layer", "head"])["direct_effect"].mean().unstack("head")
                vmax2 = float(np.nanmax(np.abs(pv2.to_numpy(dtype=float)))) or 1.0
                im2 = a2.imshow(pv2.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
                a2.set_xticks(range(pv2.shape[1]))
                a2.set_xticklabels(pv2.columns)
                a2.set_yticks(range(pv2.shape[0]))
                a2.set_yticklabels([f"L{l}" for l in pv2.index])
                a2.set_xlabel("Head")
                a2.set_title("direkter Beitrag zum Ziel-Logit (W_O-Slice, letzte Position)", fontsize=9)
                fig.colorbar(im2, ax=a2, shrink=0.8)
                savefig("head_attribution")
                plt.show()
            """
        ),
        code(
            """
            # --- Kontrast 14: Steering am schlechtesten Head vs. am Residualstrom bei gleicher relativer Dosis --
            with guarded("Head-Arm"):
                hs = table("I", "head_steering")
                hb = table("I", "head_steering_boot")
                std = hs[hs["kind"] == "standard"] if "kind" in hs else hs
                piv = std.groupby(["direction", "eps", "arm"])["B"].mean().unstack("arm")
                ax = piv.plot(kind="bar", figsize=(7, 3.8), width=0.7)
                ax.axhline(0, color="k", lw=0.6)
                ax.set_ylabel("B [nats], Mittel über Varianten")
                ax.set_title("Head-Arm: Richtung durch den Head (W_O-Urbild) vs. Addition am Block-Ausgang", fontsize=9)
                ax.legend(fontsize=8)
                savefig("head_steering")
                plt.show()
                if len(hb):
                    print("B_head − B_residual pro (Richtung, Dosis), Familien-Bootstrap:")
                    show(hb)
                if len(std):
                    print("Realisierte Richtung im Residualstrom (Kosinus) und Urbild-Fehler des Head-Arms:")
                    show(std[std["arm"] == "head"].groupby(["direction", "eps"])[["rms_delta", "cos_realised_direction", "preimage_rel_err"]].mean())
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 09 RDX, Pile difficulty and sharing (stage J)
# ======================================================================================
def nb_09() -> list[tuple[str, str]]:
    return [
        header_cell(
            "09_rdx_sharing", "J", "RDX, Pile-Schwierigkeit und Sharing",
            purpose=("`docs/DESIGN.md` §4.8 a–c: (a) **RDX** — nicht absolute Repräsentationen, sondern Differenzvektoren "
                     "Δ = mean(A) − mean(B) zwischen kontrollierten Bedingungen (Kultur/Politik ↔ Proteinfehlfaltung, Jura ↔ "
                     "Kultur/Politik, Englisch ↔ Deutsch, K=0 ↔ K=8) werden erklärt: welcher Anteil von ‖Δ‖² liegt in einer "
                     "Kandidatenbasis (Domänenachsen, SAE-Decoderzeilen, Head-Beiträge, Hauptkomponenten), z-standardisiert "
                     "gegen Zufallsbasen gleicher Größe, Stabilität über Familien; (b) der **Gegenpol wird gemessen, nicht "
                     "gesetzt**: mittlere Token-NLL pro Pile-Teilgebiet aus per Range-Request gezogenen Trainingssequenzen "
                     "(Schlüsselwortprofil, PubMed-Marker), Rang von `protein_misfolding`; (c) **Sharing-Index** (Jaccard der "
                     "aktiven TopK-SAE-Features zwischen Bedingungen) und **Netzwerkaktivität** (Anteil aktiver Features, "
                     "Masse, Participation Ratio), Kosinus der Head-Beitragsprofile."),
            inputs=("Stimulusbank (Familiengruppen F13–F16 / F17–F18 / F19–F20 / Alltag), Richtungsbank, Domänenlexikon, "
                    "SAE (`USE_SAE=True`; Rauchtest: Platzhalter), Pile per Range-Request (Cache `LSNE_CACHE/pile_nll`; "
                    "`LSNE_OFFLINE=1` lässt die Rangfolge aus)."),
            outputs="Tabellen `pile_difficulty`, `pile_ranking`, `pile_rank_check`, `rdx`, `rdx_summary`, `rdx_stability`, `sharing`.",
            steps=("Setup → Drive → Konfiguration → Stufe-J-Parameter → Modell → Stufe J → Schwierigkeitsrangfolge → "
                   "RDX-z-Karten → Stabilität → Sharing und Aktivität → Zusammenfassung."),
            prereg=("Kontrast 15 (`rdx_summary`-Zeile `contrast = culture_politics_vs_misfolding`, `basis = domain_axes`, "
                    "`readout = last`: z ≥ 2 in mindestens der Hälfte der Layer, `min_effect_met`) und Kontrast 16 "
                    "(`pile_rank_check`: Rang von `protein_misfolding` aus N ≥ 2 000 Sequenzen; liegt es nicht im schwersten "
                    "Drittel, ersetzt `counterpole` den Gegenpol durch das tatsächlich schwerste Teilgebiet)."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("09_rdx_sharing", "J"),
        code(
            """
            # --- Stufe-J-Parameter ---------------------------------------------------------------
            J_N_PILE = None     # Pile-Sequenzen der Schwierigkeitsrangfolge (Standard 2 000 = Präregistrierung 16; Rauchtest 3)
            if J_N_PILE is not None:
                limits.j_n_pile = int(J_N_PILE)
            print(f"Pile-Sequenzen: {limits.j_n_pile} (je {limits.j_pile_max_len} Tokens, Range-Requests, Cache {C.cache_dir('pile_nll')}); "
                  f"Zufallsbasen: {limits.j_n_random_bases}; PCA-Komponenten: {limits.j_n_pca}; Offline: {C.offline()}")
            """
        ),
        model_cell(),
        runner_cell(),
        substages_cell(),
        code(
            """
            # --- Stufe J ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage J --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> [--use-sae]
            res_J = run_stage("J")
            print({k: len(v) for k, v in res_J.items()})
            manifest_J = substage_report("J")
            if manifest_J:
                print("Kontraste (n_A, n_B):", manifest_J.get("contrasts"), "| Basen:", manifest_J.get("bases"), "| SAE:", manifest_J.get("sae_source"))
            """
        ),
        code(
            """
            # --- Kontrast 16: empirische Schwierigkeitsrangfolge der Pile-Teilgebiete (mittlere Token-NLL) ----
            with guarded("Pile-Schwierigkeit"):
                rk = table("J", "pile_ranking")
                chk = table("J", "pile_rank_check")
                pile = table("J", "pile_difficulty")
                if rk.empty:
                    print("Keine Rangfolge (offline oder Range-Requests fehlgeschlagen; siehe Substufen-Bericht).")
                else:
                    rk = rk.sort_values("rank")
                    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rk) + 1.5))
                    y = np.arange(len(rk))
                    ax.barh(y, rk["mean_nll"], xerr=[(rk["mean_nll"] - rk["ci_low"]).clip(lower=0), (rk["ci_high"] - rk["mean_nll"]).clip(lower=0)],
                            color=["C3" if s == "protein_misfolding" else "C0" for s in rk["subfield"]], capsize=2)
                    ax.set_yticks(y)
                    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(rk["subfield"], rk["n_seqs"])], fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xlabel("mittlere Token-NLL [nats] (Bootstrap-CI über Sequenzen)")
                    ax.set_title(f"Schwierigkeitsrangfolge aus {len(pile)} Pile-Sequenzen (rot: protein_misfolding)", fontsize=9)
                    savefig("pile_difficulty_ranking")
                    plt.show()
                    show(chk)
                    if len(chk):
                        c = chk.iloc[0]
                        print(f"Gegenpol laut Ersetzungsregel: {c['counterpole']} (protein_misfolding: Rang {c['rank']} von {c['n_subfields']}, "
                              f"im schwersten Drittel: {c['in_hardest_third']})")
                    if "subfield_lexicon" in pile.columns:
                        print("Teilgebiet (Marker) × Lexikon-Domäne derselben Sequenzen:")
                        show(pd.crosstab(pile["subfield"], pile["subfield_lexicon"].fillna("")))
            """
        ),
        code(
            """
            # --- Kontrast 15: RDX — erklärter Anteil von ‖Δ‖² pro Kontrast × Basis × Layer, z gegen Zufallsbasen --
            with guarded("RDX"):
                rdx = table("J", "rdx")
                rs = table("J", "rdx_summary")
                pooled = rdx[(rdx["unit"] == "pooled") & (rdx["readout"] == "last")]
                contrasts = list(dict.fromkeys(pooled["contrast"]))
                bases = [b for b in ["domain_axes", "sae", "heads", "pca"] if b in set(pooled["basis"])]
                fig, axes = plt.subplots(1, len(contrasts), figsize=(4.5 * len(contrasts), 3.6), squeeze=False)
                im = None
                for ax, cn in zip(axes[0], contrasts):
                    piv = pooled[pooled["contrast"] == cn].pivot_table(index="basis", columns="layer", values="z").reindex(bases)
                    im = ax.imshow(piv.to_numpy(dtype=float), cmap="viridis", aspect="auto", vmin=-2, vmax=6)
                    ax.set_xticks(range(piv.shape[1]))
                    ax.set_xticklabels([f"L{l}" for l in piv.columns])
                    ax.set_yticks(range(piv.shape[0]))
                    ax.set_yticklabels(piv.index)
                    for i in range(piv.shape[0]):
                        for j in range(piv.shape[1]):
                            if np.isfinite(piv.iat[i, j]):
                                ax.text(j, i, f"{piv.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7, color="w")
                    ax.set_title(cn, fontsize=9)
                fig.suptitle("z des erklärten Anteils gegen Zufallsbasen gleicher Größe (Last-Token; z ≥ 2 = Mindesteffekt)", fontsize=10)
                if im is not None:
                    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="z")
                savefig("rdx_z")
                plt.show()
                show(rs.sort_values(["preregistered", "contrast", "basis"], ascending=[False, True, True]), n=60)
                if "culture_politics_vs_misfolding" not in contrasts:
                    print("Hinweis: der präregistrierte Kontrast Kultur/Politik ↔ Fehlfaltung braucht die Familien F13–F16 und F19–F20 (FAMILIES = None).")
                print("Erklärter Anteil (nicht z) der Domänenachsen-Basis pro Layer:")
                show(pooled[pooled["basis"] == "domain_axes"].pivot_table(index="contrast", columns="layer", values="explained"))
                stab = table("J", "rdx_stability")
                if len(stab):
                    print("Stabilität der Erklärung über Familien / Familienpaare:")
                    show(stab, n=40)
            """
        ),
        code(
            """
            # --- Sharing-Index und Netzwerkaktivität (DESIGN §4.8c) -------------------------------------
            with guarded("Sharing"):
                sh = table("J", "sharing")
                if sh.empty:
                    print("Keine Sharing-Tabelle (kein SAE: USE_SAE=True auf step143000; im Rauchtest ein Platzhalter-SAE).")
                else:
                    print("SAE-Quelle:", sh["sae_source"].iloc[0], "(synthetic_random = Platzhalter: Funktionstest, kein Befund)")
                    js = sh[sh["kind"] == "sharing"]
                    conds = sorted((set(js["a"]) | set(js["b"].dropna())) - {"global"})
                    M = pd.DataFrame(np.nan, index=conds, columns=conds, dtype=float)
                    for _, r in js.iterrows():
                        if r["a"] in conds and r["b"] in conds:
                            M.loc[r["a"], r["b"]] = float(r["jaccard"])
                            M.loc[r["b"], r["a"]] = float(r["jaccard"])
                    act = sh[sh["kind"] == "activity"].set_index("a")[["frac_active", "mean_mass", "participation_ratio"]].astype(float)
                    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
                    im = a1.imshow(M.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
                    a1.set_xticks(range(len(conds)))
                    a1.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
                    a1.set_yticks(range(len(conds)))
                    a1.set_yticklabels(conds, fontsize=8)
                    for i in range(len(conds)):
                        for j in range(len(conds)):
                            if np.isfinite(M.iat[i, j]):
                                a1.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
                    a1.set_title("Sharing-Index S = Jaccard der aktiven TopK-Features (paarweise)", fontsize=9)
                    fig.colorbar(im, ax=a1, shrink=0.8)
                    (act / act.max()).plot(kind="bar", ax=a2, width=0.8)
                    a2.set_title("Netzwerkaktivität A pro Bedingung (relativ zum Maximum der Spalte)", fontsize=9)
                    a2.set_xlabel("Bedingung")
                    a2.legend(fontsize=7)
                    savefig("sharing_activity")
                    plt.show()
                    show(act)
                    glob = js[js["a"] == "global"]
                    if len(glob):
                        print("Globaler Sharing-Index (alle Bedingungen):", float(glob["jaccard"].iloc[0]))
                    hp = sh[sh["kind"] == "heads"]
                    if len(hp):
                        print("Kosinus der Head-Beitragsprofile zwischen Bedingungen (Layer-Raster):")
                        show(hp.pivot_table(index=["a", "b"], columns="layer", values="cosine"), n=40)
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 10 test-time training (stage K)
# ======================================================================================
def nb_10() -> list[tuple[str, str]]:
    return [
        header_cell(
            "10_ttt_injection", "K", "Test-Time-Training als strukturelle Injektion",
            purpose=("`docs/DESIGN.md` §4.8 d: fortgesetztes Training eines eingefrorenen Checkpoints auf einem kleinen "
                     "Kontrastkorpus (Ethereum/DAO-Fork 2016, vom Auftraggeber unter `data/ttt/dao_fork` bereitzustellen) "
                     "mit drei Kontrollarmen gleichen Budgets (`neutral`: thematisch neutral, gleiche Größe; `nofork`: "
                     "gleiche Ära ohne Spaltungsnarrativ; `permuted`: Tokens des DAO-Korpus permutiert). Gemessen werden "
                     "(i) die Memorisationsgeschwindigkeit pro Textstück (Schritte bis zur NLL-Schwelle) über Checkpoints, "
                     "(ii) Δ Sharing-Index und Δ Netzwerkaktivität und (iii) Δ T(K=8), Δ B(x) sowie Head- und Geometrie-Maße "
                     "vor/nach dem Training auf demselben Stimulusraster. Das übergebene Modell bleibt unverändert "
                     "(Kopien); ein Effekt zählt nur, wenn er über allen drei Kontrollarmen liegt."),
            inputs=("`data/ttt/dao_fork`, `data/ttt/neutral`, `data/ttt/nofork` (*.txt / *.jsonl, siehe `data/ttt/README.md`; "
                    "fehlen sie, läuft der gemeinfreie Platzhalter mit `is_placeholder = True` — Funktionstest, kein Befund), "
                    "Stimulusbank, Richtungsbank, SAE (oder Platzhalter), optional Stufe G/I derselben Sitzung "
                    "(Wolken, schlechtester Head)."),
            outputs="Tabellen `ttt_corpora`, `ttt_memorisation`, `ttt_memorisation_boot`, `ttt_training`, `ttt_before_after`, `ttt_effects`.",
            steps=("Setup → Drive → Konfiguration → Stufe-K-Parameter (Korpus, Checkpoints) → Modell → [Stufen G, I] → Stufe K → "
                   "Korpora → Memorisation → Trainingsverlauf → Vorher/Nachher-Effekte → Zusammenfassung."),
            prereg=("Kontrast 17: (i) `ttt_memorisation_boot`, `arm = injection-<Kontrolle>`, `quantity = steps_to_threshold` "
                    "(zweiseitig); (ii) `ttt_effects`, `measure ∈ {sharing_jaccard, activity_*}`; (iii) `ttt_effects`, "
                    "`measure ∈ {T_K8, B_lang_dom}` — jeweils `exceeds_all_controls`. Explorativ auf Entwicklungsfamilien; "
                    "Bestätigung auf Testfamilien mit eingefrorener Schrittzahl und Lernrate."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("10_ttt_injection", "K"),
        code(
            """
            # --- Stufe-K-Parameter: Korpus, Kontrollarme, Checkpoints ------------------------------------
            TTT_INPUTS = None            # z. B. "/content/drive/MyDrive/LSNE/inputs/ttt": Ordner dao_fork/, neutral/, nofork/ werden nach data/ttt/ kopiert
            TTT_CORPUS = None            # Injektionskorpus (Standard data/ttt/dao_fork; Platzhalter mit is_placeholder=True, wenn leer)
            TTT_CHECKPOINTS = None       # z. B. ["step4000", "step98000", "step143000"]: Memorisationstabelle über Checkpoints (Shard 7)
            WITH_CONTEXT_STAGES = False  # True: Stufen G und I vorher in dieser Sitzung rechnen (Wolken, schlechtester Head -> Vorher/Nachher-Maße)
            RUNS_SUBDIR = None           # z. B. "runs_ttt" (Shard 7): eigenes Laufverzeichnis <OUT_ROOT>/runs_ttt statt RUNS_ROOT

            import shutil
            from dataclasses import replace

            if RUNS_SUBDIR:
                _old = RUNS_ROOT
                RUNS_ROOT = os.path.join(OUT_ROOT, RUNS_SUBDIR)
                os.makedirs(RUNS_ROOT, exist_ok=True)
                cfg = replace(cfg, out_root=RUNS_ROOT)
                CLI_COMMAND = CLI_COMMAND.replace(f"--out {_old}", f"--out {RUNS_ROOT}")
                print("Laufverzeichnis:", RUNS_ROOT)
            if TTT_INPUTS:
                for arm in ("dao_fork", "neutral", "nofork"):
                    src_dir = os.path.join(TTT_INPUTS, arm)
                    if os.path.isdir(src_dir):
                        shutil.copytree(src_dir, os.path.join("data", "ttt", arm), dirs_exist_ok=True)
                        print("kopiert:", src_dir, "-> data/ttt/" + arm)
            for arm, p in (("injection", TTT_CORPUS or limits.k_corpus), ("neutral", "data/ttt/neutral"), ("nofork", "data/ttt/nofork")):
                files = [f for f in (os.listdir(p) if os.path.isdir(p) else []) if f.endswith((".txt", ".jsonl"))]
                print(f"{arm:9s} {p}: {len(files)} Datei(en)" + ("" if files else "  -> PLATZHALTER (Funktionstest, kein Befund)"))
            if SMOKE:
                TTT_CHECKPOINTS = None
            print(f"Training: {limits.k_train_steps} Schritte × Batch {limits.k_train_batch}, lr {limits.k_train_lr}; "
                  f"Memorisation: {limits.k_mem_steps} Schritte, lr {limits.k_mem_lr}, Schwelle {limits.k_threshold} nats, "
                  f"{limits.k_max_snippets} Stücke/Arm; Checkpoints: {TTT_CHECKPOINTS or 'nur ' + REVISION}; B(x)-Dosis {limits.k_eps}")
            if TTT_CORPUS:
                CLI_COMMAND += f" --ttt-corpus {TTT_CORPUS}"
            if TTT_CHECKPOINTS:
                CLI_COMMAND += f" --ttt-checkpoints {','.join(TTT_CHECKPOINTS)}"
            """
        ),
        model_cell(),
        runner_cell(),
        substages_cell(),
        code(
            """
            # --- Stufe K ausführen (optional nach G und I für die Kontext-Maße) ------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage K --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> \\
            #       --ttt-corpus data/ttt/dao_fork [--ttt-checkpoints step4000,step98000,step143000]
            # (--stage G,I,K additionally yields log_volume / participation_ratio of the clouds and the
            # head_delta of the worst head in ttt_before_after, because K reads them from ctx)
            if WITH_CONTEXT_STAGES:
                run_stage("G", reuse=False)
                run_stage("I", reuse=False)
            res_K = run_stage("K", ttt_corpus=TTT_CORPUS, ttt_checkpoints=TTT_CHECKPOINTS)
            print({k: len(v) for k, v in res_K.items()})
            manifest_K = substage_report("K")
            """
        ),
        code(
            """
            # --- Korpora der vier Arme (Platzhalter werden laut markiert) ---------------------------------
            with guarded("Korpora"):
                corp = table("K", "ttt_corpora")
                show(corp)
                if len(corp) and truthy(corp["is_placeholder"]).any():
                    print("ACHTUNG: mindestens ein Arm läuft auf dem gemeinfreien Platzhalter (is_placeholder = True) -> Funktionstest, "
                          "kein Befund. DAO-Fork-Korpus und Kontrollkorpora unter data/ttt/ ablegen (data/ttt/README.md) oder TTT_INPUTS setzen.")
            """
        ),
        code(
            """
            # --- Kontrast 17 (i): Memorisationsgeschwindigkeit — Schritte bis NLL ≤ Schwelle pro Arm und Checkpoint --
            with guarded("Memorisation"):
                mem = table("K", "ttt_memorisation")
                mbt = table("K", "ttt_memorisation_boot")
                arms = [a for a in ["injection", "neutral", "nofork", "permuted"] if a in set(mbt["arm"])]
                titles = {"steps_to_threshold": f"Schritte bis NLL ≤ {limits.k_threshold} (nur erreichte Stücke)",
                          "auc_drop": "AUC des NLL-Abfalls", "final_nll": "NLL nach dem Training"}
                fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
                for ax, q in zip(axes, ["steps_to_threshold", "auc_drop", "final_nll"]):
                    sub = mbt[(mbt["quantity"] == q) & (mbt["arm"].isin(arms))]
                    ck = list(dict.fromkeys(sub["checkpoint"]))
                    x = np.arange(len(ck))
                    w = 0.8 / max(1, len(arms))
                    for i, arm in enumerate(arms):
                        g = sub[sub["arm"] == arm].set_index("checkpoint").reindex(ck)
                        ax.bar(x + (i - (len(arms) - 1) / 2) * w, g["mean"], w,
                               yerr=[(g["mean"] - g["ci_low"]).clip(lower=0).fillna(0), (g["ci_high"] - g["mean"]).clip(lower=0).fillna(0)],
                               capsize=2, label=arm)
                    ax.set_xticks(x)
                    ax.set_xticklabels(ck)
                    ax.set_title(titles[q], fontsize=9)
                axes[0].legend(fontsize=7)
                savefig("ttt_memorisation")
                plt.show()
                diff = mbt[mbt["arm"].astype(str).str.startswith("injection-")]
                if len(diff):
                    print("Injektion minus Kontrolle (gepaart über Stücke, Bootstrap-CI; negativ bei steps_to_threshold = schneller memorisiert):")
                    show(diff)
                if len(mem):
                    print("Anteil der Stücke, die die Schwelle erreichen, NLL am Start und am Ende pro Arm:")
                    show(mem.assign(reached=truthy(mem["reached_threshold"]).astype(float)).groupby(["checkpoint", "arm"])
                         .agg(n=("snippet", "count"), reached=("reached", "mean"), nll_start=("nll_start", "mean"), final_nll=("final_nll", "mean")))
            """
        ),
        code(
            """
            # --- Trainingsverlauf der vier Arme (gleiches Budget: Schritte, Lernrate, Batch, Seed) ----------
            with guarded("Trainingsverlauf"):
                tr = table("K", "ttt_training")
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
                for arm, g in tr.groupby("arm"):
                    g = g.sort_values("step")
                    a1.plot(g["step"], g["loss"], label=arm)
                    a2.plot(g["step"], g["grad_norm"], label=arm)
                a1.set_xlabel("Schritt")
                a1.set_ylabel("Verlust [nats/Token]")
                a1.set_title("Test-Time-Training (AdamW, Gradienten-Clipping 1,0, FP32)", fontsize=9)
                a2.set_xlabel("Schritt")
                a2.set_ylabel("Gradientennorm")
                a2.set_yscale("log")
                a1.legend(fontsize=7)
                savefig("ttt_training")
                plt.show()
            """
        ),
        code(
            """
            # --- Kontrast 17 (ii)+(iii): Δ nach TTT pro Arm — ein Effekt zählt nur über allen drei Kontrollarmen --
            with guarded("Vorher/Nachher"):
                eff = table("K", "ttt_effects")
                corp = table("K", "ttt_corpora")
                arms = [a for a in ["injection", "neutral", "nofork", "permuted"] if a in set(eff["arm"])]
                agg = eff.groupby(["measure", "arm"])["delta"].mean().unstack("arm").reindex(columns=arms)
                order = [m for m in ["T_K8", "B_lang_dom", "logp_target", "sharing_jaccard", "activity_frac_active", "activity_mean_mass",
                                     "activity_participation_ratio", "head_delta_mean", "log_volume", "participation_ratio"] if m in agg.index]
                agg = agg.reindex(order)
                fig, axes = plt.subplots(1, len(order), figsize=(2.4 * len(order) + 1, 3.6), squeeze=False)
                for ax, m in zip(axes[0], order):
                    ax.bar(range(len(arms)), agg.loc[m].to_numpy(dtype=float), color=["C3" if a == "injection" else "C7" for a in arms])
                    ax.set_xticks(range(len(arms)))
                    ax.set_xticklabels(arms, rotation=60, fontsize=7)
                    ax.axhline(0, color="k", lw=0.6)
                    ax.set_title(m, fontsize=8)
                axes[0, 0].set_ylabel("Δ = nach − vor (Mittel über Bedingungen)")
                fig.suptitle("Wirkung des Test-Time-Trainings auf alles Übrige (rot: Injektionsarm, grau: Kontrollarme)", fontsize=10)
                savefig("ttt_effects")
                plt.show()
                inj = eff[eff["arm"] == "injection"]
                if len(inj):
                    print("Injektionsarm: liegt der Effekt über allen drei Kontrollen?")
                    show(inj[["measure", "condition", "before", "after", "delta", "exceeds_all_controls"]], n=60)
                    n_exc = int(truthy(inj["exceeds_all_controls"]).sum())
                    note = " (Platzhalterkorpora: identische Arme, erwartungsgemäß 0)" if len(corp) and truthy(corp["is_placeholder"]).all() else ""
                    print(f"{n_exc} von {len(inj)} (Maß, Bedingung)-Zeilen über allen Kontrollen{note}")
                missing = [m for m in ("head_delta_mean", "log_volume") if m not in set(eff["measure"])]
                if missing:
                    print("Nicht gemessen (Stufen G/I nicht in dieser Sitzung; WITH_CONTEXT_STAGES=True oder CLI --stage G,I,K):", missing)
            """
        ),
        summary_cell(),
    ]


# ======================================================================================
# 11 cross-model transplant (stage L)
# ======================================================================================
def nb_11() -> list[tuple[str, str]]:
    return [
        header_cell(
            "11_transplant", "L", "Cross-Modell-Transplantation",
            purpose=("`docs/DESIGN.md` §4.9: modellagnostische Maschinerie mit steckbarem Quelladapter (`lsne.transplant`), "
                     "validiert mit einem Stand-in-Quellmodell, bevor ein BlockGPT-artiges Spurenmodell eingehängt wird "
                     "(`docs/BLOCKGPT_FEASIBILITY.md`: es gibt keine BlockGPT-Gewichte; empfohlen ist ein eigenes Modell auf "
                     "Xatu-2016-Traces, `docs/TRACE_SOURCE_SPEC.md`). Schritte: gepaarte Eingaben (englische/deutsche "
                     "Einheitssätze der Familien), lineare Abbildung Quelle → Ziel (Procrustes nach PCA) mit Permutations-Null, "
                     "Layer-Matching, Transfer der im Quellmodell bestimmten Sprachachse und **funktionaler Transfertest** "
                     "(Kosinus zur nativen Achse, Verschiebung des deutschen Anteils in Fortsetzungen), dann Injektion A "
                     "(Residual-Substitution, kein Addieren), B (KV-Cache-Pflanzung über ein virtuelles Präfix auf allen Layern) "
                     "und C (beides) gegen Zufalls-, Identitäts- und Permutationskontrollen."),
            inputs=("Stimulusbank (Satzpaare, K=0-Prompts), Quellmodell `TRANSPLANT_SOURCE` (`rotated_self`: Standard-Stand-in, "
                    "`self`, oder ein HF-Modell wie `EleutherAI/pythia-70m@step143000`; ein Spurenmodell über dieselbe "
                    "`SourceModel`-Schnittstelle), Generierungsbudget `cfg.gen_n_new × cfg.gen_n_samples`."),
            outputs="Tabellen `transfer_test`, `transfer_summary`, `layer_matching`, `alignments`, `transplant`, `transplant_summary`.",
            steps=("Setup → Drive → Konfiguration → Stufe-L-Parameter → Modell → Stufe L → Transfertest pro Layer → Layer-Matching → "
                   "Transplantation A/B/C gegen Kontrollen → Zusammenfassung."),
            prereg=("Kontrast 18 (`transfer_summary`: Alignment-R² Hold-out − permutiert ≥ 0,2 `r2_gap_ok`, Kosinus transferierte "
                    "vs. native Sprachachse ≥ 0,3 im besten Layer `cos_ok`, Vorzeichengleichheit der f_D-Verschiebung "
                    "`sign_agreement`, zusammen `validated`; `transplant_summary.sign_agrees_with_native` pro Modus). Erst nach "
                    "bestandener Stand-in-Validierung wird ein Spurenmodell als Quelle gewertet; dessen „Teilen“-Merkmal wird "
                    "vorab im Quellmodell definiert und eingefroren."),
        ),
        setup_cell(),
        tnco_cell(False),
        drive_cell(),
        config_cell("11_transplant", "L"),
        code(
            """
            # --- Stufe-L-Parameter: Quellmodell ------------------------------------------------------------
            TRANSPLANT_SOURCE = None   # "rotated_self" (Standard: Stand-in, das Alignment muss eine Isometrie rückgängig machen) | "self" | "EleutherAI/pythia-70m@step143000"
            print("Quellmodell:", TRANSPLANT_SOURCE or limits.l_source, "| Satzpaare je Sprache:", limits.l_n_texts, "| Prompts:", limits.l_n_prompts,
                  "| KV-Präfix:", limits.l_n_prefix, "| Dosis:", limits.l_eps, "| Permutations-Null:", limits.l_n_null)
            try:
                import lsne.transplant  # noqa: F401
                print("lsne.transplant verfügbar")
            except Exception as e:   # the stage then writes a manifest with stub=True and no tables
                print("lsne.transplant NICHT importierbar -> Stufe L schreibt nur ein Manifest (stub):", repr(e))
            if TRANSPLANT_SOURCE:
                CLI_COMMAND += f" --transplant-source {TRANSPLANT_SOURCE}"
            """
        ),
        model_cell(),
        runner_cell(),
        substages_cell(),
        code(
            """
            # --- Stufe L ausführen ----------------------------------------------------------------
            # CLI-Äquivalent:
            #   python -m lsne.cli run --stage L --model EleutherAI/pythia-160m --revision step98000 --out <RUNS_ROOT> \\
            #       [--transplant-source EleutherAI/pythia-70m@step143000]
            res_L = run_stage("L", transplant_source=TRANSPLANT_SOURCE)
            print({k: len(v) for k, v in res_L.items()})
            manifest_L = substage_report("L")
            if manifest_L:
                print("Quelle:", manifest_L.get("source"), "| Satzpaare:", manifest_L.get("n_texts"), "| validiert:", manifest_L.get("validated"))
            """
        ),
        code(
            """
            # --- Kontrast 18: funktionaler Transfertest pro Layer des Rasters (Stand-in-Validierung) -------
            with guarded("Funktionaler Transfertest"):
                tt = table("L", "transfer_test")
                ts = table("L", "transfer_summary")
                if tt.empty:
                    print("Keine Transfer-Tabelle (Stub oder Substufe fehlgeschlagen; siehe Substufen-Bericht).")
                else:
                    tt = tt.sort_values("layer_tgt")
                    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
                    axes[0].plot(tt["layer_tgt"], tt["cos_transferred_vs_native"], "o-", label="transferierte vs. native Sprachachse")
                    axes[0].errorbar(tt["layer_tgt"], tt["cos_null_vs_native"], yerr=tt["cos_null_std"].fillna(0), fmt="s--", color="grey",
                                     capsize=2, label="permutierte Paare (Null)")
                    axes[0].plot(tt["layer_tgt"], tt["cos_null_max"], ":", color="grey", label="Null-Maximum")
                    axes[0].axhline(0.3, color="r", lw=0.8, ls="--", label="Mindesteffekt cos ≥ 0,3")
                    axes[0].set_title("Kosinus im Zielmodell", fontsize=9)
                    axes[1].plot(tt["layer_tgt"], tt["r2_holdout"], "o-", label="R² Hold-out")
                    axes[1].plot(tt["layer_tgt"], tt["r2_permuted"], "s--", color="grey", label="R² permutierte Paare")
                    axes[1].plot(tt["layer_tgt"], tt["r2_gap"], "^:", color="C2", label="Differenz (Mindestabstand 0,2)")
                    axes[1].axhline(0.2, color="r", lw=0.8, ls="--")
                    axes[1].set_title("Alignment-Güte (Procrustes nach PCA)", fontsize=9)
                    w = 0.4
                    axes[2].bar(tt["layer_tgt"] - w / 2, tt["delta_fD_transferred"], w, label="Δf_D transferierte Achse")
                    axes[2].bar(tt["layer_tgt"] + w / 2, tt["delta_fD_native"], w, label="Δf_D native Achse")
                    axes[2].axhline(0, color="k", lw=0.6)
                    axes[2].set_title("Verschiebung des deutschen Anteils in Fortsetzungen", fontsize=9)
                    for ax in axes:
                        ax.set_xlabel("Ziel-Layer (Block-Eingang)")
                        ax.legend(fontsize=7)
                    savefig("transfer_test")
                    plt.show()
                    show(tt[["layer_tgt", "layer_src", "cos_transferred_vs_native", "z_cos_vs_null", "r2_holdout", "r2_permuted", "r2_gap",
                             "sign_agreement", "readout_alive", "r2_gap_ok", "cos_ok", "validated"]])
                    if len(ts):
                        s = ts.iloc[0]
                        print(f"Bester Layer {s['best_layer']}: validiert = {s['validated']} ({s['n_validated_layers']} von {s['n_layers']} Layern). "
                              "Erst nach bestandener Validierung wird ein Spurenmodell als Quelle gewertet.")
            """
        ),
        code(
            """
            # --- Layer-Matching: R² Hold-out über Quell- × Ziel-Layer; Alignments je Ziel-Layer (KV-Pflanzung) --
            with guarded("Layer-Matching"):
                lm = table("L", "layer_matching")
                al = table("L", "alignments")
                piv = lm.pivot_table(index="layer_src", columns="layer_tgt", values="r2_holdout")
                fig, ax = plt.subplots(figsize=(0.6 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 2))
                im = ax.imshow(piv.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(piv.shape[1]))
                ax.set_xticklabels(piv.columns)
                ax.set_yticks(range(piv.shape[0]))
                ax.set_yticklabels(piv.index)
                ax.set_xlabel("Ziel-Layer (Pythia)")
                ax.set_ylabel("Quell-Layer")
                ax.set_title("R² (Hold-out) der linearen Abbildung Quelle → Ziel", fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.8)
                savefig("layer_matching")
                plt.show()
                if len(al):
                    print("Alignments je Ziel-Layer (pflanzen das KV-Präfix auf jedem Layer):")
                    show(al)
            """
        ),
        code(
            """
            # --- Transplantation: Residual-Substitution (A), KV-Pflanzung (B), beides (C) gegen die Kontrollen --
            with guarded("Transplantation"):
                tp = table("L", "transplant")
                tsum = table("L", "transplant_summary")
                modes = [m for m in ["residual", "kv", "both"] if m in set(tp["mode"])]
                controls = [c for c in ["none", "random", "identity", "permuted_alignment"] if c in set(tp["control"])]
                fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
                for ax, q, lab in zip(axes, ["fD", "logp_target"], ["f_D: deutscher Anteil der Fortsetzung", "ln p(Ziel)"]):
                    piv = tp.groupby(["mode", "control"])[q].mean().unstack("control").reindex(modes)[controls]
                    piv.plot(kind="bar", ax=ax, width=0.8)
                    ax.set_title(lab, fontsize=9)
                    ax.set_xlabel("Modus (none = Transplantat; Kontrollen: random, identity, permuted_alignment)")
                    ax.legend(fontsize=7)
                savefig("transplant_modes")
                plt.show()
                if len(tsum):
                    show(tsum)
                    print("Vorzeichen der f_D-Verschiebung gegen die Identitätskontrolle stimmt mit der nativen Achse überein:",
                          dict(zip(tsum["mode"], tsum["sign_agrees_with_native"])))
            """
        ),
        summary_cell(),
    ]


BUILDERS = {
    "00_setup_and_anchor": nb_00,
    "01_baseline_and_directions": nb_01,
    "02_switching_grid": nb_02,
    "03_nonequilibrium": nb_03,
    "04_tn_linear_response": nb_04,
    "05_scale_and_checkpoints": nb_05,
    "06_geometry_trajectory": nb_06,
    "07_lambda_dynamics": nb_07,
    "08_compass_heads": nb_08,
    "09_rdx_sharing": nb_09,
    "10_ttt_injection": nb_10,
    "11_transplant": nb_11,
}


# ======================================================================================
# assembly
# ======================================================================================
def build_notebook(stem: str) -> nbformat.NotebookNode:
    """Assemble one notebook (deterministic ids / metadata, validated)."""
    stage, title = NOTEBOOKS[stem]
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": f"{stem}.ipynb", "provenance": []},
        "lsne": {"generator": "scripts/build_notebooks.py", "stage": stage, "title": title,
                 "git_url": GIT_URL, "git_branch": GIT_BRANCH},
    }
    cells = []
    for i, (kind, src) in enumerate(BUILDERS[stem]()):
        cell = new_markdown_cell(src) if kind == "markdown" else new_code_cell(src)
        cell["id"] = f"cell-{i:02d}"
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    nb.cells = cells
    nbformat.validate(nb)
    return nb


def build_all() -> dict[str, nbformat.NotebookNode]:
    return {stem: build_notebook(stem) for stem in NOTEBOOKS}


def notebook_json(nb: nbformat.NotebookNode) -> str:
    return nbformat.writes(nb, version=4)


def write_all(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, nb in build_all().items():
        p = out_dir / f"{stem}.ipynb"
        p.write_text(notebook_json(nb) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def normalised(nb: nbformat.NotebookNode) -> dict:
    """Notebook content without outputs / execution counts (for drift checks)."""
    d = json.loads(notebook_json(nb))
    for c in d.get("cells", []):
        c.pop("outputs", None)
        c.pop("execution_count", None)
        c.get("metadata", {}).pop("executionInfo", None)
    return d


def check(out_dir: Path) -> list[str]:
    """Names of committed notebooks that differ from the regenerated ones (or are missing)."""
    drift = []
    for stem, nb in build_all().items():
        p = Path(out_dir) / f"{stem}.ipynb"
        if not p.exists():
            drift.append(f"{p}: fehlt")
            continue
        old = nbformat.read(p, as_version=4)
        if normalised(old) != normalised(nb):
            drift.append(f"{p}: weicht vom Generator ab")
    return drift


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(NOTEBOOK_DIR), help="output directory (default: notebooks/)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the notebooks in --out drift")
    args = ap.parse_args(argv)
    if args.check:
        drift = check(Path(args.out))
        for d in drift:
            print(d)
        print("ok: notebooks are up to date" if not drift else f"{len(drift)} notebook(s) out of date")
        return 1 if drift else 0
    for p in write_all(Path(args.out)):
        nb = nbformat.read(p, as_version=4)
        print(f"{p}  ({len(nb.cells)} Zellen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
