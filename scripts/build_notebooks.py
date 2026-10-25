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
                steps: str) -> tuple[str, str]:
    return md(
        """
        # LS-NEQ {{STEM}} — {{TITLE}}

        **Dossier-Stufe {{STAGE}}** (`lsne.experiments.STAGES["{{STAGE}}"]`, siehe `docs/DESIGN.md` §8).
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

        Hinweise: Die Zellen sind gegen Fehler abgesichert (ein fehlgeschlagener Schritt druckt
        den Traceback und das Notebook läuft weiter). Mit `SMOKE = True` läuft alles in
        < 3 min auf CPU mit `pythia-14m` (Rauchtest, keine wissenschaftliche Aussage).
        Mit `REUSE = True` werden vorhandene Ergebnisse (manifest.json mit `status: ok`)
        geladen statt neu gerechnet — so lässt sich nach einem Colab-Abbruch fortsetzen.
        """,
        STEM=stem, TITLE=title, STAGE=stage, PURPOSE=purpose, INPUTS=inputs, OUTPUTS=outputs, STEPS=steps,
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


BUILDERS = {
    "00_setup_and_anchor": nb_00,
    "01_baseline_and_directions": nb_01,
    "02_switching_grid": nb_02,
    "03_nonequilibrium": nb_03,
    "04_tn_linear_response": nb_04,
    "05_scale_and_checkpoints": nb_05,
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
