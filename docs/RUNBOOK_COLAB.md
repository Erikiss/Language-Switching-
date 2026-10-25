# Runbook Colab: zwei Konten, sechs A100-Shards

Schritt-für-Schritt-Anleitung für den vollständigen Rechenplan aus `jobs/shards.yaml`
(DESIGN §7) auf zwei Colab-Konten (Pro+/Enterprise, je bis zu drei gleichzeitige
A100-Sitzungen). Alles, was hier steht, lässt sich auch mit der Kommandozeile
(`python -m lsne.cli`) auf einer anderen GPU-Maschine ausführen (Abschnitt 8).

## 1. Voraussetzungen

| | Konto A (`colab-A`) | Konto B (`colab-B`) |
|---|---|---|
| Sitzungen | 3 × A100 (40/80 GB) | 3 × A100 |
| Shards | 1, 2, 3 | 4, 5, 6 |
| Drive | `MyDrive/LSNE` (wird vom Notebook angelegt) | `MyDrive/LSNE` |
| Secrets (optional) | `HF_TOKEN` (nur bei Rate-Limits der Hugging-Face-Range-Requests) | `HF_TOKEN` |

- Branch: `claude/language-switching-non-equilibrium-ejpz0q` von
  `https://github.com/Erikiss/Language-Switching-.git` (in jeder Setup-Zelle als `GIT_URL`
  / `GIT_BRANCH` änderbar).
- Laufzeittyp in Colab: *GPU → A100*. Die Notebooks laufen auch auf T4/L4, nur langsamer
  (Shard 6 passt dann nicht in den Speicher; `--dtype bfloat16` ist als eigener Arm mit
  eigener Null-Dosis-Baseline vorgesehen, DESIGN §6).
- Kein Datensatz-Download: der Anker (4 KB) und die Unigramm-Zählung (200 Sequenzen à
  4 KB) kommen per HTTP-Range-Request und werden unter `MyDrive/LSNE/cache` gecacht
  (`LSNE_CACHE`, von der Drive-Zelle gesetzt).

## 2. Drive-Ordnerlayout

```
MyDrive/LSNE/
├── runs/                                   = RUNS_ROOT  (CLI: --out /content/drive/MyDrive/LSNE/runs)
│   ├── A/pythia-160m@step98000/            manifest.json + *.csv/*.parquet      (Shard 1)
│   ├── B/pythia-160m@step98000/            + directions.npz, directions_meta.json
│   ├── C/…, D/…, E/…
│   ├── A/pythia-160m@step99000/ …          (Shard 2)
│   ├── A/pythia-160m@step143000/ …         (Shard 3, mit sae_features.csv in B)
│   ├── A/pythia-410m@step143000/ …         (Shard 5)
│   ├── F/pythia-160m@step143000/           plan.csv, runs.csv                   (Shard 4)
│   │   └── shard0of1/B/pythia-160m@step4000/, …/D/pythia-160m@step4000/, … (27 Checkpoints)
│   ├── F/pythia-1b@step143000/             plan.csv, runs.csv                   (Shard 6)
│   │   └── shard0of1/B/pythia-1b@step143000/, …/pythia-1.4b@step143000/
│   └── merged/                             Ausgabe von `lsne.cli merge`
├── cache/state2143/, cache/unigram/        Anker + Unigramm-Zählungen (LSNE_CACHE)
├── figures/<notebook>/*.png                Abbildungen der Notebooks
├── summary_<notebook>_<modell>_<rev>_<dtype>.json
└── progress_05_scale_and_checkpoints_shard<N>.json
```

Jede `manifest.json` trägt Git-SHA, Modell, dtype, Hook-Site, Anker-Hashes, Laufzeit und
`status` (`ok` / `error`); ein Fehler hinterlässt zusätzlich `error.txt`.

## 3. Welches Notebook auf welcher Sitzung

| Sitzung | Konto | Shard | Notebook(s) | Einstellungen in der Konfigurationszelle |
|---|---|---|---|---|
| 1 | A | 1 | `00` → `01` → `02` → `03` → `04` nacheinander | `MODEL="EleutherAI/pythia-160m"`, `REVISION="step98000"` (Standard) |
| 2 | A | 2 | `00` → … → `04` | `REVISION="step99000"` |
| 3 | A | 3 | `00` → … → `04` | `REVISION="step143000"`, `USE_SAE=True` |
| 4 | B | 4 | `05` | `SHARD=4` (27 Checkpoints, Stufen B+D limitiert) |
| 5 | B | 5 | `05` mit `SHARD=5` **oder** `00` → … → `04` mit `MODEL="EleutherAI/pythia-410m"`, `REVISION="step143000"` | 410M; SAE nur MLP-Hookpoints, hier nicht genutzt |
| 6 | B | 6 | `05` | `SHARD=6` (pythia-1b, pythia-1.4b; B/C/D limitiert; nur C1) |

Empfehlung: Sitzung 1 zuerst interaktiv mit den Notebooks 00–04 fahren (dort entstehen
die Abbildungen und die Abbruchkriterien der Stufen A–C sind sichtbar); Sitzungen 2, 3 und
5 können alternativ mit Notebook `05` und `SHARD=2/3/5` in einem Durchlauf gerechnet werden
(gleiche Laufverzeichnisse, gleicher Resume-Schutz, keine Abbildungen — die liefern die
Notebooks 00–04 später mit `REUSE=True` aus den gespeicherten Tabellen).

**Reihenfolge innerhalb einer Sitzung.** Die Stufen hängen nur lose voneinander ab: B
baut die Richtungsbank und speichert sie (`directions.npz`); C, D und E laden sie aus dem
Stufe-B-Verzeichnis derselben (Modell, Revision), wenn die Signatur passt (gleiche
Varianten, gleicher Seed, gleiche Zufallszahl), sonst bauen sie sie neu. Notebook 00
ist nur für den Anker-/Stimulus-Bericht nötig, nicht als Voraussetzung.

**Pro Notebook:** alle Zellen ausführen (*Laufzeit → Alle ausführen*). Die zweite Zelle
klont bzw. aktualisiert das Repository, installiert `requirements.txt` und das Paket;
die Drive-Zelle fragt die Berechtigung ab; danach läuft die Stufe ohne Interaktion.
Die TNCO-Zelle ist nur in Notebook 04 standardmäßig aktiv (`INSTALL_TNCO=True`).

## 4. Fortsetzen nach Abbruch (Resume)

Colab trennt Sitzungen nach Inaktivität oder nach 24 h. Nichts geht verloren, weil jede
Stufe ihre Tabellen direkt nach Drive schreibt:

1. Notebook neu öffnen, Laufzeit neu verbinden, alle Zellen ausführen.
2. `REUSE=True` (Standard): `run_stage()` findet `manifest.json` mit `status: ok` im
   Laufverzeichnis und lädt die CSV-Tabellen statt neu zu rechnen; die Plot-Zellen laufen
   dann in Sekunden. Eine abgebrochene Stufe hat kein `manifest.json` (oder `status:
   error`) und wird neu gerechnet. `REUSE=False` erzwingt die Neuberechnung.
3. Notebook 05: die Schleife prüft vor jedem Paar (Modell, Revision) jede Stufe; Stufen
   mit `status: ok` werden als `skipped_done` protokolliert, das Modell wird nur geladen,
   wenn noch etwas fehlt. `progress_05_…_shard<N>.json` in `MyDrive/LSNE` zeigt den Stand
   auch ohne Notebook.
4. Fein-Aufteilung einer Shard: `SHARD_INDEX, SHARD_COUNT = 0, 2` auf der einen und
   `1, 2` auf der anderen Sitzung (Zuordnung deterministisch über die sortierten Tripel
   (Modell, Revision, Familie), `plan.csv`); `merge` fügt beide Hälften zusammen.
5. Ein Stufenfehler (Traceback in der Ausgabe, `error.txt` im Laufverzeichnis) stoppt das
   Notebook nicht; die restlichen Zellen laufen mit leeren Tabellen weiter und melden
   „übersprungen“. Nach der Korrektur die Stufe mit `REUSE=False` wiederholen.

## 5. Zusammenführen (`lsne.cli merge`)

Beide Konten haben getrennte Drives. Zum Zusammenführen:

1. Auf Konto B den Ordner `MyDrive/LSNE/runs` für Konto A freigeben (Leserecht); auf
   Konto A „Verknüpfung zu Meine Ablage hinzufügen“ — der Ordner erscheint nach dem Mount
   z. B. als `/content/drive/MyDrive/runs-B` (oder über *Shared with me*:
   `/content/drive/Othercomputers/…`). Alternativ die Laufverzeichnisse per
   `drive.google.com` herunterladen und auf Konto A hochladen.
2. In einer Colab-Zelle auf Konto A die Läufe von B in das eigene `runs/` kopieren (die
   Verzeichnisnamen überschneiden sich nicht, da Modell/Revision/Shard im Pfad stehen):
   ```bash
   rsync -a --ignore-existing /content/drive/MyDrive/runs-B/ /content/drive/MyDrive/LSNE/runs/
   ```
3. Zusammenführen:
   ```bash
   python -m lsne.cli merge --root /content/drive/MyDrive/LSNE/runs --out /content/drive/MyDrive/LSNE/runs/merged
   ```
   `merge` konkateniert alle gleichnamigen CSV-Dateien unterhalb von `--root` (beliebige
   Tiefe, auch die Stufe-F-Teilläufe aller Shards) nach `merged/<name>.csv` mit den
   vorangestellten Spalten `run_id` (Pfad relativ zur Wurzel), `run_stage`, `run_model`,
   `run_revision`, sammelt alle Manifeste in `merged/manifests.json` und schreibt
   `merged/merge_index.csv` (Tabelle, Zeilen, Läufe). Die Auswertung (Familien-Bootstrap
   über alle Shards, Phasendiagramme über Checkpoints) arbeitet auf `merged/`.
   Notebook 05 ruft `merge_runs` für die eigene Shard am Ende automatisch auf.

## 6. Erwartete Laufzeiten (A100, FP32, alle 16 Familien × A/B)

| Shard | Stufen | Schätzung | Treiber |
|---|---|---|---|
| 1, 2 (160M) | A: < 1 min; B: 3–5 min; C: 10–20 min; D: 45–90 min; E: 5–15 min | 1–3 h | D3–D7: 64 Tokens × 8 Samples × Rampen/Quench/Konstantdosen für die Kernvarianten; D1: ~50 000 Forwards |
| 3 (160M + SAE) | wie 1 + SAE-Download (~130 MB) + FVU-Messung | 1–3 h | zusätzliche Richtungen `sae_D*`/`sae_E*` in C und D |
| 4 (Trace, 27 Checkpoints) | B + D (limitiert) je Checkpoint, ~5–8 min | 2–4 h | 27 × Modell-Download (~330 MB) + D limitiert (8 Zufallsrichtungen, nur `lang_dom` in der Generierung) |
| 5 (410M) | A–E | 1–3 h | 24 Blöcke, d = 1024; Stufe E mit r = 64, T_w = 32 |
| 6 (1b, 1.4b) | B, C, D limitiert; nur C1 (6 Varianten/Formulierung) | 3–6 h | Modellgröße; Generierung dominiert |
| Rauchtest (CPU, pythia-14m) | alle | < 3 min | `SMOKE=True` bzw. `--smoke` |

Die Zahlen sind Schätzungen aus DESIGN §7 (ein Forward von 160M auf ≤ 320 Tokens ≈ 3 ms);
die tatsächlichen Laufzeiten stehen in jeder `manifest.json` (`runtime_s`) und in
`runs.csv` von Stufe F.

## 7. Wenn TNCO nicht baut

TNCO (`pip install "tnco @ git+https://github.com/google-research/tnco"`) braucht die
Boost-Header (`apt-get install -y libboost-dev`), einen C++-Compiler und einige Minuten.
Typische Fehler: fehlendes `libboost-dev` (Build bricht mit `boost/...hpp: No such file`
ab), zu alte `pip`/`setuptools` (`pip install -U pip setuptools wheel` und erneut
versuchen), oder ein Python-ABI-Wechsel nach einem Colab-Update.

Was dann passiert: **nichts Schlimmes.** `lsne.tn_lattice.optimize_path(tn, backend="tnco")`
fängt `ImportError` und jeden TNCO-Fehler ab und nutzt `opt_einsum` („dp“ für ≤ 12
Tensoren, sonst „greedy“); in `tn_paths.csv` steht dann `backend_used = opt_einsum` und
`fallback_reason` nennt die Ursache. Die kontrahierte Antwort (c) ist identisch, nur die
Spalten `cost_log10` / `runtime_s` des Pfadvergleichs fehlen für TNCO. Stufe E gilt damit
als vollständig; der TNCO-vs-opt_einsum-Vergleich kann später auf einer Maschine mit
funktionierendem Build nachgeholt werden (`REUSE=False`, nur Stufe E; ≈ 5–15 min).

## 8. Alternative ohne Colab: gpu.ai (oder jede andere GPU-Maschine) über die CLI

1. Instanz mit CUDA-PyTorch-Image starten (A100 40 GB reicht für alle Shards; Shard 6
   braucht ≥ 24 GB), SSH einloggen.
2. Einrichten:
   ```bash
   git clone --branch claude/language-switching-non-equilibrium-ejpz0q https://github.com/Erikiss/Language-Switching-.git
   cd Language-Switching- && pip install -r requirements.txt && pip install -e .
   apt-get install -y libboost-dev && pip install "tnco @ git+https://github.com/google-research/tnco"   # optional
   export LSNE_CACHE=/data/lsne/cache LSNE_RUNS=/data/lsne/runs HF_TOKEN=...                           # optional
   python -m lsne.cli run --stage all --model EleutherAI/pythia-14m --revision step98000 --device cpu --families DEV01 --out /tmp/smoke --smoke
   ```
3. Shard-Befehl aus `jobs/shards.yaml` kopieren und `--out` ersetzen, z. B. Shard 4:
   ```bash
   nohup python -m lsne.cli run --stage F --model EleutherAI/pythia-160m --revision step143000 \
       --dtype float32 --device auto --out /data/lsne/runs --seed 0 \
       --pairs trace --f-stages B,D --shard-index 0 --shard-count 1 > shard4.log 2>&1 &
   ```
   Mehrere GPUs auf einer Maschine: `CUDA_VISIBLE_DEVICES=0 … --shard-index 0 --shard-count 2`
   und `CUDA_VISIBLE_DEVICES=1 … --shard-index 1 --shard-count 2`.
4. Fortsetzen: die CLI überspringt nichts automatisch; für Stufe F genügt es, den Befehl
   mit `--pairs <fehlende Revisionen>` erneut zu starten (z. B. `--pairs step98000,step99000`),
   für A–E eine Stufenliste (`--stage D,E`). Die Laufverzeichnisse werden ergänzt, nicht
   gelöscht.
5. Ergebnisse zurück: `rsync -a /data/lsne/runs/ <lokal>/runs/` (oder `rclone` nach Drive)
   und wie in Abschnitt 5 zusammenführen.

## 9. Fehlersuche

| Symptom | Ursache / Abhilfe |
|---|---|
| `State 2143 anchor verification failed` | Range-Request lieferte eine andere Sequenz (Proxy, Datensatz-Revision). Hashes in `lsne.config.STATE2143` vergleichen; nie mit falschem Text weiterrechnen (Präregistrierung §1). |
| `range request failed: HTTP 429/403` | Hugging-Face-Rate-Limit: `HF_TOKEN` in den Colab-Secrets setzen und in der Setup-Zelle `os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")` ergänzen; der Cache verhindert Wiederholungen. |
| `unigram counts could not be fetched` (Warnung) | Offline oder Rate-Limit: `freq` wird aus der Stimulusbank gebildet (`bank.meta["unigram"]["source"] == "stimulus_bank"`), in `direction_meta.csv` dokumentiert. |
| `CUDA out of memory` | Generierungsbudget senken (CLI `--gen-n-samples 4`, im Notebook `cfg.gen_n_samples = 4` vor `run_stage`), für Stufe E `TN_RANK`/`TN_WINDOW` verkleinern, für 1.4b `--dtype bfloat16` (eigener Arm mit eigener Null-Dosis-Baseline). |
| `SAE directions skipped: no published SAE …` | Für dieses Modell/Site gibt es keinen SAE (410M: nur MLP-Hookpoints). `USE_SAE=False` lassen. |
| Stufe D sehr langsam auf T4 | Generierung dominiert: `FAMILIES` einschränken oder `--limit-variants`; der Rauchtest (`SMOKE=True`) prüft nur die Pipeline. |
| Notebook weicht vom Repository ab | Notebooks werden generiert: `python scripts/build_notebooks.py`; `tests/test_notebooks.py` schlägt bei Drift fehl. |
