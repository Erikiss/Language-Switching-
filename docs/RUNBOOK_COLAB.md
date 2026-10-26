# Runbook Colab: zwei Konten, sieben A100-Shards

Schritt-für-Schritt-Anleitung für den vollständigen Rechenplan aus `jobs/shards.yaml`
(DESIGN §7; Stufen A–F des Dossiers und die Erweiterungsstufen G–L aus `docs/STAGES.md`)
auf zwei Colab-Konten (Pro+/Enterprise, je bis zu drei gleichzeitige A100-Sitzungen).
Alles, was hier steht, lässt sich auch mit der Kommandozeile (`python -m lsne.cli`) auf
einer anderen GPU-Maschine ausführen (Abschnitt 8).

## 1. Voraussetzungen

| | Konto A (`colab-A`) | Konto B (`colab-B`) |
|---|---|---|
| Sitzungen | 3 × A100 (40/80 GB) | 3 × A100 |
| Shards | 1, 2, 3 (A–E und G–L) | 4, 5, 6; danach 7 (Stufe K über drei Checkpoints) auf einer frei gewordenen Sitzung |
| Drive | `MyDrive/LSNE` (wird vom Notebook angelegt) | `MyDrive/LSNE` |
| Secrets (optional) | `HF_TOKEN` (nur bei Rate-Limits der Hugging-Face-Range-Requests) | `HF_TOKEN` |
| Eingaben des Auftraggebers (optional) | `MyDrive/LSNE/inputs/ttt/{dao_fork,neutral,nofork}` (Stufe K), `inputs/atlas/*.txt` (Stufe I), `inputs/s4/axes.npz` (Stufe B) | wie A (Shard 7 braucht `inputs/ttt`) |

- Branch: `claude/language-switching-non-equilibrium-ejpz0q` von
  `https://github.com/Erikiss/Language-Switching-.git` (in jeder Setup-Zelle als `GIT_URL`
  / `GIT_BRANCH` änderbar).
- Laufzeittyp in Colab: *GPU → A100*. Die Notebooks laufen auch auf T4/L4, nur langsamer
  (Shard 6 passt dann nicht in den Speicher; `--dtype bfloat16` ist als eigener Arm mit
  eigener Null-Dosis-Baseline vorgesehen, DESIGN §6).
- Kein Datensatz-Download: der Anker (4 KB) und die Unigramm-Zählung (2 000 Sequenzen à
  4 KB) kommen per HTTP-Range-Request und werden unter `MyDrive/LSNE/cache` gecacht
  (`LSNE_CACHE`, von der Drive-Zelle gesetzt); Stufe J zieht auf dieselbe Weise 2 000
  Pile-Sequenzen für die Schwierigkeitsrangfolge (`cache/pile_nll`).
- Die Notebooks klonen das Repository frisch nach `/content/lsne-repo`; vom Auftraggeber
  gelieferte Eingaben (DAO-Korpus, Atlas-Extraktion, S4-Achsen) liegen deshalb auf Drive
  unter `MyDrive/LSNE/inputs/` und werden in der Parameterzelle des jeweiligen Notebooks
  angegeben (`TTT_INPUTS`, `ATLAS_DIR`, `AXIS_NPZ`); Notebook 10 kopiert `inputs/ttt/*`
  in den Checkout (`data/ttt/`), weil die Kontrollarme dort gelesen werden. Fehlen die
  Eingaben, laufen die Stufen mit Platzhaltern bzw. Proxys und markieren das
  (`is_placeholder`, leeres `atlas_overlap`) — Funktionstests, keine Befunde.

## 2. Drive-Ordnerlayout

```
MyDrive/LSNE/
├── runs/                                   = RUNS_ROOT  (CLI: --out /content/drive/MyDrive/LSNE/runs)
│   ├── A/pythia-160m@step98000/            manifest.json + *.csv/*.parquet      (Shard 1)
│   ├── B/pythia-160m@step98000/            + directions.npz, directions_meta.json
│   ├── C/…, D/…, E/…
│   ├── G/…, H/…, I/…, J/…, K/…, L/…        Erweiterungsstufen (Shard 1; I: + directions_domain.npz;
│   │                                       G-L: manifest.json → substages, prereg; substage_errors.txt bei Fehlern)
│   ├── A/pythia-160m@step99000/ …          (Shard 2, ebenfalls A-E und G-L)
│   ├── A/pythia-160m@step143000/ …         (Shard 3, mit sae_features.csv in B; H/J/K mit dem publizierten SAE)
│   ├── A/pythia-410m@step143000/ …         (Shard 5)
│   ├── F/pythia-160m@step143000/           plan.csv, runs.csv                   (Shard 4)
│   │   └── shard0of1/B/pythia-160m@step4000/, …/D/pythia-160m@step4000/, … (27 Checkpoints)
│   ├── F/pythia-1b@step143000/             plan.csv, runs.csv                   (Shard 6)
│   │   └── shard0of1/B/pythia-1b@step143000/, …/pythia-1.4b@step143000/
│   └── merged/                             Ausgabe von `lsne.cli merge`
├── runs_ttt/K/pythia-160m@step98000/       Shard 7: Stufe K mit --ttt-checkpoints (getrennt vom Kernlauf)
├── inputs/ttt/{dao_fork,neutral,nofork}/   Eingaben des Auftraggebers (optional; nie im Repository)
├── inputs/atlas/*.txt, inputs/s4/axes.npz
├── cache/state2143/, cache/unigram/, cache/pile_nll/   Anker, Unigramm-Zählungen, Pile-NLL (LSNE_CACHE)
├── figures/<notebook>/*.png                Abbildungen der Notebooks
├── summary_<notebook>_<modell>_<rev>_<dtype>.json
└── progress_05_scale_and_checkpoints_shard<N>.json
```

Jede `manifest.json` trägt Git-SHA, Modell, dtype, Hook-Site, Anker-Hashes, Laufzeit und
`status` (`ok` / `error`); ein Fehler hinterlässt zusätzlich `error.txt`. Die Manifeste der
Stufen G–L enthalten außerdem `substages` (Laufzeit und Fehler je Substufe) und `prereg`
(welche Tabelle jeden präregistrierten Kontrast trägt; Gesamtkarte in `docs/STAGES.md`).

## 3. Welches Notebook auf welcher Sitzung

| Sitzung | Konto | Shard | Notebook(s) | Einstellungen in der Konfigurations-/Parameterzelle |
|---|---|---|---|---|
| 1 | A | 1 | `00` → `01` → `02` → `03` → `04` → `06` → `07` → `08` → `09` → `10` → `11` nacheinander | `MODEL="EleutherAI/pythia-160m"`, `REVISION="step98000"` (Standard); `08`: `ATLAS_DIR`, `10`: `TTT_INPUTS`, `11`: `TRANSPLANT_SOURCE="EleutherAI/pythia-70m@step143000"` |
| 2 | A | 2 | `00` → … → `04` → `06` → … → `11` | `REVISION="step99000"` (+ dieselben Parameter wie Sitzung 1) |
| 3 | A | 3 | `00` → … → `04` → `06` → … → `11` | `REVISION="step143000"`, `USE_SAE=True` (Stufen B–D: `sae_*`-Richtungen; H, J, K: publizierter SAE statt Platzhalter) |
| 4 | B | 4 | `05` | `SHARD=4` (27 Checkpoints, Stufen B+D limitiert) |
| 5 | B | 5 | `05` mit `SHARD=5` **oder** `00` → … → `04` mit `MODEL="EleutherAI/pythia-410m"`, `REVISION="step143000"` | 410M; SAE nur MLP-Hookpoints, hier nicht genutzt; G–L laufen nicht auf 410M |
| 6 | B | 6 | `05` | `SHARD=6` (pythia-1b, pythia-1.4b; B/C/D limitiert; nur C1) |
| 7 | B | 7 (nach 4 oder 6) | `10` | `TTT_INPUTS`, `TTT_CHECKPOINTS=["step4000", "step98000", "step143000"]`, `RUNS_SUBDIR="runs_ttt"` (eigenes Laufverzeichnis, damit der Stufe-K-Lauf von Shard 1 nicht per `REUSE` geladen wird) |

Empfehlung: Sitzung 1 zuerst interaktiv mit den Notebooks 00–04 fahren (dort entstehen
die Abbildungen und die Abbruchkriterien der Stufen A–C sind sichtbar), dann 06–11;
Sitzungen 2, 3 und 5 können alternativ mit Notebook `05` und `SHARD=2/3/5` in einem
Durchlauf gerechnet werden (gleiche Laufverzeichnisse, gleicher Resume-Schutz, keine
Abbildungen — die liefern die Notebooks 00–04 später mit `REUSE=True` aus den
gespeicherten Tabellen; für G–L stattdessen den Shard-Befehl aus `jobs/shards.yaml` in
einer Zelle ausführen oder die Notebooks 06–11 nacheinander laufen lassen).

**Reihenfolge innerhalb einer Sitzung.** Die Stufen hängen nur lose voneinander ab: B
baut die Richtungsbank und speichert sie (`directions.npz`); C, D, E und G–L laden sie aus
dem Stufe-B-Verzeichnis derselben (Modell, Revision), wenn die Signatur passt (gleiche
Varianten, gleicher Seed, gleiche Zufallszahl), sonst bauen sie sie neu. Notebook 00
ist nur für den Anker-/Stimulus-Bericht nötig, nicht als Voraussetzung. Zwei weiche
Abhängigkeiten der Erweiterungsstufen: Stufe G liest für T5 (Kopplung) die Tabellen
`damage_profile` und `lens` des Stufe-D-Laufs unter `RUNS_ROOT` (Notebook 03 zuerst,
sonst bleibt `coupling` leer); Stufe K misst `head_delta` und die Wolkengeometrie nur,
wenn G und I in derselben Sitzung liefen (`WITH_CONTEXT_STAGES=True` in Notebook 10 oder
CLI `--stage G,I,K`), alle übrigen Maße immer.

**Pro Notebook:** alle Zellen ausführen (*Laufzeit → Alle ausführen*). Die zweite Zelle
klont bzw. aktualisiert das Repository, installiert `requirements.txt` und das Paket;
die Drive-Zelle fragt die Berechtigung ab; danach läuft die Stufe ohne Interaktion.
Die TNCO-Zelle ist nur in Notebook 04 standardmäßig aktiv (`INSTALL_TNCO=True`).
Die Notebooks 06–11 drucken nach der Stufe einen Substufen-Bericht (fehlgeschlagene
Substufen mit Fehlertext und Hinweis auf `substage_errors.txt`) und die
Präregistrierungs-Karte; die Abbildungen jeder Stufe (G0-Gate, Kontraktion, λ-Pfade,
Domänenprofil, RDX-z-Karten, Memorisation, Transfertest, …) liegen unter
`figures/<notebook>/`.

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
6. Stufen G–L: eine fehlgeschlagene *Substufe* (z. B. die Pile-Rangfolge ohne Netz) lässt
   nur ihre Tabelle leer, die Stufe endet mit `status: ok` und listet den Fehler unter
   `substages.errors` (Traceback in `substage_errors.txt`). `REUSE=True` würde diesen Lauf
   wiederverwenden; nach der Korrektur deshalb die Stufe mit `REUSE=False` neu rechnen.

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
| 1–3, Stufen G–L (160M) | G: 10–20 min; H: 30–60 min; I: 10–20 min; J: 10–20 min (+ 2 000 Range-Requests beim ersten Lauf); K: 10–20 min; L: 5–15 min | 1–2 h zusätzlich | H: ≈ 110 000 Forward+Backward-Pässe der λ-Pfade (`H_N_STEPS` / `H_MAX_STARTS` senken auf kleiner GPU); G: Realisierungswolken × Snapshots; K: vier Trainingsarme à 200 Schritte |
| 7 (TTT über Checkpoints) | K mit `--ttt-checkpoints step4000,step98000,step143000` | 1–2 h | 3 × Modell-Download (~330 MB) + Memorisationstabelle je Checkpoint |
| Rauchtest (CPU, pythia-14m) | alle A–L | 1–2 min unbelastet (Testbudget 20 min) | `SMOKE=True` bzw. `--smoke` |

Die Zahlen sind Schätzungen aus DESIGN §7 (ein Forward von 160M auf ≤ 320 Tokens ≈ 3 ms)
und `jobs/shards.yaml`; die tatsächlichen Laufzeiten stehen in jeder `manifest.json`
(`runtime_s`, für G–L auch je Substufe unter `substages.seconds`) und in `runs.csv` von
Stufe F.

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
   Shard 1 (A–E und G–L; Eingaben des Auftraggebers vorher nach `data/ttt/` und
   `data/atlas/` kopieren):
   ```bash
   nohup python -m lsne.cli run --stage A,B,C,D,E,G,H,I,J,K,L --model EleutherAI/pythia-160m --revision step98000 \
       --dtype float32 --device auto --out /data/lsne/runs --seed 0 \
       --ttt-corpus data/ttt/dao_fork --transplant-source EleutherAI/pythia-70m@step143000 > shard1.log 2>&1 &
   ```
   Mehrere GPUs auf einer Maschine: `CUDA_VISIBLE_DEVICES=0 … --shard-index 0 --shard-count 2`
   und `CUDA_VISIBLE_DEVICES=1 … --shard-index 1 --shard-count 2`.
4. Fortsetzen: die CLI überspringt nichts automatisch; für Stufe F genügt es, den Befehl
   mit `--pairs <fehlende Revisionen>` erneut zu starten (z. B. `--pairs step98000,step99000`),
   für A–E und G–L eine Stufenliste (`--stage D,E` bzw. `--stage H,I`). Die
   Laufverzeichnisse werden ergänzt, nicht gelöscht. Die CLI druckt am Ende die Dauer jeder
   Stufe; G–L laufen auf einem Modell in der Reihenfolge G H I J K L und reichen Wolken
   (G → H, K) und Domänenachsen / schlechtesten Head (I → J, K) über `ctx` weiter.
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
| `Substufen fehlgeschlagen: n` im Bericht von Notebook 06–11 | Fehlertext im Bericht, Traceback in `substage_errors.txt` des Laufverzeichnisses; die übrigen Tabellen sind gültig. Nach der Korrektur die Stufe mit `REUSE=False` neu rechnen. |
| `is_placeholder = True` in `ttt_corpora` (Notebook 10) | `data/ttt/dao_fork` (bzw. `neutral`, `nofork`) leer: `TTT_INPUTS` auf den Drive-Ordner setzen oder die Korpora in den Checkout kopieren (`data/ttt/README.md`). Platzhalterläufe sind Funktionstests, keine Befunde. |
| `pile_ranking` leer / `LSNE_OFFLINE is set` (Stufe J) | Die Schwierigkeitsrangfolge braucht Range-Requests; `LSNE_OFFLINE` entfernen bzw. Rate-Limit abwarten (`HF_TOKEN`); gezogene Sequenzen werden unter `cache/pile_nll` gecacht. |
| `g0_gate.degenerate = True` (Stufe G) | Zu wenige gepoolte Zustände für den Mahalanobis-Rang (`r_eff ≥ N−1`): mehr Familien (`FAMILIES=None`) und alle Formulierungen A–D; im Rauchtest erwartet. |
| `sae_source = synthetic_random` in H/J/K | Kein publizierter SAE für dieses Modell/diese Site; ein zufälliger Platzhalter-SAE prüft nur die Codepfade. Für SAE-Befunde Shard 3 (`step143000`, `USE_SAE=True`). |
| Stufe L: `STUB: lsne.transplant nicht importierbar` oder `validated = False` | Der Stand-in `rotated_self` muss den Transfertest bestehen, bevor ein Spurenmodell als Quelle gewertet wird (`docs/BLOCKGPT_FEASIBILITY.md`); ein HF-Quellmodell wird mit `TRANSPLANT_SOURCE="EleutherAI/pythia-70m@step143000"` geladen (Download ~160 MB). |
| Notebook 10 lädt per `REUSE` den Stufe-K-Lauf von Shard 1 statt Shard 7 zu rechnen | `RUNS_SUBDIR="runs_ttt"` setzen (eigenes Laufverzeichnis) oder `REUSE=False`. |
