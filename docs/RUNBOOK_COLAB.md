# Runbook Colab: zwei Konten, sieben A100-Shards

Schritt-für-Schritt-Anleitung für den vollständigen Rechenplan aus `jobs/shards.yaml`
(DESIGN §7; Stufen A–F des Dossiers und die Erweiterungsstufen G–L aus `docs/STAGES.md`)
auf zwei Colab-Konten (Pro+/Enterprise, je bis zu drei gleichzeitige A100-Sitzungen).
Alles, was hier steht, lässt sich auch mit der Kommandozeile (`python -m lsne.cli`) auf
einer anderen GPU-Maschine ausführen (Abschnitt 8). Ausnahme: Der Kontrolllauf von Notebook 14
ist nur auf der Plattform des Hauptlaufs von Stufe D gültig (A100, dieselbe Torch- und
transformers-Version; Abschnitt 12).

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
├── runs_ctl/D/pythia-160m@step98000/       Notebook 14: Kontrolllauf von Stufe D nach Nachtrag 6.11 (damage_ctl*,
│                                           manifest.json mit control = token_language); getrennt vom Hauptlauf in runs/D/…
├── runs_ctl/logs/                          Logs der CLI-Aufrufe von Notebook 14 (Vorabprüfung, Kontrolllauf; bleiben
│                                           nach Sitzungsende erhalten)
├── smoke_14/runs/, smoke_14/runs_ctl/      nur Rauchtest von Notebook 14 (SMOKE = True): eigener Rauch-Hauptlauf und
│                                           Kontrollwurzel; die Standardwurzeln von Notebook 13 sehen ihn nicht
├── prereg_eval/pythia-160m@step98000/      Notebook 13: prereg_results.{csv,md}, prereg_runs.csv, prereg_meta.json
│   └── archive/<zeitstempel>/              frühere Auswertungen (ein neuer Lauf verschiebt die vorige hierher, Abschnitt 11)
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
7. Stufen C und D setzen *innerhalb* der Stufe fort: Grid, Attribution und Gradient ×
   Aktivierung (C) bzw. D1, D2/D9 und D3–D7 (D) werden je 50 Varianten unter
   `<Laufverzeichnis>/_resume/` gesichert. Bricht die Stufe ab (Sitzungsende, Fehler in
   einer späteren Substufe), rechnet der nächste Start mit denselben Eingaben (Varianten,
   Richtungsbank, Dosen, Modell, Limits) nur die fehlenden Blöcke; das Manifest zählt die
   geladenen Blöcke unter `resumed_checkpoints`. Nach `status: ok` wird `_resume/`
   gelöscht. Geänderte Eingaben verwerfen die alten Blöcke automatisch.

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

## 6. Erwartete Laufzeiten (A100, FP32, alle 96 Formulierungen aus 24 Familien, 1 152 Varianten + Original)

| Shard | Stufen | Schätzung | Treiber |
|---|---|---|---|
| 1, 2 (160M) | A: < 1 min; B: 3–5 min (+ einmalig die Unigramm-Zählung, s. u.); C: 3,05 h gemessen mit dem alten sequenziellen Grid, mit dem gebatchten Grid geschätzt 20–40 min; D: D1 ≈ 20–40 min (geschätzt) + D3–D7 1–2 h; E: 5–15 min | 2–4 h | C: ≈ 568 000 Grid-Zeilen (35 Richtungen × 2 Vorzeichen × 7 Dosen je Variante) in Forwards zu 64 Zeilen + ≈ 59 000 Attributions-Forwards; D1: 1 153 Varianten × 2 Injektionen × 204 (Richtung, Dosis)-Paare in Forwards zu 64 Zeilen; D3–D7: 64 Tokens × 8 Samples × Rampen/Quench/Konstantdosen für die Kernvarianten |
| 3 (160M + SAE) | wie 1 + SAE-Download (~130 MB) + FVU-Messung | 2–4 h | zusätzliche Richtungen `sae_D*`/`sae_E*` in C und D |
| 4 (Trace, 27 Checkpoints) | B + D (limitiert) je Checkpoint | ≈ 1–1,5 h je Checkpoint mit vollem Generierungskern (25–45 h); mit dem auf 24 Formulierungen je (K, C)-Zelle begrenzten Kern (`Limits.gen_max_core`, im limitierten Modus Standard) geschätzt 30–50 min je Checkpoint (≈ 15–25 h) | 27 × Modell-Download (~330 MB) + D limitiert (8 Zufallsrichtungen, nur `lang_dom` in der Generierung); passt nicht in eine Colab-Sitzung: auf 3–4 Sitzungen mit disjunkten `FORCE_PAIRS` verteilen (`REUSE=True`) |
| 5 (410M) | A–E | 1–3 h | 24 Blöcke, d = 1024; Stufe E mit r = 64, T_w = 32 |
| 6 (1b, 1.4b) | B, C, D limitiert; nur C1 (6 Varianten/Formulierung) | 3–6 h | Modellgröße; Generierung dominiert |
| 1–3, Stufen G–L (160M) | G: 0,4–1,2 h; H: 0,7–1,6 h mit dem alten vollen Forward je Schritt, mit dem Teil-Forward ab Block L kürzer; I: 0,2–1,1 h; J: 0,3–1,3 h (davon 10–30 min für 2 000 Range-Requests beim ersten Lauf); K: 0,15–0,6 h; L: 0,1–0,3 h | 2–6 h zusätzlich | Audit vom 26.10.2026: Forward-Zahlen gezählt, Zeit je Forward an der gemessenen Stufe C kalibriert (≈ 15–18 ms je sequenziellem 160M-Forward statt der früher angenommenen 3 ms); G ≈ 59 000 Forwards, H ≈ 100 000 Forward+Backward-Pässe der λ-Pfade, I ≈ 33 000 Forwards; nicht auf einer GPU gemessen |
| 7 (TTT über Checkpoints) | K mit `--ttt-checkpoints step4000,step98000,step143000` | 1–2 h | 3 × Modell-Download (~330 MB) + Memorisationstabelle je Checkpoint |
| Rauchtest (CPU, pythia-14m) | alle A–L | 1–2 min unbelastet (Testbudget 20 min) | `SMOKE=True` bzw. `--smoke` |

Die Zahlen sind Schätzungen aus DESIGN §7 (ein Forward von 160M auf ≤ 320 Tokens ≈ 3 ms)
und `jobs/shards.yaml`; gemessen ist nur die alte Stufe C (3,05 h). Die Angaben für das
gebatchte Grid (C), das gebatchte D1 und den begrenzten Generierungskern (Shard 4) sind
noch nicht auf einer GPU gemessen. Die tatsächlichen Laufzeiten stehen in jeder
`manifest.json` (`runtime_s`, für G–L auch je Substufe unter `substages.seconds`) und in
`runs.csv` von Stufe F.

**Unigramm-Zählung (Frequenzachse `freq`).** Jede Stufe, die eine Richtungsbank baut (B,
und D, F, wenn sie keine passende Bank von B finden), braucht beim ersten Mal auf einem
Konto 2 000 Pile-Sequenzen per Range-Request: mit 8 Threads und Wiederholung bei
429/5xx/Abbruch etwa 3 min (früher sequenziell 15–45 min), fortsetzbar über
`unigram_n2000_s0.partial.npz`. Notebook 00 holt sie vorab; auf dem zweiten Konto
entweder Notebook 00 laufen lassen oder `MyDrive/LSNE/cache/unigram/unigram_n2000_s0.npz`
kopieren, bevor die parallelen Notebooks starten. Scheitert der Abruf in einem vollen
Lauf, bricht die Bank mit `UnigramFetchError` ab, statt still auf die Zählung der
Stimulusbank auszuweichen (die deutsche Funktionswörter enthält); `LSNE_OFFLINE=1` erlaubt
diesen Ersatz ausdrücklich. Die Quelle steht in den Manifesten von B, C und D (`unigram`).

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

## 10. Spurenmodell 2016 auf der A100 (Notebook 12, Quelle für Stufe L)

`notebooks/12_trace_source.ipynb` ist der volle Lauf des Quelladapters aus
`docs/TRACE_SOURCE.md` (dort nur auf der CPU als Rohrleitung validiert): alle
Xatu-Chunks 2016 (Blöcke 778 000–2 912 999) nach Drive, Serialisierung, Vokabular aus dem
Hash-Trainingsteil mit Hold-out-OOV, Training `blockgpt` in bf16, Evaluation gegen
Unigramm/Bigramm/untrainiert, die kontrollierten Sonden, das Alignment mit
Längenresidualisierung und Stufe L mit `trace:<modell>` gegen `pythia-160m`. Daten:
ethPandaOps Xatu, **CC BY 4.0** (Namensnennung „Data: ethPandaOps Xatu
(https://ethpandaops.io), CC BY 4.0“; gilt auch für Shards, Vokabular und Modell).

**Voraussetzung (Präregistrierung 18).** Das Spurenmodell zählt erst als Quelle, wenn der
Stand-in-Lauf von Notebook 11 mit demselben `MODEL`/`REVISION` bestanden ist: maßgeblich ist
`runs/L/<modell>@<rev>/verdict.csv` (Kontrast 18: `validated` True — Transfer-Kriterien und
Vorzeichengleichheit für Residual und KV — und `counts_for_prereg` True; Selbstkonsistenz-Quellen
`self`/`rotated_self` geben kein Urteil), nicht `transfer_summary.validated`; zudem muss der
Stand-in-Lauf mit dem Code ab 27.10.2026 geschrieben sein (Manifest: Prompts reihum über die
Familien, `v_src` je Quell-Layer — ein älterer Lauf mit eps 1,0 und Prompts aus einer Familie zählt
nicht: **Notebook 11 zuerst neu laufen lassen**, dann Notebook 12). Abschnitt 8a prüft
das, Stufe L prüft es selbst (Option `standin` → `trace_verdict.counts_for_prereg`), und die
Zusammenfassung vermerkt es (`standin_validated`, `stage_L.counts_for_prereg`). Das
Merkmal „Teilen“ (Netto-fan-out ≥ 5 gegen 1, längengematcht, letzter Snapshot, Kriterium
trainiert − untrainiert) wird beim ersten Ausführen der Konfigurationszelle nach
`trace_2016/frozen_feature.json` geschrieben; eine spätere Änderung wird erkannt und macht
Sonden und Stufe L explorativ (strengere Ergänzungen der Entscheidungsregel, etwa seit 27.10.2026
`beats_surface`, stehen mit Datum unter `amendments` und ändern das Merkmal nicht). Erwarteter Ausgang nach dem CPU-Lauf: „no claim“ (das
untrainierte Modell trennt die Sonden genauso gut) — ein Ergebnis, kein Fehler. Die Pipeline-Zellen
brechen bei einem gescheiterten Schritt oder einer fehlenden Voraussetzung ab, statt weiterzulaufen
(`docs/TRACE_SOURCE.md` §13); nach dem ersten A100-Lauf genügt es, mit `REUSE = True` alle Zellen erneut
auszuführen (veraltete Schritte rechnen neu, das Modell mit dem leeren Vokabular wird beiseitegelegt).

**Sitzungen.**

| Sitzung | Laufzeittyp | Schritte | Dauer (Plan) | Einstellungen |
|---|---|---|---|---|
| 1 | CPU mit hohem RAM (oder gleich A100) | 1 Download, 2 Serialisierung, 3 Split/Vokabular/Hold-out-OOV | ≈ 1–3 h je nach Kernzahl | keine; ohne GPU überspringt das Notebook die Schritte 4–8 mit Hinweis |
| 2 | A100 (40/80 GB) | 4 Durchsatz + Training, 5 Evaluation, 6 Sonden, 7 Paare/Alignment, 8 Stufe L | ≈ 2,3–4,8 h | alle Zellen erneut ausführen; erledigte Schritte werden übersprungen |

Gesamt ≈ 3,3–7,8 h (Planungswerte aus `docs/TRACE_SOURCE.md` §8, Unsicherheit mindestens
Faktor 2; die Tabelle „Ist gegen Plan“ am Ende des Notebooks zeigt die echten Zeiten). Eine
A100 kostet ≈ 12 Recheneinheiten pro Stunde, also ≈ 27–57 CU für Sitzung 2 und ≈ 40–95 CU,
wenn alles auf der A100 läuft (Preise vor dem Lauf in Colab prüfen). Reines Training:
20 000 × 32 × 512 = 328 M Tokens ≈ 1 Epoche 2016, hochgerechnet 0,4–0,75 h bei 15–30 % MFU;
Abschnitt 4a misst den bf16-Durchsatz vor dem Start.

**Drive (≈ 14 GB).**

```
MyDrive/LSNE/
├── cache/xatu/                         ≈ 5,2 GB Parquet (12 810 Dateien; download_log.csv des letzten Bereichs)
├── trace_2016/                         = TRACE_ROOT
│   ├── itr/  itr_train/                ≈ 3,2 GB + ≈ 1,3 GB Shards (gzip-JSONL je Chunk; split_counts.json)
│   ├── vocab/                          Vokabular, Tokenizer, oov.json, oov_heldout.json
│   ├── trace_lm_blockgpt/              Modell, checkpoint-*/ (2 × ≈ 1,2 GB), train_log.csv, manifest.json
│   ├── blocks/                         gepackte Blöcke der Evaluation (≈ 1,4 GB int32)
│   ├── eval.json  pairs_2000.jsonl  frozen_feature.json
│   ├── probes/<sonde>/                 criterion.csv, probe_report.json, match_table.csv, sharing_directions.pt
│   ├── align/                          align_controls.csv, align_summary.json
│   ├── logs/<schritt>.log              Ausgabe jedes CLI-Aufrufs (auch nach Abbruch der Sitzung)
│   └── steps/<schritt>.json            Marker erledigter Schritte (Status, Wandzeit)
├── runs_trace/L/pythia-160m@step98000/ Stufe L mit dem Spurenmodell (getrennt vom Stand-in-Lauf in runs/L/…)
├── figures/12_trace_source/
└── summary_12_trace_source_pythia-160m_step98000_float32.json
```

Das Sondenkorpus (jeder `PROBE_SHARD_EVERY`-te Shard plus die Drain- und Fork-Chunks)
besteht aus symbolischen Links unter `/content/trace_itr_probe` (Drive kann keine Links
speichern) und wird in jeder Sitzung neu angelegt.

**Fortsetzen.** Download: vorhandene vollständige Dateien werden übersprungen, ein
fehlgeschlagener Bereich wird beim nächsten Ausführen wiederholt. Serialisierung: vorhandene
Shards werden übersprungen. Split: je Shard (`split_counts.json`). Training: setzt beim
neuesten vollständigen Checkpoint auf Drive fort (`SAVE_EVERY = 2000`, die Batch-Reihenfolge
hängt nur von Saat und Schritt ab); vor dem ersten Schritt werden alle Shards neu gepackt
(≈ 25 min für 2016). Ein höheres `STEPS` setzt ein fertiges Training fort. Alle übrigen
Schritte: Marker unter `steps/` (`REUSE = True`); `FORCE_STEPS = {"evaluate", ...}` rechnet
gezielt neu. Stufe L: `run_stage` lädt einen fertigen Lauf (`manifest.json`, `status: ok`).

**Ohne Colab (CLI auf einer GPU-Maschine).**

```bash
export OUT=/data/lsne/trace_2016 CACHE=/data/lsne/cache/xatu
for a in $(seq 778000 100000 2899999); do b=$((a + 100000)); [ $b -gt 2913000 ] && b=2913000
  python -m lsne.chain.cli download --from $a --to $b --tables traces,transaction,native_transfers,erc20_transfers,logs,storage_diffs \
      --cache $CACHE --workers 16 --retries 6; done
python -m lsne.chain.cli serialize --cache $CACHE --from 778000 --to 2913000 --out $OUT/itr --workers $(nproc)
ITR=$OUT/itr ITR_TRAIN=$OUT/itr_train VOCAB=$OUT/vocab VAL_FRAC=0.01 N_ADDRESSES=50000 VOCAB_REPORT=0 \
    OUT=$OUT scripts/trace_e2e.sh vocab                      # Hash-Split + vocab (Hold-out-OOV: Notebook-Zelle 3c)
python -m lsne.chain.cli train --in $OUT/itr --vocab $OUT/vocab --size blockgpt --out $OUT/trace_lm_blockgpt \
    --steps 20000 --batch-size 32 --lr 3e-4 --warmup 200 --val-frac 0.01 --eval-every 500 --eval-blocks 512 \
    --save-every 2000 --keep-checkpoints 2 --bf16 on --device auto
python -m lsne.chain.cli probe --model $OUT/trace_lm_blockgpt --in $OUT/itr --kind fanout --k 5 --match length \
    --length-ratio 1.25 --n 2000 --untrained auto --out $OUT/probes/fanout_len_k5        # ebenso k 3, k 2
python -m lsne.chain.cli probe --model $OUT/trace_lm_blockgpt --in $OUT/itr --kind dao --match length \
    --mask all_addresses --n 2000 --untrained auto --out $OUT/probes/dao_len_mask_all    # ebenso --mask dao
python -m lsne.chain.cli pair --in $OUT/itr --n 2000 --balance --out $OUT/pairs_2000.jsonl
python -m lsne.chain.cli align --model $OUT/trace_lm_blockgpt --pairs $OUT/pairs_2000.jsonl \
    --target EleutherAI/pythia-160m@step98000 --alphas 1,100 --out $OUT/align
python -m lsne.cli run --stage L --model EleutherAI/pythia-160m --revision step98000 --out /data/lsne/runs_trace \
    --transplant-source "trace:$OUT/trace_lm_blockgpt,shards=$OUT/itr,k=5,standin=/data/lsne/runs/L/pythia-160m@step98000"
```

Die Spuren-Optionen der Stufe L stehen als `key=value` hinter dem Modellverzeichnis
(`shards`, `direction`, `layer`, `k`, `n_probe`, `n_pairs`, `length_ratio`, `site_layer`,
`n_prompts`, `split`, `standin`, `eps` (Dosis der Transplantation, Voreinstellung `TRACE_EPS` = 1,0,
unabhängig von der Stand-in-Dosis `l_eps`), …; `lsne.cli run` hat keine `--trace-*`-Flags); `standin`
zeigt auf den Stand-in-Lauf (Notebook 11), ohne ihn zählt kein Ergebnis für
Präregistrierung 18. Für Sonden und Paare ohne Trainingsdaten des Spurenmodells:
`probe … --split heldout`, `pair … --split heldout --model $OUT/trace_lm_blockgpt` und in Stufe L
`split=heldout` (der Hash-Split des Trainings, `val_frac` aus dem Modell-Manifest).

Die Evaluation gegen Unigramm/Bigramm/untrainiert hat keinen CLI-Befehl; sie steht im
Notebook (Zelle 5a) und in `scripts/trace_e2e.sh` (Schritte `baseline evaluate untrained`).
Auf der CLI lesen `probe`, `pair` und Stufe L das ganze Korpus; für Speicher und Zeit ein
Verzeichnis mit Links auf jeden 4. Shard übergeben (wie das Notebook).

**Fehlersuche (Notebook 12).**

| Symptom | Ursache / Abhilfe |
|---|---|
| `download: … failed` / `FAILED <tabelle> <start>: HTTP 404` | Der Chunk fehlt bei Xatu für diese Tabelle (wird nicht wiederholt); die Serialisierung nutzt die vorhandenen Tabellen. Netzfehler: Zelle erneut ausführen. |
| Drive voll / `OSError: [Errno 28]` | ≈ 14 GB nötig; alte Checkpoints (`KEEP_CHECKPOINTS = 1`) oder `blocks/` löschen; `cache/xatu` kann nach der Serialisierung archiviert werden. |
| Training startet erst nach ≈ 25 min | Normal: `train` packt alle Shards vor dem ersten Schritt neu (je Start und Fortsetzung). |
| `CUDA out of memory` im Training | `BATCH = 16`, `GRAD_ACCUM = 2` (gleiche Tokens pro Schritt; beim Fortsetzen eines laufenden Trainings warnt `train`, dass die Batch-Reihenfolge dann von einem ununterbrochenen Lauf abweicht). |
| RAM-Abbruch in 5a oder 6a | `BASELINE_MAX_TRAIN_BLOCKS = 200_000` bzw. `PROBE_SHARD_EVERY = 8`. |
| `probe: only N matched pairs … need >= 2` | Zu wenige längengematchte Paare: der Netto-fan-out ≥ 5 (Empfänger mit Netto-Zufluss über die wirksamen Transfers, sichtbar im 512-Token-Fenster) ist 2016 extrem selten (CPU-Stichprobe: 4 von 225 854 Transaktionen); das ist ein Ergebnis und wird so markiert. Stufe L fällt dann auf eine gelockerte Sonde zurück und behauptet nichts (`feature_frozen = False`). |
| `labels_df comes from an older serialisation (fan definition ['value_gt_0'] …)` | Die Shards stammen von vor Fix-Runde 1 (Brutto-fan-out ohne Netto-Flüsse und ohne Fenster-Merkmale, Logs/Diffs am ersten Knoten): `python -m lsne.chain.cli serialize … --overwrite` (bzw. die Serialisierungs-Marker löschen), danach Split, Vokabular und Training neu. |
| `WARNUNG: das eingefrorene Merkmal wurde geändert` | `TRACE_K`, `TRACE_LAYER`, `LENGTH_RATIO`, `TRACE_DIRECTION` oder die fan-out-Definition (`fan_definition`, seit Fix-Runde 1 `net_effective`; `visible_window`) weichen von `frozen_feature.json` ab: der Lauf ist explorativ (Präregistrierung 18). Ein vor Fix-Runde 1 eingefrorenes Merkmal ist damit immer „geändert“; vor dem ersten echten Lauf `frozen_feature.json` bewusst neu schreiben (löschen) und das dokumentieren. |
| 8a: `stand-in run predates the 27.10.2026 fix … rerun notebook 11` | Der Stand-in-Lauf auf Drive stammt aus dem alten Design (eps 1,0, 3 Prompts aus einer Familie, layer-inkonsistentes `v_src`): Notebook 11 mit dem aktuellen Code und demselben `MODEL`/`REVISION` neu laufen lassen, dann Notebook 12 ab Zelle 8a wiederholen. Sonst bleibt `counts_for_prereg` False. |
| Stufe L lädt per `REUSE` einen alten Lauf | Seit Fix-Runde 2 nur, wenn `steps/stage_L.json` ok ist und nach dem Training geschrieben wurde; sonst rechnet 8c neu. Erzwingen: `FORCE_STEPS = {"stage_L"}` oder `REUSE = False`. |
| `StepFailed: [<schritt>] …` / `FEHLGESCHLAGEN`, „Alle ausführen“ stoppt | Gewollt (Fix-Runde 2, `docs/TRACE_SOURCE.md` §13): ein Schritt ist gescheitert oder eine Voraussetzung fehlt bzw. hat nichts erzeugt (`Voraussetzung fehlt: serialize: error …`). Ursache im genannten Log unter `logs/` beheben und ab dieser Zelle erneut ausführen; erledigte Schritte werden übersprungen. Die Zusammenfassung (letzte Zelle) lässt sich einzeln ausführen; `counts_for_prereg` bleibt dann False (`stage_L.steps_not_ok`). |
| `serialize: ERROR N of M chunks missing … MISSING <start>: BrokenProcessPool` | Ein Worker starb (meist Speicher) auch nach den Wiederholungen mit weniger Workern: `SERIALIZE_WORKERS` senken (z. B. 4) oder eine Laufzeit mit mehr RAM wählen und Zelle 2 erneut ausführen; nur die fehlenden Chunks werden serialisiert. |
| `serialize: the shards … were written with other options` | Die vorhandenen Shards stammen von anderen Serialisierungsoptionen: mit `--overwrite` neu schreiben (bzw. `itr/` umbenennen), nie mischen. `--overwrite` entfernt die alten Shards dieser Chunks, bevor es die neuen Optionen schreibt: ein Chunk, der dabei scheitert, fehlt (Code 2, `MISSING …`) und wird beim nächsten Aufruf — auch ohne `--overwrite` — neu geschrieben, statt mit den alten Optionen liegen zu bleiben. |
| `vocab: ERROR … hold 0 transactions` / `no ITR shards` | Der Hash-Split (3a) hat nichts geschrieben: Marker `steps/split.json` und `itr_train/` prüfen; 3a erneut ausführen. |
| `train: ERROR address OOV rate … exceeds 0.5` | Das Vokabular gehört nicht zum Korpus (anderer Split, leeres oder altes Vokabular): 3a/3b neu (`FORCE_STEPS = {"split", "vocab"}`). `MAX_OOV_ADDRESS = 1` schaltet die Prüfung ab (nur zur Diagnose). |
| `trace_lm_blockgpt beiseitegelegt -> ….stale-<Zeit>` | Das vorhandene Modell wurde mit einem anderen Vokabular trainiert (oder `train` erzwungen): es wird nicht fortgesetzt, sondern verschoben; nichts wird gelöscht. Alte `.stale-*`-Verzeichnisse können nach Prüfung gelöscht werden. |

## 11. Präregistrierte Auswertung (Notebook 13, CPU)

`notebooks/13_prereg_evaluation.ipynb` berechnet die konfirmatorischen Endpunkte aus den
Laufverzeichnissen, die schon auf Drive liegen (`lsne.prereg_eval`; Tabellenkarte in
`docs/STAGES.md`, Abschnitt „Pre-registered evaluation“). Es lädt kein Modell; eine
**CPU-Laufzeit** genügt, und der Checkpoint-Sweep (Stufe F) wird nie geöffnet, ein laufendes
Notebook 05 wird also nicht gestört.

1. **Zweites Konto einbinden.** Liegen die Läufe auf zwei Konten, im anderen Konto den Ordner
   `LSNE` freigeben, im eigenen Konto unter „Für mich freigegeben“ → Rechtsklick → „Organisieren“
   → „Verknüpfung hinzufügen“ → „Meine Ablage“ und die Verknüpfung in `LSNE_B` umbenennen.
2. **Notebook öffnen und alle Zellen ausführen.** Die Konfigurationszelle fragt nicht nach den
   Laufwurzeln (`ASK = False`, seit `4365c87`): Sie nimmt die vorhandenen unter
   `MyDrive/LSNE/{runs,runs_trace,runs_ctl}` und `MyDrive/LSNE_B/{runs,runs_trace,runs_ctl}`
   (außerhalb von Colab die vorhandenen unter `$LSNE_RUNS/{runs,runs_trace,runs_ctl}`) und druckt sie mit
   dem CLI-Äquivalent. Andere Wurzeln (z. B. `MyDrive/LSNE/runs_ttt`) komma-getrennt in `RUN_ROOTS_TEXT`
   eintragen; die Liste ersetzt die Standardwurzeln, die gewünschten also mit aufzählen. Mit `ASK = True`
   fragt die Zelle danach, Enter übernimmt die Standardwurzeln.
3. **Welcher Lauf zählt.** Gibt es von einer Stufe mehrere fertige Läufe (`status: ok`), gilt ein voller
   Lauf vor einem Rauch- oder eingeschränkten, unter gleichartigen der neueste (`prereg_runs.csv` nennt
   die übrigen mit Grund). Der Kontrolllauf von Notebook 14 (`runs_ctl`, Abschnitt 12) wird als eigener Schlüssel
   `D_ctl` geführt und ersetzt nie den Hauptlauf von Stufe D. Für ihn gilt nicht einfach der neueste:
   Liegen mehrere fertige Kontrollläufe vor (etwa unter `LSNE/runs_ctl` und `LSNE_B/runs_ctl`), wird in der
   Reihenfolge „voller Lauf vor Rauch- oder eingeschränktem Lauf, ganzes Design vor eingeschränktem,
   neuester zuerst“ der erste verwendet, dessen vier Audits bestehen. Ein neuerer ungültiger Lauf, etwa
   einer von einer anderen Plattform, verdrängt also keinen gültigen. Besteht keiner, wird der zuerst
   bevorzugte ausgewertet, und es gibt keinen gültigen Kontrolllauf. Die Begründung steht in
   `prereg_runs.csv`, unter den Hinweisen des Berichts und in der Notiz von `audit.D1ctl.run`.
4. **Ergebnis.** `MyDrive/LSNE/prereg_eval/pythia-160m@step98000/prereg_results.{csv,md}`,
   `prereg_runs.csv` und `prereg_meta.json`. Liegt dort schon eine Auswertung, werden diese Dateien von ihr
   vorher nach `archive/<Zeitstempel>/` verschoben, nie überschrieben. Der Zeitstempel kommt aus ihrem
   `prereg_meta.json` (UTC); der Bericht vom 28.10.2026, 07:07 UTC, liegt so seit der Auswertung des
   Kontrolllaufs (20:54 UTC, jetzt maßgeblich für `docs/RESULTS_PREREG.md`) in
   `archive/2026-10-28T07-07-48Z/`. Ein schon vorhandener Ordner wird nie
   wiederverwendet (dann `_2`, `_3`, …). `prereg_meta.json` nennt den Ordner unter
   `previous_evaluation_archive`. Die Übersichtsabbildung (`figures/13_prereg_evaluation/`) und
   `summary_13_prereg_evaluation_<modell>_<revision>.json` in `MyDrive/LSNE` tragen feste Namen und werden
   bei jedem Lauf überschrieben.
5. **Prüfen.** Die Zelle nach der Auswertung druckt
   - je nicht bestandenem Audit des Kontrolllaufs eine Zeile `ACHTUNG: Kontrolllauf ungültig – <Audit>:
     <Urteil>; <Notiz>` (dann ist P höchstens „offen“) oder `Kontrolllauf gültig: alle vier Audits
     bestanden`; ließen sich die Audits des verwendeten Kontrolllaufs gar nicht berechnen (keine Zeilen
     `audit.D1ctl…`, der Fehler steht in der Zeile `P.ctl`), stattdessen `ACHTUNG: Kontrolllauf nicht
     auswertbar (keine Audit-Zeilen; siehe Zeile P.ctl): der Primärendpunkt ist höchstens „offen“.`, und die
     Zusammenfassung führt `control_valid: false`,
   - bei mehreren Kontrollläufen `Mehrere Kontrollläufe gefunden (n); verwendet: …` mit der Begründung,
   - `Code der Auswertung: Git <Kurz-SHA>, lsne/prereg_eval.py SHA-256 <erste 12 Zeichen>` und, wenn ein
     Kontrolllauf verwendet wird, ob er mit demselben Code der Auswertung gerechnet wurde
     (`eval_code_sha256` in seinem Manifest; `derselbe Code` oder `ANDERER Code`),
   - `Frühere Auswertung archiviert nach: …` oder dass nichts archiviert wurde.

**Herkunft.** Der Bericht nennt in der Kopfzeile „Code der Auswertung“ den Git-Stand und den SHA-256 von
`lsne/prereg_eval.py`; `prereg_meta.json` führt beide als `git_sha` und `prereg_eval_sha256` (der SHA-256
erfasst auch nicht committete Änderungen). Der Code der Auswertung ist eingefroren, sobald der
Kontrolllauf gelaufen ist (`docs/RESULTS_PREREG.md` §9): Die Notiz von `audit.D1ctl.run` vergleicht
`prereg_eval_sha256` mit `eval_code_sha256` im Manifest des Kontrolllaufs. Weichen sie ab, ändert das kein
Urteil, ist aber eine Abweichung von Nachtrag 6.11, die berichtet werden muss.

CLI-Äquivalent (jede Maschine mit Zugriff auf die Laufverzeichnisse; dieselben Standardwurzeln wie das
Notebook, eine fehlende Wurzel ergibt nur einen Hinweis):

```
python -m lsne.prereg_eval \
    --runs /content/drive/MyDrive/LSNE/runs --runs /content/drive/MyDrive/LSNE/runs_trace \
    --runs /content/drive/MyDrive/LSNE/runs_ctl --runs /content/drive/MyDrive/LSNE_B/runs \
    --runs /content/drive/MyDrive/LSNE_B/runs_trace --runs /content/drive/MyDrive/LSNE_B/runs_ctl \
    --out /content/drive/MyDrive/LSNE/prereg_eval/pythia-160m@step98000 \
    --model pythia-160m --revision step98000 --n-boot 10000 --seed 0
```

Lesart: `not_tested` nennt den Grund (Stufe fehlt, Tabelle fehlt, keine Zeilen der
Bestätigungsmenge); `not_interpretable` heißt, die Präregistrierung erlaubt keine Deutung (G0
verfehlt, Lauf durch das Audit ungültig, Platzhalter-Korpus, synthetisches SAE,
Abweichungszeilen C2/C3/seg des Primärendpunkts). Ohne eine auf DEV01–DEV04 eingefrorene
Schadensdosis (`FROZEN_DOSE` bzw. `--frozen-dose`) wird der Primärendpunkt für jede
präregistrierte Dosis einzeln berichtet; die Notiz nennt die Bonferroni-Korrektur über die Dosen.
Seit Nachtrag 6.11 hängt der Primärendpunkt außerdem an der Kontrolle der Sprache an der
Injektionsstelle (Zeilen `P.ctl…`, Rolle `primary_control`, nicht in der BH-Gruppe), je Dosis: Beim
Ausgang „Wechsel“ bleibt P unverändert; bei „lokal“ wird ein gestütztes oder offenes P nicht
interpretierbar; beim Ausgang „offen“ und ohne gültigen Kontrolllauf von Notebook 14 ist P höchstens
„offen“. Gültig ist ein Kontrolllauf nur, wenn seine vier Audits bestehen (`audit.D1ctl`, `.rep`,
`.site`, `.run`); sonst sind alle Zeilen `P.ctl…` nicht interpretierbar. Die Notiz von P nennt den
Ausgang und das Ergebnis der Regel ohne die Kontrolle.

## 12. Kontrolle der Sprache an der Injektionsstelle (Notebook 14, A100)

`notebooks/14_token_control.ipynb` rechnet den Kontrolllauf aus Nachtrag 6.11 der Präregistrierung
(`docs/PREREGISTRATION.md` §6, Commit `2ce9c71`; Tabellen und Regeln in `docs/STAGES.md`). Anlass: In
C1 injiziert der Primärendpunkt bei K = 0 und K > 0 in denselben Punkt „.“ am Ende von Einheit 9; bei
K > 0 schließt er einen deutschen Satzteil ab, und der letzte Wechsel zurück ins Englische folgt direkt
danach. Der Kontrolllauf injiziert deshalb auch dort, wo „.“ einen englischen Satzteil abschließt: am
Ende von Einheit 9 in C2/C3 (Kontrolle 1) und am Ende von Einheit 10 in C1 (Kontrolle 2), je gegen K = 0
an derselben Stelle; nur Einheit 9 deutsch (`u9D`, Kontrolle 3) wird nur berichtet. An der
präregistrierten Stelle (Ende von Einheit 9, K = 0 und C1) muss er den Hauptlauf reproduzieren.

**Voraussetzungen.**

- **Hauptlauf.** Der Hauptlauf von Stufe D (Notebook 03) mit `status: ok`, der Tabelle `damage` und der
  Tiefen-Richtungsbank `directions_depth.npz` (+ `directions_depth_meta.json`) liegt auf Konto A unter
  `MyDrive/LSNE/runs/D/pythia-160m@step98000/`. Das Notebook nimmt ihn von dort; nur wenn er dort fehlt,
  nimmt es `MyDrive/LSNE_B/runs/D/…` (Verknüpfung zum anderen Konto, Abschnitt 11), oder den Pfad in
  `MAIN_D_RUN`. Seine Bank hat den SHA-256 `d9f05cd61e73…a120be` (vollständig in
  `docs/RESULTS_PREREG.md` §9); Zelle 1b druckt den Anfang, das Manifest des Kontrolllaufs den ganzen Wert
  (`bank.sha256`).
- **Plattform des Hauptlaufs.** Eine A100 mit Torch `2.11.0+cu130` und transformers `5.20.2` (laut Manifest
  des Hauptlaufs; die GPU verzeichnet das Manifest nicht, laut einer früheren OOM-Meldung war es eine A100
  40 GB). Bedingung (b) des Nachtrags (|Δλ| ≤ 1e-3 an der präregistrierten Stelle bei ε 0,01) verlangt
  praktisch bitgleiches λ: Schon ein anderer Attention-Kernel allein verschob λ auf pythia-160m um bis zu
  1,8e-2 (25 von 32 Zeilen über 1e-3). Auf derselben Plattform ist die Reproduktion bitgleich (auf der CPU
  geprüft gegen einen Hauptlauf, der mit dem Code des echten Hauptlaufs, `76a4a5a`, auf derselben CPU
  gerechnet wurde: 2 448 Zeilen, max |Δλ| = 0). Dass die A100 bei gleicher Form bitgleich rechnet (cuBLAS,
  Eager-Attention), wird angenommen, war aber nie geprüft; das prüft Zelle 1b an den Daten des Hauptlaufs.
- Der Kontrolllauf muss jünger sein als der Commit des Nachtrags (das prüft Notebook 13 mit
  `audit.D1ctl.run`).

**Ablauf.**

1. Auf **Konto A** eine **A100-Laufzeit** wählen (Laufzeit → Laufzeittyp ändern → A100) und
   `notebooks/14_token_control.ipynb` öffnen. Auch eine wiederverwendete Laufzeit ist in Ordnung: Die
   Setup-Zelle bringt den Checkout auf den Stand des Zweigs, und Vorabprüfung und Lauf starten als eigene
   Prozesse mit dem aktuellen Code. Wurden Torch oder transformers per `pip` gewechselt, vorher die Sitzung
   neu starten (Laufzeit → Sitzung neu starten).
2. **Alle Zellen ausführen.** Die Voreinstellungen passen: `REVISION = "step98000"`, `MAIN_D_RUN = None`,
   `REUSE = True`, `ALLOW_PLATFORM_MISMATCH = False`, `SMOKE = False`. Zelle 1a tut ohne Rauchtest nichts.
3. **Prüfungen vor jeder Rechnung (Zelle 1b).** Sie bricht mit einer deutschen Meldung ab („Alle
   ausführen“ stoppt), bevor irgendwo ein Modell geladen wird:
   - **Hauptlauf:** `manifest.json` lesbar, Stufe D, `status: ok`, kein Kontrolllauf, Modell, Revision und
     dtype wie in der Konfiguration, `lattice_directions`, `doses` und `limits.damage_batch_size` im
     Manifest, Bank und Tabelle `damage` vorhanden. Die Zelle druckt Zeitstempel, Git-Stand, Torch- und
     transformers-Version des Hauptlaufs und den Anfang des SHA-256 der Bank.
   - **Kontrollordner:** Liegt in `runs_ctl/D/pythia-160m@step98000/` ein anderer Lauf (ohne
     `control = "token_language"` oder mit einer Richtungsbank), bricht sie ab. Einen fertigen Kontrolllauf
     genau dieses Hauptlaufs (Pfad, Zeitstempel und SHA-256 der Bank wie im Hauptlauf, Nachtrag 6.11) prüft
     das Notebook mit `REUSE = True` nur; dann rechnet es nichts, und die beiden folgenden Prüfungen dienen
     nur der Information.
   - **Plattform:** GPU, Torch und transformers dieser Laufzeit neben denen des Hauptlaufs (`gleich` /
     `ABWEICHEND`). Muss gerechnet werden und weicht etwas ab (keine A100, eine andere Torch- oder
     transformers-Version), bricht die Zelle ab und nennt die Abhilfe mit den pip-Befehlen aus dem Manifest
     des Hauptlaufs, hier in einer neuen Codezelle
     ```
     !pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
     !pip install transformers==5.20.2
     ```
     danach Laufzeit → Sitzung neu starten und alle Zellen erneut ausführen. A100 mit 40 und 80 GB
     unterscheidet diese Prüfung nicht; das deckt die Vorabprüfung ab.
   - **Reproduzierbarkeit des Hauptlaufs** (aus fünf Spalten seiner Tabelle `damage`, ohne GPU): Zeile 0
     jedes D1-Forwards ist die ungestörte Referenz, ihr `logp_target_ref` steht in jeder Zeile dieses
     Forwards. Je Variante und Injektion liefen die 204 Paare aus Richtung und Dosis in der Reihenfolge des
     Manifests in Forwards zu 63 gestörten Zeilen plus Referenz, also drei volle Forwards zu 64 Zeilen und
     ein kurzer zu 16. Rechnet die GPU bei gleicher Form bitgleich, ist `logp_target_ref` in allen Forwards
     gleicher Größe einer Variante gleich, über beide Injektionen. Die Zelle druckt die Spannweiten; jede
     Spannweite > 0 (bei vollen oder bei kurzen Forwards gleicher Größe) heißt „NICHT reproduzierbar bei
     gleicher Form“ und stoppt (Fehlersuche unten). Den Unterschied zwischen kurzem und vollem Forward
     (andere Form) berichtet sie nur. Lassen sich die Zeilen den Forwards nicht zuordnen („nicht prüfbar“),
     warnt sie nur; die Vorabprüfung vergleicht λ dann direkt.
4. **Vorabprüfung (Zelle 2a, etwa 1 min, nur wenn gerechnet wird).** Derselbe Befehl mit `--families F01A
   --limit-variants 2` (K = 0 und K = 8 in C1) in einen frischen lokalen Ordner (`tempfile`, nie auf
   Drive, danach gelöscht; Log unter `runs_ctl/logs/`). λ an `u9` / C1 muss den Zeilen `depth` des
   Hauptlaufs genau gleichen (max |Δλ| = 0 bei gleicher Position; NaN zählt als ungleich); die Zelle zeigt
   max |Δλ| je Dosis. Sonst bricht sie ab, ebenso wenn der Aufruf scheitert oder verweigert wird.
5. **Kontrolllauf (Zelle 2b).** `python -m lsne.cli run --stage D --model EleutherAI/pythia-160m --revision
   step98000 --dtype float32 --device auto --out MyDrive/LSNE/runs_ctl --d-control token_language
   --d-main-run <Hauptlauf>` in einem eigenen Prozess; die Ausgabe erscheint im Notebook und im Log unter
   `runs_ctl/logs/`. Scheitert der Lauf oder wird er verweigert, bricht die Zelle ab. Sonst druckt sie
   Varianten, Rechenzeit, fortgesetzte Blöcke, die Marken, die Plattform (`replication_platform`),
   `platform_mismatch` und `eval_code_sha256` gegen den Code dieses Checkouts.
6. **Audits (Abschnitt 3).** Deskriptiv die Zeilen je Stelle, K und C und max |Δλ| je Dosis an `u9` / C1
   gegen den Hauptlauf; dann die vier Validitätsaudits `audit.D1ctl`, `audit.D1ctl.rep`,
   `audit.D1ctl.site` und `audit.D1ctl.run` genau so, wie Notebook 13 sie berechnet (derselbe Code,
   `lsne.prereg_eval.control_audits_for`), je mit Urteil und Notiz. Zum Schluss `Kontrolllauf gültig: alle
   vier Audits bestanden …` oder `ACHTUNG, Kontrolllauf ungültig: <Audits> nicht bestanden. …`; ein
   ungültiger Lauf zählt in Notebook 13 nicht, und P bleibt höchstens „offen“.
7. **Zusammenfassung und Drive leeren.** `MyDrive/LSNE/summary_14_token_control_pythia-160m_step98000_float32.json`
   (Plattformvergleich, Reproduzierbarkeit, Vorabprüfung, Lauf, Audits, `control_valid`, SHA-256 der Bank,
   Code der Auswertung). Sie trägt einen festen Namen, und ein späterer Durchlauf, der den fertigen Lauf nur
   prüft, überschreibt sie (die Vorabprüfung steht dann auf „nicht nötig (nur geprüft)“); das Ergebnis der
   Vorabprüfung schreibt Zelle 2a deshalb auch neben ihr Log:
   `runs_ctl/logs/14_token_control_pythia-160m_step98000_preflight.json`, auch wenn sie nicht bitgleich ist und
   die Zelle deshalb anhält (scheitert der Aufruf selbst, steht die Ursache nur im Log). Die letzte
   Zelle schreibt alle gepufferten Dateien nach Drive und trennt Drive (`drive.flush_and_unmount()`). Danach
   in dieser Sitzung nichts mehr auf Drive schreiben.
8. **Danach Notebook 13 in einer neuen Sitzung** (CPU-Laufzeit genügt, Abschnitt 11), sobald der
   Kontrolllauf in Google Drive sichtbar ist (`MyDrive/LSNE/runs_ctl/D/pythia-160m@step98000/manifest.json`).
   Erst dort fallen die Urteile: je Dosis `P.ctl` mit „Wechsel“, „lokal“ oder „offen“, die Audits
   `audit.D1ctl…` und die Folge für P.

`ALLOW_PLATFORM_MISMATCH = True` macht aus den Abbrüchen wegen Plattform, Reproduzierbarkeit und
Vorabprüfung Warnungen und gibt Vorabprüfung und Lauf `--d-allow-platform-mismatch` mit; die Abweichung
steht dann im Manifest unter `platform_mismatch`. Ein solcher Lauf ist fast sicher ungültig, und bei einem
nicht reproduzierbaren Hauptlauf ist der Schalter keine Abhilfe (Fehlersuche unten).

**Sicherheit.** Das Notebook liest den Hauptlauf nur; in `MyDrive/LSNE/runs/` schreibt es nichts und legt
den Ordner nicht einmal an. Die Konfigurationszelle bricht ab, wenn die Kontrollwurzel in einem
`runs`-Ordner oder in der Wurzel des Hauptlaufs läge oder den Hauptlauf enthielte. Bevor gerechnet wird,
legt das Notebook nur `runs_ctl/logs/` an (für das Log der Vorabprüfung; Notebook 13 vermerkt eine Wurzel
`runs_ctl` ohne Stufenordner nur als Hinweis); eine Prüfung, die abbricht, hinterlässt keinen Ordner. Der
Kontrolllauf selbst (`lsne.experiments.stage_d_control`) prüft in zwei Schritten und schreibt vorher nichts
in sein Laufverzeichnis:

- **Vor dem Laden des Modells** (`lsne.cli`; Abbruch mit einer Meldung `[lsne] …`, bei den Prüfungen des
  Kontrolllaufs `control run refused: …`): `--d-control` nur zusammen mit `--d-main-run` und nur mit
  `--stage D` (`--d-allow-platform-mismatch` nur mit beiden); das Laufverzeichnis `<--out>/D/<modell>@<rev>`
  ist das des Hauptlaufs (auch über einen Link), liegt in ihm oder enthält einen Lauf von Stufe D ohne
  `control = "token_language"` (gleich welcher Status) oder eine `directions_depth.npz` (ein abgebrochener
  Hauptlauf); `--out` ist ein `runs`-Ordner (auch über einen Link oder `runs/x/..`), liegt in der Wurzel des
  Hauptlaufs oder enthält ihn; der Hauptlauf ist kein fertiger Lauf von Stufe D, selbst ein Kontrolllauf,
  hat ein anderes Modell, eine andere Revision oder einen anderen dtype, oder ihm fehlen
  `lattice_directions`, `doses`, `limits.damage_batch_size` oder die Bank; `--families` wählt keine
  Testformulierung (F01–F12 × {A, B}); Torch oder transformers weichen von denen des Hauptlaufs ab (außer
  mit `--d-allow-platform-mismatch`).
- **Nach dem Laden des Modells**, noch vor jedem Schreiben: Die Bank liegt an einer anderen Stelle als die
  Tiefenstelle dieses Modells, oder alle Varianten der Auswahl fallen weg (K- und C-Stufen, Ausschlüsse nach
  §4). Das Laufverzeichnis entsteht erst, wenn der Lauf schreibt; ein leeres entfernt die Kommandozeile nach
  einer solchen Verweigerung wieder, und der Laufbericht `run_report_*.json` in `--out` vermerkt sie.
- Ein Lauf, der nach dem Start scheitert (nicht verweigert wird), hinterlässt ein Fehler-Manifest mit den
  Kontrollmarken; der nächste Start setzt ihn fort, statt ihn als Hauptlauf abzulehnen. Der Kontrolllauf hat
  sein eigenes `_resume/` mit eigener Signatur; das `_resume/` des Hauptlaufs wird nie berührt.

**Laufzeit.** 264 Varianten (24 Formulierungen × 11) an 360 Paaren aus Variante und Stelle, je 204 Paare
aus Richtung und Dosis: ≈ 1 440 Forwards zu 64 Zeilen, etwa ein Sechstel von D1 im Hauptlauf; geschätzt
5–15 min plus Modell-Download (nicht gemessen; die echte Zeit steht in `manifest.json`, `runtime_s`).
Davor die Vorabprüfung, etwa 1 min (Modell laden und 16 Forwards); die Prüfungen in Zelle 1b lesen fünf
Spalten der Tabelle `damage` des Hauptlaufs (Parquet, etwa 25 MB) von Drive.

**Fortsetzen.** Bricht die Sitzung ab, alle Zellen erneut ausführen. Ein abgebrochener Lauf setzt bei den
gesicherten Blöcken unter `runs_ctl/D/pythia-160m@step98000/_resume/` fort, aber nur auf derselben
Plattform und mit demselben Code-Stand: Die Signatur von `_resume/` enthält außer den Eingaben (Varianten,
Bank, Richtungen, Dosen, Batch-Aufteilung, Stellen) Torch, CUDA, Gerät, GPU, transformers,
Attention-Implementierung und den Git-SHA des Codes. Passt eines davon nicht, verwirft der Lauf die alten
Blöcke und beginnt neu, statt Blöcke zweier Plattformen oder Code-Stände zu mischen. Nach `status: ok` wird
`_resume/` gelöscht. Prüfungen und Vorabprüfung laufen bei jedem neuen Start erneut. `REUSE = True` prüft
einen fertigen Kontrolllauf genau dieses Hauptlaufs nur, ohne ihn neu zu rechnen, auch wenn seine Audits
scheitern; einen eingeschränkten Lauf (`design_filters` im Manifest, außer im Rauchtest) und einen
Kontrolllauf eines früheren Hauptlaufs am selben Pfad rechnet es neu. `REUSE = False` rechnet einen
fertigen Kontrolllauf neu und überschreibt ihn; einen gültigen deshalb nicht ohne Grund neu rechnen.

**Rauchtest.** `SMOKE = True` ist eigenständig und braucht kein Notebook 03: Zelle 1a baut mit der
Kommandozeile einen eigenen Rauch-Hauptlauf von Stufe D (`pythia-14m`, F01A, `--smoke`) unter
`<OUT_ROOT>/smoke_14/runs` und vermerkt danach das Gerät in `<OUT_ROOT>/smoke_14/smoke_main_device.txt`
(Name der GPU, sonst `cpu/<CPU-Fähigkeit>`). Einen fertigen mit Zeilen von F01A und derselben Torch- und
transformers-Version nimmt sie nur mit `REUSE = True` wieder und nur, wenn er auf diesem Gerät gerechnet
wurde; sonst entfernt sie ihn und baut ihn neu, und die Ausgabe nennt den Grund (etwa „anderes Gerät“
oder „REUSE = False“). Ein auf einer CPU-Laufzeit gerechneter Rauch-Hauptlauf wird auf der A100 also nicht
wiederverwendet (die Vorabprüfung würde an ihm scheitern). Danach laufen die Prüfungen (Plattform
und Reproduzierbarkeit werden nur berichtet), die Vorabprüfung (mit `--limit-variants 2`), der Kontrolllauf
unter `<OUT_ROOT>/smoke_14/runs_ctl` und die Audits; `audit.D1ctl.site` und `audit.D1ctl.run` scheitern dort
absichtlich (wenige Varianten, Rauchlauf). Auf der CPU dauert das wenige Minuten. `MyDrive/LSNE/runs` und
`MyDrive/LSNE/runs_ctl` bleiben unberührt, und die Standardwurzeln von Notebook 13 sehen den Rauchtest
nicht; zum Ausprobieren dort `RUN_ROOTS_TEXT` auf `<OUT_ROOT>/smoke_14/runs,<OUT_ROOT>/smoke_14/runs_ctl`
und `MODEL_TAG = "pythia-14m"` setzen (die Zusammenfassung von Notebook 14 druckt beide Zeilen).

**CLI-Äquivalent** (nur auf der Plattform des Hauptlaufs: A100 mit Torch `2.11.0+cu130` und transformers
`5.20.2`):

```bash
python -m lsne.cli run --stage D --d-control token_language \
    --d-main-run /data/lsne/runs/D/pythia-160m@step98000 --out /data/lsne/runs_ctl \
    --model EleutherAI/pythia-160m --revision step98000 --dtype float32 --device auto
```

Die Kommandozeile prüft von der Plattform nur Torch und transformers; GPU, Reproduzierbarkeit des
Hauptlaufs und Vorabprüfung prüft nur das Notebook. `--d-allow-platform-mismatch` rechnet trotz
abweichender Versionen und verzeichnet sie unter `platform_mismatch`; die Reproduktion (`audit.D1ctl.rep`)
scheitert dann sehr wahrscheinlich.

**Fehlersuche (Notebook 14).**

| Symptom | Ursache / Abhilfe |
|---|---|
| `Hauptlauf von Stufe D nicht verwendbar (…): …/manifest.json fehlt` | Auf diesem Konto liegt kein Hauptlauf unter `runs/D/<modell>@<revision>` und auch keiner unter `LSNE_B`: Notebook auf Konto A ausführen, die Verknüpfung `LSNE_B` anlegen (Abschnitt 11) oder `MAIN_D_RUN` setzen. |
| `… Status 'error' statt 'ok'` / `directions_depth.npz fehlt` / `damage.parquet / damage.csv fehlt` | Der Hauptlauf ist nicht fertig: Notebook 03 bis zum Ende laufen lassen (`REUSE = True` lädt einen fertigen Lauf). |
| `In …/runs_ctl/D/… liegt ein anderer Lauf als dieser Kontrolllauf` | Dort liegt ein Lauf ohne `control = token_language` (etwa ein kopierter Hauptlauf): Er wird nie überschrieben. Den Ordner prüfen und umbenennen. |
| `Diese Laufzeit weicht vom Hauptlauf ab: GPU … statt A100 …` / `Torch … statt 2.11.0+cu130` / `transformers … statt 5.20.2` (Zelle 1b) | Keine A100 (auch: keine GPU) oder andere Versionen als im Hauptlauf; der Kontrolllauf wäre fast sicher ungültig. Laufzeit → Laufzeittyp ändern → A100; die Versionen mit den pip-Befehlen aus der Meldung installieren (Ablauf, Schritt 3); Laufzeit → Sitzung neu starten; alle Zellen erneut ausführen. `ALLOW_PLATFORM_MISMATCH = True` nur bewusst. |
| `Der Hauptlauf war auf seiner GPU bei gleicher Form nicht bitgleich reproduzierbar …` (Zelle 1b) | `logp_target_ref` der ungestörten Referenzzeile weicht zwischen Forwards gleicher Größe derselben Variante ab: Schon der Hauptlauf rechnete nicht bitgleich, also kann kein neuer Lauf Bedingung (b) erfüllen. Den Kontrolllauf **nicht** rechnen, auch nicht mit `ALLOW_PLATFORM_MISMATCH = True`: Vor jedem Kontrolllauf ist ein datierter Nachtrag zu 6.11 (b) in `docs/PREREGISTRATION.md` §6 nötig. |
| `Die Vorabprüfung reproduziert den Hauptlauf nicht bitgleich …` (Zelle 2a) | Diese Laufzeit rechnet den Hauptlauf nicht Bit für Bit nach, obwohl GPU-Typ und Versionen passen (etwa eine A100 mit anderem Speicher, Treiber- oder Bibliotheksstand); der ganze Lauf würde Bedingung (b) sehr wahrscheinlich verfehlen. Eine andere A100-Laufzeit mit den Versionen des Hauptlaufs nehmen (pip-Befehle aus der Meldung), neu starten, alle Zellen erneut ausführen. Im Rauchtest (`SMOKE = True`, Meldung mit „Rauchtest: …“) stammt der Rauch-Hauptlauf am wahrscheinlichsten von einem anderen Gerät, etwa einer anderen Maschine mit derselben Gerätebezeichnung (`smoke_main_device.txt`); dort helfen weder A100 noch pip: `REUSE = False` setzen oder `<OUT_ROOT>/smoke_14/runs` löschen und alle Zellen erneut ausführen, Zelle 1a baut ihn dann auf dieser Laufzeit neu. |
| `Kontrolllauf fehlgeschlagen oder verweigert (exit 1, …)` / `Die Vorabprüfung ist fehlgeschlagen oder wurde verweigert (…)` | Ursache im Log unter `runs_ctl/logs/` (`control run refused: …`, `REFUSED`, `FAILED`, z. B. `CUDA out of memory`); nach der Korrektur die Zelle erneut ausführen, ein abgebrochener Lauf setzt fort (gleiche Plattform und gleicher Code-Stand). |
| `ACHTUNG, Kontrolllauf ungültig: audit.D1ctl.rep …` (Abschnitt 3) / Notebook 13: `ACHTUNG: Kontrolllauf ungültig – audit.D1ctl.rep: nicht bestanden …` (`\|difference\| > 1e-3 or not finite`) | λ an der präregistrierten Stelle weicht um mehr als 1e-3 vom Hauptlauf ab, fast immer wegen einer anderen Plattform (`replication_platform`, `platform_mismatch` im Manifest). Der Kontrolllauf ist ungültig, und P bleibt höchstens „offen“. Abhilfe: auf einer A100 mit Torch `2.11.0+cu130` und transformers `5.20.2` wiederholen (die pip-Befehle, die das Notebook druckt), Sitzung neu starten, `REUSE = False` setzen (sonst wird der fertige Lauf nur erneut geprüft) und alle Zellen ausführen. Der neue Lauf ersetzt den alten im selben Ordner; liegen mehrere fertige Kontrollläufe vor (etwa einer unter `LSNE_B/runs_ctl`), nimmt Notebook 13 den ersten gültigen in der Reihenfolge aus Abschnitt 11, bei vollen Läufen des ganzen Designs also den neuesten gültigen. |
| Notebook 13: `audit.D1ctl.rep` „… without a control row“, `audit.D1ctl.site` „cells of the design missing“ oder „(tag, site) of the design without exactly the … (direction, eps) pairs“ | Der Kontrolllauf deckt nicht alle Zeilen von P, nicht alle Zellen oder nicht alle Paare aus Richtung und Dosis ab, meist weil er mit `--families`, `--limit-*` oder eingeschränkten K-/C-Stufen gerechnet wurde (`design_filters` im Manifest). Ohne diese Optionen neu rechnen; ein Lauf des ganzen Designs wird ohnehin vor einem eingeschränkten verwendet. |
| Notebook 13: `audit.D1ctl.run` nicht bestanden (`smoke / limited run`, `predates the commit of amendment 6.11`, `control … instead of 'token_language'`) | Ein Rauch- oder eingeschränkter Lauf, ein Lauf älter als der Nachtrag oder einer ohne die Manifest-Marken zählt nie: den vollen Lauf mit Notebook 14 (`SMOKE = False`) rechnen. |
| Notebook 13: `Kontrolllauf gerechnet mit … ANDERER Code` | `lsne/prereg_eval.py` wurde nach dem Kontrolllauf geändert (`eval_code_sha256` im Manifest ≠ `prereg_eval_sha256` des Berichts). Das ändert kein Urteil, ist aber eine Abweichung von Nachtrag 6.11 (der Code der Auswertung ist eingefroren, sobald die Kontrolle gelaufen ist) und muss im Bericht vermerkt werden. |
| `control run refused: … is a runs folder, lies inside the root of the main stage-D run …` | `--out` ist ein `runs`-Ordner (auch der des anderen Kontos) oder enthält den Hauptlauf: den Kontrolllauf unter `MyDrive/LSNE/runs_ctl` rechnen. |
| `Der Rauch-Hauptlauf ist fehlgeschlagen …` (nur `SMOKE = True`) | Ursache im Log unter `<OUT_ROOT>/smoke_14/runs_ctl/logs/`; Zelle 1a erneut ausführen. |
