# Spurenmodell als Quellmodell (Stufe L): End-to-End-Lauf auf echten 2016-Daten

Stand 26.10.2026. Dieses Dokument beschreibt den Quelladapter `lsne.chain` (Vertrag:
`docs/TRACE_SOURCE_SPEC.md`, Hintergrund: `docs/BLOCKGPT_FEASIBILITY.md`, DESIGN §4.9) und
einen vollständigen Durchlauf der Pipeline auf echten Ethereum-Spuren aus der DAO-Zeit
2016 – auf der CPU, in kleinem Maßstab.

> **Einordnung vorweg.** Der hier dokumentierte Lauf ist eine **Validierung der
> Rohrleitungen** (Download → Serialisierung → Vokabular → Training → Sonden →
> Alignment mit Pythia), **kein wissenschaftliches Ergebnis**. Ein kleines Modell
> (3,2 M Parameter ohne Embeddings), 200 Optimierungsschritte (0,82 M Tokens) auf einer
> Stichprobe von 31 der 672 Chunks der DAO-Zeit kann die Frage, ob ein Spurenmodell ein
> Merkmal „Teilen“ lernt und ob es sich nach Pythia übertragen lässt, nicht beantworten.
> Er zeigt, dass jeder Schritt auf echten Daten läuft, was er kostet und welche Kontrollen
> für den A100-Lauf zwingend sind.

**Ergebnis in Kürze** (Einzelheiten in §4–§8; alle Rohzahlen in `runs/trace_e2e/summary.json`):

- **Daten:** 31 Xatu-Chunks (jeder 25. der Blöcke 1 428 000–2 099 999 plus Drain und
  Fork vollständig), 186 Dateien, 59,5 MB in 47 s; 225 854 Transaktionen, 7,05 M ITR-Tokens.
- **Vokabular:** 42 019 Tokens; OOV auf den zurückgehaltenen 10 %: Adressen 11,9 %,
  Selektoren 0,13 %, Topics 0,03 %.
- **Training `small`** (CPU, 30 min, 200 Schritte, 0,82 M Tokens = 0,16 Epochen):
  Validierungsverlust 10,70 → **1,121 nats** (alle Val-Blöcke), klar unter Unigramm
  (4,422) und Bigramm (1,490).
- **Sonden:** fanout (106 bzw. 52 Paare) und dao (je 500 Paare) sind in allen Layern
  trennbar (Split-Half-AUC 0,87–1,00), aber das **untrainierte** Modell trennt genauso gut;
  fanout ist eine Längensonde, dao eine Adress-Identitätssonde. Kein Beleg für ein gelerntes
  Merkmal „Teilen“.
- **Alignment mit Pythia-14m** (300 Paare): r2_holdout bis 0,673 (L4 → L4) gegen
  r2_permuted −0,789; aber zwei Längenmerkmale allein erreichen 0,556, das untrainierte
  Modell 0,605.
- **Hochrechnung (Extrapolation):** `blockgpt` über ganz 2016 (≈ 13,6 M Transaktionen,
  ≈ 338 M Trainings-Tokens, ≈ 5,2 GB Download) braucht auf einer A100 ≈ 0,3–1,1 h reines
  Training pro Epoche, mit Datenaufbereitung eine Sitzung von 2–3 h; auf dieser CPU wären
  es ≈ 80–160 Tage.

## 1. Was gebaut wurde

| Modul | Aufgabe |
|---|---|
| `lsne/chain/xatu.py` | URLs, resumierbarer Download (atomar, Backoff), normalisiertes Lesen der Xatu-Parquet-Chunks (Hashes/Adressen klein, Little-Endian-uint256 → int, `trace_address` → Tupel) |
| `lsne/chain/itr.py` | Call-Baum je Transaktion aus `trace_address`, Logs/Storage-Diffs an den ersten passenden Knoten, ITR-Serialisierung im BlockGPT-Format (Breitensuche, `[START] [CALL] [D{Tiefe}] [CT:…] A:von A:an F:Selektor V:Wert G:Gas [INs] … [OUTs] [END]`, `[STATE] [W]`, `[LOG]`, Zahlen auf zwei signifikante Stellen `N{m}e{k}`) |
| `lsne/chain/text.py` | deterministische englische Beschreibung derselben Transaktion für Pythia (`render_text`) |
| `lsne/chain/labels.py` | Teilen-Merkmale (ökonomischer fan-out/fan-in über Transfers mit Wert > 0, DAO-/WithdrawDAO-Beteiligung, Ären), Labels, gematchte Sondensets, Oberflächen-Baseline |
| `lsne/chain/vocab.py` | Vokabular (geschlossene Klassen + häufigste Adressen/Selektoren/Topics, Rest → `[A_OOV]`/`[F_OOV]`/`[T_OOV]`), HF-Tokenizer (WordLevel) |
| `lsne/chain/train.py` | GPT-NeoX-Causal-LM (Pythia-Konventionen: rotary 25 %, paralleles Residual), Packing, resumierbares Training, Durchsatzmessung |
| `lsne/chain/source.py` | `TraceLMSource` (implementiert `lsne.transplant.SourceModel`), gepoolte Zustände, Teilen-Richtung (Difference-of-Means), Paare (ITR, Text) |
| `lsne/chain/cli.py` | `python -m lsne.chain.cli download / serialize / vocab / train / probe / pair / bench` |
| `scripts/trace_e2e.sh` | dieser End-to-End-Lauf, Schritt für Schritt, mit allen Kontrollen |

Architektur bewusst GPT-NeoX statt GPT-2, damit `lsne.model.load_model`, die Hooks, der
`Injector` und `lsne.transplant` unverändert auf dem Spurenmodell laufen. Größen:
`tiny` (2 × 64), `small` (4 Layer, d = 256, 4 Heads, MLP 1 024), `blockgpt` (8 Layer,
d = 512, 8 Heads, MLP 2 048, Kontext 512 – Layerzahl und Kontext wie im BlockGPT-Paper,
Breite nicht veröffentlicht).

## 2. Datenherkunft und Lizenz

Alle Spuren stammen aus **ethPandaOps Xatu** (https://ethpandaops.io/posts/xatu-execution-layer/),
öffentliche Parquet-Dateien, 1 000 Blöcke pro Datei:

```
https://data.ethpandaops.io/xatu/mainnet/databases/default/canonical_execution_<tabelle>/1000/<start>.parquet
```

**Lizenz: CC BY 4.0. Namensnennung: „Data: ethPandaOps Xatu (https://ethpandaops.io), CC BY 4.0“.**
Die abgeleiteten ITR-Shards, Vokabulare und Modelle sind Bearbeitungen dieser Daten und
tragen dieselbe Namensnennung (`serialize_params.json`, `manifest.json`: `data_source`).
Rohdaten, Shards und Modelle liegen unter `cache/` bzw. `runs/` und werden nie committet
(`.gitignore`).

Verwendete Tabellen (Standard der CLI): `traces`, `transaction`, `native_transfers`,
`erc20_transfers`, `logs`, `storage_diffs`. `storage_reads` wurde **nicht** geladen,
daher enthält dieser Lauf keine `[STATE] [R]`-Tokens.

Stichprobe dieses Laufs (Blöcke 1 428 000 – 2 100 000, DAO aktiv bis nach dem Fork):

| Teil | Chunks | Dateien | Bytes | Zeit |
|---|---|---|---|---|
| jeder 25. Chunk 1 428 000–2 099 999 | 27 | 162 | 48,8 MB | 39,2 s |
| Drain, voll: 1 718 000–1 719 999 | 2 | 12 | 5,9 MB | 3,8 s |
| Fork, voll: 1 920 000–1 921 999 | 2 | 12 | 4,9 MB | 4,2 s |
| **Summe** | **31** | **186** | **59,5 MB** | **47,2 s** (≈ 1,3 MB/s, 4 Threads) |

Aufteilung nach Tabelle: traces 18,7 MB, transaction 17,5 MB, native_transfers 16,0 MB,
storage_diffs 3,8 MB, logs 2,8 MB, erc20_transfers 0,7 MB. Alle 186 Dateien kamen
vollständig an (Parquet-Magie an beiden Enden geprüft), kein Fehler.

Für die Hochrechnung (§8) wurde zusätzlich jeder 100. Chunk des ganzen Jahres 2016
(Blöcke 778 000–2 912 999, 22 Chunks, 132 Dateien, davon 90 neu, 42,2 MB in 23,1 s)
geladen und serialisiert. 2016 umfasst laut Xatu-Blocktabelle die Blöcke 778 483
(01.01.2016 00:00:04 UTC) bis 2 912 406 (31.12.2016 23:59:31 UTC), also 2 135 Chunks.

## 3. Befehle in Reihenfolge

Alles steht ausführbar in `scripts/trace_e2e.sh` (jeder Schritt einzeln aufrufbar,
resumierbar, Logs unter `runs/trace_e2e/logs/`); die Kernbefehle sind die der CLI:

```bash
# 1. Download (resumierbar; vorhandene vollständige Dateien werden übersprungen)
python -m lsne.chain.cli download --from 1428000 --to 2100000 --every 25 --cache cache/xatu
python -m lsne.chain.cli download --from 1718000 --to 1720000 --cache cache/xatu      # Drain
python -m lsne.chain.cli download --from 1920000 --to 1922000 --cache cache/xatu      # Fork
# 2. Serialisierung: ein gzip-JSONL-Shard pro Chunk (Tokens, Merkmale, Ära, Label, Text)
python -m lsne.chain.cli serialize --cache cache/xatu --from 1428000 --to 2100000 \
    --out runs/trace_e2e/itr --workers 4
# 3. Vokabular NUR aus dem Trainingsteil (Hash-Split 90/10, siehe Skript: itr_train)
python -m lsne.chain.cli vocab --in runs/trace_e2e/itr_train --out runs/trace_e2e/vocab
# 4. Durchsatz (CPU) für small und blockgpt mit diesem Vokabular
python -m lsne.chain.cli bench --sizes small,blockgpt --vocab runs/trace_e2e/vocab \
    --block-size 512 --batch-size 8 --optimizer-step --out runs/trace_e2e/bench.csv
# 5. Unigramm-/Bigramm-Baseline auf exakt den gepackten Blöcken des Trainings (Skript)
# 6. Training 'small', begrenzt (200 Schritte x 8 x 512 Tokens)
python -m lsne.chain.cli train --in runs/trace_e2e/itr --vocab runs/trace_e2e/vocab --size small \
    --out runs/trace_e2e/trace_lm_small --steps 200 --batch-size 8 --lr 1e-3 --warmup 30 \
    --val-frac 0.1 --eval-every 20 --eval-blocks 64 --save-every 20 --device cpu
# 7. Validierungsverlust auf allen Validierungsblöcken, je Tokenklasse (Skript)
# 8. untrainierte Kontrolle: gleiche Architektur, gleiches Vokabular, gleicher Seed (Skript)
# 9. Sonden (Spec: Gas-Dezile; Sensitivität: 1 000 Gas-Strata) + Kontrollen (Skript)
python -m lsne.chain.cli probe --model runs/trace_e2e/trace_lm_small --in runs/trace_e2e/itr \
    --kind fanout --layer all --n 500 --gas-bins 10 --out runs/trace_e2e/probe_fanout_g10
python -m lsne.chain.cli probe --model runs/trace_e2e/trace_lm_small --in runs/trace_e2e/itr \
    --kind dao --layer all --n 500 --gas-bins 10 --out runs/trace_e2e/probe_dao_g10
# 10. 300 Paare (ITR, render_text)
python -m lsne.chain.cli pair --in runs/trace_e2e/itr --n 300 --out runs/trace_e2e/pairs.jsonl
# 11. Layer-Matching Spurenmodell -> EleutherAI/pythia-14m (Mean-Pooling, Ridge) (Skript)
# 12. 2016-Profil für die Hochrechnung (jeder 100. Chunk), 13. Zusammenfassung summary.json
```

```bash
scripts/trace_e2e.sh                       # alles
scripts/trace_e2e.sh probe align report    # einzelne Schritte
STEPS=2000 SIZE=blockgpt scripts/trace_e2e.sh train   # Parameter per Umgebungsvariable
```

Ergebnisse unter `runs/trace_e2e/` (nicht im Repository): `itr/` (Shards),
`vocab/` (Vokabular, Tokenizer, `vocab_report.json`), `bench.csv`, `baselines.json`,
`trace_lm_small/` (Modell, `train_log.csv`, `manifest.json`), `trace_lm_untrained/`,
`eval.json`, `probe_{fanout,dao}_g{10,1000}/` (`match_table.csv`, `probe_summary.csv`,
`separability.csv`, `sharing_directions.pt`), `probe_report.json`, `pairs.jsonl`,
`layer_matching.csv`, `profile2016.{csv,json}`, `summary.json`, `logs/`.

Der Validierungssplit ist überall derselbe: eine Transaktion gehört zum Validierungsteil,
wenn die letzten 8 Hex-Stellen ihres Hashes / 2^32 < 0,1 sind (`lsne.chain.cli._is_val`).
Das Vokabular sieht den Validierungsteil nicht; die OOV-Raten unten sind damit echte
Hold-out-Raten.

## 4. Gemessene CPU-Zahlen

Maschine: 4 geteilte Kerne (torch 2.14 cpu, 4 Threads, fp32), 15 GB RAM. Während des
Laufs liefen auf derselben Maschine Testläufe anderer Arbeitsstränge (Load Average 7–12);
alle Zeiten sind deshalb **pessimistisch und verrauscht** (±50 %), sie geben die
Größenordnung, nicht die Leistung einer freien CPU.

| Schritt | Umfang | Wandzeit |
|---|---|---|
| Download | 186 Dateien, 59,5 MB | 47 s |
| Serialisierung (4 Prozesse) | 31 Chunks → 225 854 Transaktionen, 7 051 538 ITR-Tokens, 39 MB gzip | 12–28 s (zwei Läufe¹) |
| Split 90/10 + Vokabular | 203 304 Trainings-Transaktionen | 7 s + 8 s |
| Vokabular-Bericht (Hold-out-OOV, Budgets) | 22 550 Hold-out-Transaktionen | 154 s (1 Kern, parallel zum Training) |
| Unigramm/Bigramm-Baseline | 10 044 + 1 099 Blöcke | 10 s |
| Durchsatz `small`, 8 × 512, fwd+bwd+AdamW (`bench`) | — | 16,2 s/Schritt = 253 Tokens/s² |
| Durchsatz `blockgpt`, 8 × 512, fwd+bwd+AdamW (`bench`) | — | 167 s/Schritt = 24,5 Tokens/s² |
| Training `small` | 200 Schritte × 8 × 512 = 819 200 Tokens | 30,0 min (1 797 s, s. u.) |
| Validierung auf allen Blöcken (fwd) | 1 099 Blöcke = 561 589 vorhergesagte Tokens | 405 s |
| Sonden-CLI `probe` (Paare suchen, 5 Snapshots, Richtungen) | fanout: 106 / 52 Paare; dao: je 500 Paare | 253 s / 135 s; 420 s / 135 s |
| Sonden-Kontrollen (untrainiert, 200 Zufallsrichtungen, 100 Permutationen, Längenbereinigung) | alle vier Sondensets | 1 862 s |
| `pair` | 300 Paare | 11 s |
| Layer-Matching (2 Quellen × 3 Ridge-α + Längenkontrolle, Pythia-14m) | 300 Paare, 5 × 7 Layer | 43 s |
| 2016-Profil: Download / Serialisierung | 22 Chunks (90 neue Dateien, 42 MB) / 46 Chunks | 23 s / 104 s |

¹ Die Shards wurden nach einer Präzisierung der fan-out-Definition in `labels.py`
(ökonomisch, nur Transfers mit Wert > 0) mit `--overwrite` neu geschrieben; alle Zahlen
in diesem Dokument stammen aus diesen Shards.
² Der Benchmark lief gleichzeitig mit einem zweiten Benchmark-Prozess; das Training selbst
erreichte mit `small` effektiv 495 Tokens/s, also etwa das Doppelte. Die
`blockgpt`-CPU-Zahl ist entsprechend eher zu niedrig.

Training: effektiv 495 Tokens/s inklusive der 11 Zwischenvalidierungen (64 Blöcke), etwa
6–9 s pro Schritt je nach Fremdlast. Der Lauf wurde nach dem Checkpoint bei Schritt 20
(geplant waren 320 Schritte, zu langsam unter der Fremdlast) abgebrochen und mit
`--steps 200` fortgesetzt; weil Schritt 20 noch im linearen Warm-up liegt (30 Schritte) und
die Batch-Reihenfolge eine reine Funktion von Seed und Schritt ist, ist der fortgesetzte
Lauf identisch mit einem ununterbrochenen 200-Schritt-Lauf (nebenbei ein Test der
Resume-Logik). Wandzeit 401 s (Abschnitt 1, davon ~125 s nach dem Checkpoint verworfen)
+ 1 397 s (Abschnitt 2).

## 5. Korpus, Vokabular, Training

**Korpus.** 225 854 Transaktionen aus 31 Chunks (Blocklevel-Zeilen ausgeschlossen), Ären:
dao_active, drain_crisis, fork (die 10 Fork-Blöcke), post_fork. Länge in ITR-Tokens:
Median 14, 90 % ≤ 40, 99 % ≤ 230, Maximum 6 953; 0,26 % der Transaktionen sind länger als
512 und werden im Training auf 512 Tokens abgeschnitten (593 Transaktionen; auch die
DAO-Drain-Transaktion mit 1 409 Tokens). Die beiden Drain-Chunks liefern 29 % aller
Tokens (Split-Angriffe und Nachahmer). Transaktionen je Ära: dao_active 82 509,
drain_crisis 70 263, fork 110, post_fork 72 972. DAO-beteiligt (berührt DAO, extraBalance,
Creator oder WithdrawDAO): 11 207 (dao_active 3 502, drain_crisis 3 251, fork 8, post_fork
4 446), davon 1 732 mit WithdrawDAO. Label-Verteilung (ökonomischer fan-out/fan-in über
Transfers mit Wert > 0, k = 5): `single` 198 070 (87,7 %), `none` 14 851 (6,6 %),
`many_to_many` 12 874 (5,7 %), `other` 55, `distribution` 4, `pooling` 0. Ökonomischer
fan-out: 0 → 19 785, 1 → 193 181, 2 → 12 218, 3 → 548, 4 → 16, ≥ 5 → **106**
Transaktionen. Hoher fan-out ist 2016 also extrem selten, und die 106 Fälle haben fast
alle mehrere Sender (nur 4 sind reine „Verteilung“).

**Vokabular** (nur aus dem Trainingsteil, 203 304 Transaktionen, 6,36 M Tokens):

| Teil | Tokens | davon im Training benutzt |
|---|---|---|
| geschlossen: Wert-Tokens `V:N…` | 7 030 | 2 899 |
| geschlossen: Gas-Tokens `G:N…` | 7 030 | 268 |
| geschlossen: nackte Zahlen (Argumente) `N…` | 7 030 | 1 101 |
| geschlossen: Tiefe `[D0]`…`[D32]`, Slot-Buckets, Call-Typen, Fehler, Struktur | 105 | 87 |
| offen: Adressen (Top 20 000 von 65 108 verschiedenen; DAO-Adressen immer) | 20 000 | — |
| offen: Funktionsselektoren (alle) | 639 | — |
| offen: Event-Topics (alle) | 185 | — |
| **Summe** | **42 019** | |

80 % der geschlossenen Klasse (16 840 Tokens, fast alle Zahlentokens) kommen im Training
nie vor; sie kosten nur Embedding-Zeilen (bei `blockgpt` 2 × 8,6 M Parameter).

**OOV auf den zurückgehaltenen 10 %** (22 550 Transaktionen, 695 133 Tokens; Anteil der
Vorkommen der jeweiligen Klasse, die auf `[X_OOV]` fallen):

| | Adressen | Selektoren | Topics | alle offenen | `[UNK]` |
|---|---|---|---|---|---|
| Hold-out gesamt | **11,9 %** | **0,13 %** | **0,03 %** | 9,7 % | 0 |
| dao_active (8 222 Tx) | 13,9 % | 0,04 % | 0,06 % | | |
| drain_crisis (7 126 Tx) | 8,3 % | 0,13 % | 0 | | |
| post_fork (7 186 Tx) | 16,1 % | 0,25 % | 0,10 % | | |
| Trainingsteil (in-sample) | 7,9 % | 0 | 0 | | |

Adress-OOV gegen Budget (Hold-out): 1 000 → 23,1 %, 5 000 → 17,7 %, 20 000 → 11,9 %,
50 000 → 7,2 %, alle 65 108 Trainingsadressen → 4,9 %. Selektoren und Topics sind 2016
praktisch geschlossen (639 bzw. 185 verschiedene). Tokenmix im Hold-out: 47 % Struktur,
17 % Adressen, 7,5 % Werte, je 6,2 % Tiefe/Call-Typ/Selektor/Gas, Rest Slots, Topics,
Argumente.

**Training** (`small`: 24,67 M Parameter, davon 10,76 M Embedding, 10,76 M Unembedding,
3,16 M im Transformer; AdamW, Spitzen-LR 1e-3, 30 Schritte Warm-up, Kosinus bis 1e-4,
Batch 8 × 512, fp32). Korpus gepackt: 10 044 Trainings- und 1 099 Validierungsblöcke à
512 (5,14 M bzw. 0,56 M Tokens); 200 Schritte = 819 200 Tokens = **0,16 Epochen**.

| Schritt | Tokens | Train-Loss (Mittel seit letzter Zeile) | Val-Loss (64 Blöcke) |
|---|---|---|---|
| 0 | 0 | – | 10,699 |
| 20 | 81 920 | 8,185 | 6,395 |
| 40 | 163 840 | 3,986 | 2,628 |
| 60 | 245 760 | 2,034 | 1,819 |
| 80 | 327 680 | 1,684 | 1,617 |
| 100 | 409 600 | 1,557 | 1,433 |
| 120 | 491 520 | 1,394 | 1,213 |
| 140 | 573 440 | 1,347 | 1,136 |
| 160 | 655 360 | 1,189 | 1,085 |
| 180 | 737 280 | 1,185 | 1,021 |
| 200 | 819 200 | 1,143 | 0,983 |

**Baselines** (nats pro Token, dieselben gepackten Blöcke, dieselben Zielpositionen):

| | alle 1 099 Val-Blöcke | 64 Monitor-Blöcke |
|---|---|---|
| gleichverteilt, ln V | 10,646 | 10,646 |
| Unigramm (Trainingszählung, Add-one) | 4,422 | 4,587 |
| Bigramm (interpoliert, λ = 0,95 auf Trainings-Dev gewählt) | 1,490 | 1,513 |
| **Modell nach 200 Schritten** | **1,121** | **0,983** |

Der Validierungsverlust liegt **klar unter der Unigramm-Baseline** (−3,30 nats) und auch
unter der Bigramm-Baseline (−0,37 nats). Aufschlüsselung nach Klasse des vorhergesagten
Tokens (alle Val-Blöcke, nats):

| Zielklasse | Anteil | Modell | Bigramm | Unigramm |
|---|---|---|---|---|
| Struktur (`[START]`, `[END]`, `[CALL]`, …) | 47,2 % | 0,08 | 0,46 | 2,95 |
| Adresse im Vokabular | 12,9 % | 3,95 | 4,41 | 7,82 |
| Wert `V:` | 7,1 % | 4,11 | 3,99 | 7,53 |
| Gas `G:` | 5,5 % | 1,57 | 2,48 | 5,96 |
| Tiefe `[D…]` | 5,5 % | 0,15 | 0,87 | 3,73 |
| Selektor `F:` | 5,5 % | 0,64 | 0,99 | 4,70 |
| Call-Typ | 5,5 % | 0,12 | 0,22 | 3,15 |
| `[EOS]` (Transaktionsende) | 4,0 % | 0,02 | 0,06 | 3,24 |
| `[A_OOV]` | 2,5 % | 2,99 | 2,62 | 4,13 |
| Slot-Bucket | 1,7 % | 0,70 | 1,86 | 5,42 |
| Topic | 1,5 % | 0,77 | 1,41 | 6,61 |
| Zahl-Argument | 1,1 % | 3,89 | 4,41 | 8,37 |

Gelernt ist vor allem die Grammatik der Spuren (Struktur, Tiefe, Call-Typ, Gas, Selektor
nach Adresse). Bei Werten und bei `[A_OOV]` ist das Modell nach 0,16 Epochen noch
schlechter als das Bigramm; Adressen bleiben mit ~4 nats das Schwierigste. Der Train-Loss
liegt über dem Val-Loss, weil er über die frühen, schlechteren Schritte eines Intervalls
mittelt und die 64 Monitor-Blöcke leichter als der Durchschnitt sind (alle Blöcke: 1,121).

## 6. Sondensets und Teilen-Richtung

**Definition** (`lsne.chain.labels.probe_sets`, CLI `probe`). *fanout*: Positive =
ökonomischer fan-out ≥ 5 (≥ 5 verschiedene Empfänger von ETH mit Wert > 0), Negative =
fan-out = 1, alle Ären. *dao*: Positive = DAO-beteiligt, Negative = nicht beteiligt, nur
drain_crisis und post_fork. 1:1-Matching innerhalb (Ära, Gas-Dezil) wie in der Spec
(`--gas-bins 10`) und als Sensitivität innerhalb (Ära, 1/1 000-Gas-Quantil)
(`--gas-bins 1000`). Richtung: Difference-of-Means der über alle Positionen gemittelten
Residuen je Snapshot 0–4 (0 = Eingang Block 0, also Embeddings; 4 = Ausgang des letzten
Blocks), Sequenzen auf 512 Tokens gekappt. Trennschärfe: **Split-Half-AUC** (Richtung auf
der einen Hälfte der Paare geschätzt, AUC der Projektionen auf der anderen, Mittel über
beide Richtungen; `lsne.chain.source.holdout_auc`).

Kontrollen (Skript, Schritt `probe`): (a) 200 zufällige Einheitsrichtungen auf denselben
Hälften, AUC gefaltet als max(a, 1 − a) (großzügig für die Null); (b) Label-Permutation
(Positiv/Negativ innerhalb zufälliger Paare vertauscht, 100 ×, 95-%-Quantil der
Split-Half-AUC der neu geschätzten Richtung); (c) **untrainiertes Modell** (gleiche
Architektur, gleiches Vokabular, gleicher Seed); (d) **Oberflächen-Baseline**
(`surface_baseline_auc`: Tokenzahl, `[CALL]`-Zahl, `V:`-Zahl, Calls mit Wert, derselbe
Split); (e) **Längenbereinigung**: log(Tokenzahl) linear aus den gepoolten Zuständen
herausregressiert, dann dieselbe Split-Half-AUC.

**Größe und Matching-Qualität:**

| Sondenset | Paare | Pool pos / neg | Ären der Paare | Gas: Median-Faktor pos/neg | Anteil innerhalb Faktor 2 | SMD log Gas | Tokens (≤ 512) pos / neg | Oberflächen-AUC (kombiniert) |
|---|---|---|---|---|---|---|---|---|
| fanout, Gas-Dezile (Spec) | 106 | 106 / 193 181 | post_fork 79, dao_active 20, drain 7 | 63 | 0 % | 6,25 | 494 / 41 | 1,000 |
| fanout, 1 000 Gas-Strata | 52 | 106 / 193 181 | post_fork 25, dao_active 20, drain 7 | 6,5 | 15 % | 1,85 | 475 / 116 | 0,969 |
| dao, Gas-Dezile (Spec) | 500 | 7 697 / 135 538 | post_fork 290, drain 210 | 1,85 | 51 % | −0,67 | 79 / 54 | 0,610 |
| dao, 1 000 Gas-Strata | 500 | 7 697 / 135 538 | post_fork 273, drain 227 | 1,01 | 100 % | −0,02 | 66 / 72 | 0,684 |

Das Ära-Matching ist überall exakt. Das **Gas-Matching der Spec versagt bei fanout**: es
gibt nur 106 Positive, die alle im obersten Gas-Dezil des Pools liegen (Median 2,15 M
Gas), während die Negativen dort zufällig gezogene einfache Transaktionen sind (Median
23 000 Gas). Auch 1 000 Strata halbieren nur die Paarzahl und lassen einen Faktor 6,5. Die
Positiven sind zudem kaum divers: 68 der 106 rufen denselben Vertrag
(0x3f4a…227c, post_fork) auf, 95 % sind länger als 512 Tokens (Median 2 687) und werden
abgeschnitten, 64 % sind DAO-beteiligt (Negative 18 %). Bei *dao* ist das Matching mit
1 000 Strata praktisch perfekt (Gas und Länge balanciert).

**Trennschärfe pro Layer** (Split-Half-AUC; „untrainiert“ = Kontrolle (c);
„längenbereinigt“ = Kontrolle (e), trainiert / untrainiert):

*fanout, Gas-Dezile (106 Paare)* — Oberflächen-Baseline 1,000 (`[CALL]`-Zahl allein 0,999)

| Layer | AUC trainiert | AUC untrainiert | Zufallsrichtungen Mittel / p95 / max | Label-Permutation p95 | längenbereinigt trainiert / untrainiert |
|---|---|---|---|---|---|
| 0 | 0,923 | 0,926 | 0,72 / 0,88 / 0,98 | 0,59 | 0,503 / 0,500 |
| 1 | 0,906 | 0,967 | 0,75 / 0,92 / 0,98 | 0,59 | 0,494 / 0,483 |
| 2 | 0,908 | 0,984 | 0,75 / 0,95 / 0,99 | 0,60 | 0,515 / 0,499 |
| 3 | 0,954 | 0,986 | 0,75 / 0,95 / 1,00 | 0,59 | 0,496 / 0,481 |
| 4 | 0,946 | 0,989 | 0,74 / 0,94 / 1,00 | 0,59 | 0,503 / 0,467 |

*fanout, 1 000 Gas-Strata (52 Paare)* — Oberflächen-Baseline 0,969

| Layer | AUC trainiert | AUC untrainiert | Zufallsrichtungen Mittel / p95 / max | Label-Permutation p95 | längenbereinigt trainiert / untrainiert |
|---|---|---|---|---|---|
| 0 | 0,917 | 0,919 | 0,69 / 0,86 / 0,93 | 0,64 | 0,617 / 0,599 |
| 1 | 0,911 | 0,954 | 0,71 / 0,91 / 0,97 | 0,64 | 0,655 / 0,575 |
| 2 | 0,868 | 0,964 | 0,72 / 0,93 / 0,98 | 0,63 | 0,581 / 0,570 |
| 3 | 0,951 | 0,958 | 0,72 / 0,93 / 0,99 | 0,61 | 0,635 / 0,564 |
| 4 | 0,956 | 0,961 | 0,71 / 0,94 / 0,99 | 0,62 | 0,641 / 0,587 |

*dao, Gas-Dezile (500 Paare)* — Oberflächen-Baseline 0,610

| Layer | AUC trainiert | AUC untrainiert | Zufallsrichtungen Mittel / p95 / max | Label-Permutation p95 | längenbereinigt trainiert / untrainiert |
|---|---|---|---|---|---|
| 0 | 0,997 | 0,996 | 0,72 / 0,93 / 0,98 | 0,55 | 0,997 / 0,998 |
| 1 | 1,000 | 0,996 | 0,76 / 0,98 / 1,00 | 0,55 | 0,999 / 0,999 |
| 2 | 0,999 | 0,998 | 0,74 / 0,98 / 1,00 | 0,55 | 0,990 / 0,997 |
| 3 | 0,999 | 0,997 | 0,77 / 0,99 / 1,00 | 0,55 | 0,998 / 0,997 |
| 4 | 0,999 | 0,994 | 0,77 / 0,99 / 1,00 | 0,54 | 0,998 / 0,993 |

*dao, 1 000 Gas-Strata (500 Paare)* — Oberflächen-Baseline 0,684

| Layer | AUC trainiert | AUC untrainiert | Zufallsrichtungen Mittel / p95 / max | Label-Permutation p95 | längenbereinigt trainiert / untrainiert |
|---|---|---|---|---|---|
| 0 | 0,968 | 0,973 | 0,67 / 0,86 / 0,92 | 0,54 | 0,972 / 0,978 |
| 1 | 0,991 | 0,981 | 0,76 / 0,96 / 1,00 | 0,55 | 0,993 / 0,989 |
| 2 | 0,988 | 0,988 | 0,73 / 0,94 / 0,99 | 0,55 | 0,986 / 0,991 |
| 3 | 0,992 | 0,988 | 0,75 / 0,96 / 0,98 | 0,55 | 0,992 / 0,991 |
| 4 | 0,985 | 0,986 | 0,74 / 0,96 / 0,98 | 0,55 | 0,985 / 0,991 |

Die entrauschte Variante (eine Within-Class-Hauptkomponente entfernt) liegt bei 0,88–1,00
und ändert nichts am Bild (`probe_*/probe_summary.csv`).

**Ehrliche Lesart.** Formal ist die Teilen-Richtung in jedem Layer hoch trennscharf
(Split-Half-AUC 0,87–1,00, weit über der Label-Permutations-Null von 0,54–0,64). Das ist
aber **kein Befund über gelernte Repräsentationen**:
- Das **untrainierte** Modell trennt genauso gut oder besser (fanout 0,92–0,99, dao
  0,97–1,00), und schon Snapshot 0 (gemittelte Embeddings) trennt. Die Trennung kommt aus
  der Eingabe, nicht aus dem Training.
- *fanout* ist eine **Längen-/Größensonde**: die Oberflächen-Baseline erreicht 0,97–1,00;
  nach Herausrechnen der Länge fällt die AUC auf 0,49–0,52 (Gas-Dezile) bzw. 0,58–0,66
  (1 000 Strata, trainiert) gegenüber 0,56–0,60 (untrainiert) bei einer
  Permutations-Null von 0,61–0,64 — mit 52 Paaren nicht unterscheidbar.
- *dao* ist eine **Adress-Identitätssonde**: die DAO-Adressen stehen immer im Vokabular,
  schon ein zufälliges Embedding macht sie im Mittelwert sichtbar (Längenbereinigung ändert
  nichts, Oberflächen-Baseline nur 0,61–0,68). Selbst zufällige Richtungen erreichen im
  95-%-Quantil 0,86–0,99.
- Eine Teilen-Richtung zählt erst, wenn sie die untrainierte Kontrolle, die
  Oberflächen-Baseline und die längenbereinigte Null **gleichzeitig** schlägt. Das tut sie
  in diesem Lauf nirgends. Die Rohrleitung (Sondensets → Zustände → Richtung →
  `sharing_directions.pt` → Kontrollen) funktioniert; die Frage selbst ist offen.

## 7. Alignment Spurenmodell → Pythia-14m

300 zufällig gezogene Transaktionen (Seed 0; 258 `single`, 23 `none`, 19 `many_to_many`;
Ären dao_active 119, drain_crisis 89, post_fork 92), ITR (Median 14, max 369 Tokens) ins
Spurenmodell, `render_text` (Median 125 Zeichen) in `EleutherAI/pythia-14m@step98000`
(d = 128, 6 Layer). `lsne.transplant.layer_matching`, Mean-Pooling, Ridge, 20 % Hold-out
(60 Paare), Null = Hold-out-Ziele permutiert. Spurenmodell d = 256, Snapshots L0–L4;
Pythia L0–L6.

*Trainiertes Spurenmodell, Ridge α = 1 (Voreinstellung von `fit_alignment`):*

r2_holdout

| Quelle \ Ziel | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| L0 | 0,024 | 0,025 | 0,037 | 0,051 | 0,053 | 0,046 | 0,031 |
| L1 | 0,371 | 0,382 | 0,473 | 0,569 | 0,617 | 0,596 | 0,551 |
| L2 | 0,377 | 0,391 | 0,492 | 0,610 | 0,656 | 0,630 | 0,579 |
| L3 | 0,373 | 0,382 | 0,490 | 0,607 | 0,653 | 0,627 | 0,580 |
| L4 | 0,376 | 0,394 | 0,501 | 0,625 | **0,673** | 0,643 | 0,595 |

r2_permuted

| Quelle \ Ziel | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| L0 | −0,027 | −0,027 | −0,022 | −0,006 | −0,009 | −0,021 | −0,045 |
| L1 | −0,444 | −0,517 | −0,541 | −0,676 | −0,755 | −0,742 | −0,688 |
| L2 | −0,522 | −0,588 | −0,605 | −0,755 | −0,833 | −0,820 | −0,753 |
| L3 | −0,546 | −0,613 | −0,613 | −0,719 | −0,791 | −0,790 | −0,763 |
| L4 | −0,556 | −0,623 | −0,625 | −0,709 | −0,789 | −0,793 | −0,785 |

Bestes Paar nach r2_holdout: Quelle L4 → Pythia L4 (0,673 gegen −0,789 permutiert);
`layer_matching` wählt nach der Lücke `r2_holdout − r2_permuted` L2 → L4 (0,656 gegen
−0,833). Kontrollen (jeweils bestes r2_holdout, immer bei Pythia L4):

| Quelle | α = 1 | α = 100 | α = 10 000 |
|---|---|---|---|
| Spurenmodell trainiert (L4) | 0,673 | 0,554 | 0,065 |
| Spurenmodell untrainiert (L4) | 0,605 | 0,197 | −0,010 |
| nur zwei Längenmerkmale (log ITR-Tokens, log Text-Tokens) | 0,556 (α = 0,001) | | |

**Ehrliche Lesart.** Die Hold-out-R² liegen weit über der Permutations-Null, aber diese
Null ist hier schwach: Eine Abbildung, die überhaupt etwas vorhersagt, wird mit
vertauschten Zielen *schlechter* als der Mittelwert, deshalb ist r2_permuted stark
negativ und die „Lücke“ überzeichnet. Die aussagekräftigen Vergleiche sind die Kontrollen:
**zwei Längenmerkmale erklären allein 0,56** der Varianz der gemittelten Pythia-Zustände
(Pythia-L4), das untrainierte Spurenmodell 0,61, das trainierte 0,67. Das Training fügt
also +0,07 gegenüber untrainiert und +0,12 gegenüber der reinen Länge hinzu; bei stärkerer
Regularisierung (α = 100) behält das trainierte Modell 0,55 (≈ Länge), das untrainierte
fällt auf 0,20 — die gelernten Zustände kodieren Größe und Struktur der Transaktion
kompakter. Mit 60 Hold-out-Paaren (Streuung grob ±0,05–0,1), 240 Trainingspaaren bei
d_S = 256 (α = 1 ist fast interpolierend) und schablonenhaften Texten ist das ein
Funktionsnachweis der Kette `TraceLMSource` → `layer_matching`, **kein Nachweis einer
inhaltlichen Entsprechung** zwischen Spurenmodell und Pythia.


## 8. Hochrechnung: `blockgpt` über ganz 2016 auf einer A100

> **EXTRAPOLATION, nicht gemessen.** Gemessen sind nur die Datenmengen der Stichprobe und
> die CPU-Zahlen oben. Die A100-Zeiten folgen aus FLOPs pro Token und einer *angenommenen*
> Auslastung (MFU); sie sind bis zur ersten echten Messung auf Colab als Planungswerte zu
> lesen (Unsicherheit mindestens Faktor 2).

**Datenmenge 2016 (hochgerechnet aus 22 Profil-Chunks, jeder 100. Chunk):** im Mittel
6 362 Transaktionen, 572 700 rohe ITR-Tokens und 158 227 Trainings-Tokens (je Transaktion
auf 512 gekappt, plus `[EOS]`) und 2,41 MB Download (6 Tabellen) pro Chunk. Mit 2 135
Chunks: **≈ 13,6 M Transaktionen, ≈ 338 M Trainings-Tokens (≈ 1,2 Mrd. rohe Tokens),
≈ 5,2 GB Download.** Zwei Ausreißer prägen die Rohmenge: Chunk 2 378 000 (DoS-Angriffe
Sept./Okt. 2016) mit 6,5 M rohen Tokens und 19 MB, Chunk 2 678 000 (Zustandsbereinigung
nach Spurious Dragon) mit 3,2 M; das Kappen auf 512 Tokens begrenzt sie auf 0,22 M bzw.
0,44 M Trainings-Tokens. Die Stichprobe ist klein; ±30 % auf die Summen sind realistisch.

**Rechenaufwand.** `blockgpt` mit dem Vokabular dieses Laufs (V = 42 019): 68,2 M
Parameter (25,2 M im Transformer, je 21,5 M Ein-/Ausgabe-Embedding), 293 MFLOP pro
Trainings-Token (6 × (Nicht-Embedding + Unembedding) + Attention, `flops_per_token`). Mit
100 000 Adressen (V = 122 019, BlockGPT: 93 233; Hold-out-OOV dann ≈ 5 % statt 12 %):
150,2 M Parameter, 539 MFLOP pro Token. A100 bf16-Spitze 312 TFLOP/s, angenommene MFU
15–30 % (kleines Modell, d = 512):

| | Tokens/s (A100, Annahme) | eine Epoche 2016 (338 M Tokens) | Spec-Befehl 20 000 × 32 × 512 = 328 M Tokens |
|---|---|---|---|
| V = 42 019 | 160 000 – 320 000 | **0,3 – 0,6 h** | 0,3 – 0,6 h |
| V = 122 019 | 87 000 – 174 000 | **0,5 – 1,1 h** | 0,5 – 1,0 h |
| zum Vergleich: diese CPU, gemessen | 24,5–50 (`blockgpt`, s. Fußnote ²) | ≈ 1 900–3 800 h ≈ 80–160 Tage | – |

Dazu kommen Zwischenvalidierungen und Checkpoints (+10–20 %) und die Datenaufbereitung,
die auf Colab vermutlich länger dauert als das Training: Download 5,2 GB (hier gemessen
1,3–1,8 MB/s mit 4 Threads → 50–70 min; mit mehr Threads und Colab-Netz wohl weniger),
Serialisierung (gemessen 3,6–9,1 CPU-Sekunden pro Chunk unter Fremdlast, je nach Anteil
der DoS-Transaktionen → 2,1–5,4 CPU-Stunden, auf 8–12 vCPUs 10–40 min), Vokabular
(einprozessig, ≈ 10 min) und das Einlesen/Packen in `train` (einprozessig; hier 8,7 s für
7 M rohe Tokens → für 1,2 Mrd. ≈ 25 min, 338 M Tokens als int32 = 1,4 GB RAM). Die
Python-Schritte, die alle Shards zeilenweise lesen, skalieren linear mit: der
OOV-Bericht des Skripts (hier 154 s → für 2016 ≈ 2–3 h; mit `VOCAB_REPORT=0`
abschaltbar), und `probe`/`pair` durchsuchen für Merkmale und Paare jeweils das ganze
Korpus (hier ≈ 1–2 min je Aufruf → für 2016 ≈ 1–2 h; besser auf eine Ära-Teilmenge der
Shards oder mit `--limit`). Die Bigramm-Baseline über 338 M Tokens braucht einige GB RAM
(Colab mit hohem RAM). **Grobe
Planung: eine A100-Sitzung von 2–3 Stunden für Daten + eine Epoche `blockgpt` über 2016,
davon unter einer Stunde reines Training.** Das deckt sich mit der Schätzung in
`docs/BLOCKGPT_FEASIBILITY.md` (1–3 h für ≈ 0,5 Mrd. Tokens).

## 9. Wie die Richtung in Stufe L eingeht

1. **Paare:** `pair` liefert (ITR-Tokens, `render_text`) derselben Transaktion; ITR geht
   ins Spurenmodell, der Text in Pythia.
2. **Alignment:** `lsne.transplant.layer_matching(TraceLMSource(model_dir), mb_pythia,
   pairs)` gibt die R²-Tabelle über (Quell-Layer × Ziel-Layer), `layer_alignments(...)` je
   Ziel-Layer die beste Abbildung `H_T ≈ H_S W + b` (das braucht Injektion B auf allen
   Layern).
3. **Quellmerkmal:** `probe` schreibt `sharing_directions.pt` (Difference-of-Means je
   Residual-Snapshot, plus entrauschte Variante). Die Richtung des Quell-Layers, den das
   Alignment benutzt, ist `v_src`; `transfer_direction(align, v_src)` = `unit(Wᵀ v_src)`.
4. **Transplantation:** `transplant_experiment(mb, src, aligns, v_src, prompts, site, eps)`
   (A: Residual-Substitution, B: KV-Präfix, C: beides, gegen Zufallsrichtung, Identität,
   permutiertes Alignment).

**Integrationslücke (nicht in diesem Lauf behoben, liegt außerhalb dieses Auftrags):**
`transplant_experiment` kodiert in Quelle und Ziel *denselben* Prompt-String
(`src.encode(text)`), und `TraceLMSource.encode(str)` liest einen String als
leerzeichengetrennte ITR-Tokens. Für das Spurenmodell braucht es einen kleinen Adapter,
der zum Ziel-Text die ITR der Transaktion liefert (z. B. Nachschlagen in `pairs.jsonl`).
Ebenso nimmt `--transplant-source` in `lsne.cli` bisher nur `rotated_self`, `self` oder
ein HF-Modell `name@rev` an, kein Spurenmodell-Verzeichnis.

## 10. Grenzen

- **Maßstab:** `small`, 0,16 Epochen, 31 von 672 Chunks der DAO-Zeit; die beiden
  Drain-Chunks stellen 29 % der Tokens. Kein Befund über ein trainiertes BlockGPT.
- **Keine Baum-Positionsembeddings** (BlockGPT): die Position im Call-Baum steckt in
  `[D{Tiefe}]`-Tokens und der Breitensuche-Reihenfolge; Rotary nur über die Sequenz.
- **Keine ABI-Dekodierung:** Argumente sind die ersten 4 32-Byte-Wörter (Adresse, Zahl
  < 2^64 oder `[WORD]`); Selektoren bleiben unbenannte 4-Byte-IDs.
- **OOV-Adressen:** 11,9 % der Adressvorkommen im Hold-out (post_fork 16 %); Adressen sind
  Identitäten, das Modell lernt Verträge, nicht Rollen. Für 2016 vollständig: Budget
  ≥ 50 000–100 000.
- **Packing über Transaktionsgrenzen** ohne Attention-Maske (Standard-GPT-Packing).
- **Kappen auf 512 Tokens** pro Transaktion (Training und `TraceLMSource`): gerade die
  vielteiligen Transaktionen (Drain: 1 409 Tokens) werden abgeschnitten; die
  Mean-Pooling-Zustände sehen nur die ersten 512.
- **Keine Storage-Reads** (`storage_reads` nicht geladen).
- **Teilen-Labels:** ökonomischer fan-out ≥ 5 kommt in der Stichprobe 106-mal vor; das
  Gas-Matching der Spec versagt bei diesen Extremfällen (§6). Die DAO-Sonde ist zum
  großen Teil eine Adress-Identitätssonde (DAO-Adressen stehen immer im Vokabular).
- **Paare:** Zufallsstichprobe, 86 % einfache Überweisungen; der englische Text ist
  schablonenhaft, ein großer Teil der gemeinsamen Varianz ist Länge/Struktur (§7).
- **CPU-Zeiten** auf einer geteilten Maschine gemessen (verrauscht, eher zu langsam).

## 11. Nächster Schritt auf Colab (A100)

```bash
# im Repository-Checkout auf Colab; Cache und Läufe auf Drive
export OUT=/content/drive/MyDrive/LSNE/runs/trace_2016 CACHE=/content/drive/MyDrive/LSNE/cache/xatu
BLOCK_FROM=778000 BLOCK_TO=2913000 EVERY=1 EXTRA_RANGES="" WORKERS=12 \
  scripts/trace_e2e.sh download serialize                      # ganz 2016
N_ADDRESSES=100000 VOCAB_REPORT=0 scripts/trace_e2e.sh vocab   # Hold-out-OOV Adressen ≈ 5 %
SIZE=blockgpt STEPS=20000 BATCH=32 LR=3e-4 WARMUP=200 EVAL_EVERY=500 EVAL_BLOCKS=512 \
  SAVE_EVERY=1000 DEVICE=auto scripts/trace_e2e.sh baseline train evaluate untrained
N_PROBE=2000 N_PAIRS=2000 TARGET=EleutherAI/pythia-160m DEVICE=auto \
  scripts/trace_e2e.sh probe pair align report
```

Was der A100-Lauf zusätzlich braucht, damit die Sonden aussagekräftig werden:
1. **Längen-/Strukturkontrolle in den Sondensets:** zusätzlich auf Tokenzahl (oder
   Knotenzahl) matchen, nicht nur auf Gas-Dezil; die Oberflächen-Baseline
   (`surface_baseline_auc`) und die untrainierte Kontrolle bleiben Pflicht. Eine
   Teilen-Richtung zählt erst, wenn sie die untrainierte Kontrolle und die
   Oberflächen-Baseline schlägt.
2. **DAO-Sonde ohne Adress-Abkürzung:** dieselbe Sonde mit auf `[A_OOV]` maskierten
   DAO-Adressen bzw. innerhalb der DAO-Transaktionen (z. B. Split vs. Transfer).
3. **Ausgewogene Paare** (`pair --balance`) und mehr davon (≥ 2 000), Ziel Pythia-160m
   (Stufe L), Alignment gegen die Längen-Kontrolle.
4. Der Adapter aus §9 für `transplant_experiment` mit Spuren-Prompts.
