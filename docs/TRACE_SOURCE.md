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

**Ergebnis in Kürze** (Einzelheiten in §4–§8 und §12; alle Rohzahlen in `runs/trace_e2e/summary.json`;
alle Zahlen nach Fix-Runde 1, §12):

- **Daten:** 31 Xatu-Chunks (jeder 25. der Blöcke 1 428 000–2 099 999 plus Drain und
  Fork vollständig), 186 Dateien, 59,5 MB in 47 s; 225 854 Transaktionen, 7,05 M ITR-Tokens.
- **Vokabular:** 42 019 Tokens; OOV auf den zurückgehaltenen 10 %: Adressen 11,9 %,
  Selektoren 0,13 %, Topics 0,03 %.
- **Training `small`** (CPU, 12 min, 200 Schritte, 0,82 M Tokens = 0,16 Epochen):
  Validierungsverlust 10,70 → **1,175 nats** (alle Val-Blöcke), klar unter Unigramm
  (4,455) und Bigramm (1,526).
- **Labels (korrigiert, §12):** Netto-fan-out über die wirksamen Transfers; fan-out ≥ 5 hat
  nur noch **4** Transaktionen (vorher 106: durchgereichte ETH, Rückzahlungen, zurückgerollte
  Versuche), `many_to_many` 618 statt 12 874.
- **Sonden:** die fan-out-Sonde ist mit k ≥ 5 (4 Paare) und k ≥ 3 (15 Paare) nicht
  auswertbar; mit k ≥ 2 (471 längengematchte Paare) trennt das **untrainierte** Modell
  besser als das trainierte (Δ −0,08 bis −0,18, KI unter 0, auch nur auf dem Hold-out).
  Die DAO-Sonde bleibt eine Adress-Identitätssonde (beide Modelle ≈ 0,996–0,999). Kein
  Beleg für ein gelerntes Merkmal „Teilen“.
- **Alignment mit Pythia-14m** (300 Paare): r2_holdout bis 0,666 (L3 → L4) gegen
  r2_permuted −0,872; aber zwei Längenmerkmale allein erreichen 0,642, das untrainierte
  Modell 0,628. Mit 2 000 Paaren bleibt nach Längenresidualisierung ein kleiner, im
  KI positiver Gewinn des trainierten Modells (+0,05 [+0,01; +0,09] bei α = 1), auch
  auf 2 000 Paaren nur aus dem Hold-out (+0,07 [+0,02; +0,13]).
- **Hochrechnung (Extrapolation):** `blockgpt` über ganz 2016 (≈ 13,6 M Transaktionen,
  ≈ 338 M Trainings-Tokens, ≈ 5,2 GB Download) braucht auf einer A100 ≈ 0,3–1,1 h reines
  Training pro Epoche, mit Datenaufbereitung eine Sitzung von 2–3 h; auf dieser CPU wären
  es ≈ 80–160 Tage.

## 1. Was gebaut wurde

| Modul | Aufgabe |
|---|---|
| `lsne/chain/xatu.py` | URLs, resumierbarer Download (atomar, Backoff), normalisiertes Lesen der Xatu-Parquet-Chunks (Hashes/Adressen klein, Little-Endian-uint256 → int, `trace_address` → Tupel) |
| `lsne/chain/itr.py` | Call-Baum je Transaktion aus `trace_address`, Logs/Storage-Diffs nach Adresse **und Machbarkeit** an den Rahmen, dessen eigenes Gas sie bezahlen konnte (§12), Markierung zurückgerollter und `call_code`/`delegate_call`-Transfers, wirksame Transfers und Netto-Flüsse, ITR-Serialisierung im BlockGPT-Format (Breitensuche, `[START] [CALL] [D{Tiefe}] [CT:…] A:von A:an F:Selektor V:Wert G:Gas [INs] … [OUTs] [END]`, `[STATE] [W]`, `[LOG]`, Zahlen auf zwei signifikante Stellen `N{m}e{k}`) |
| `lsne/chain/text.py` | deterministische englische Beschreibung derselben Transaktion für Pythia (`render_text`; Wertfluss mit derselben Definition wie die Labels) |
| `lsne/chain/labels.py` | Teilen-Merkmale (Netto-fan-out/-fan-in über die wirksamen Transfers, auch im sichtbaren 512-Token-Fenster; DAO-Beteiligung nach der EIP-779-Liste, WithdrawDAO, Ären), Labels, gematchte Sondensets, Oberflächen-Baseline |
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
ersten Laufs liefen auf derselben Maschine Testläufe anderer Arbeitsstränge (Load Average
7–12); die Wiederholung nach Fix-Runde 1 (§12) lief auf einer weitgehend freien Maschine
(Training allein, die übrigen Schritte zu dritt parallel mit je 1 Thread). Alle Zeiten
geben die Größenordnung (±50 %), nicht die Leistung einer freien CPU.

| Schritt | Umfang | Wandzeit (Wiederholung; erster Lauf) |
|---|---|---|
| Download | 186 Dateien, 59,5 MB | – ; 47 s |
| Serialisierung | 31 Chunks → 225 854 Transaktionen, 7 051 538 ITR-Tokens, 39 MB gzip | 37 s (3 Prozesse); 12–28 s (4 Prozesse, zwei Läufe¹) |
| Split 90/10 + Vokabular | 203 304 Trainings-Transaktionen | 8 s + 10 s; 7 s + 8 s |
| Vokabular-Bericht (Hold-out-OOV, Budgets) | 22 550 Hold-out-Transaktionen | 164 s; 154 s (1 Kern, parallel zum Training) |
| Unigramm/Bigramm-Baseline | 10 044 + 1 099 Blöcke | 13 s; 10 s |
| Durchsatz `small`, 8 × 512, fwd+bwd+AdamW (`bench`, erster Lauf) | — | 16,2 s/Schritt = 253 Tokens/s² |
| Durchsatz `blockgpt`, 8 × 512, fwd+bwd+AdamW (`bench`, erster Lauf) | — | 167 s/Schritt = 24,5 Tokens/s² |
| Training `small` | 200 Schritte × 8 × 512 = 819 200 Tokens | 12,1 min (727 s, 1 154 Tokens/s); 30,0 min (1 797 s, s. u.) |
| Validierung auf allen Blöcken (fwd) | 1 099 Blöcke = 561 589 vorhergesagte Tokens | 409 s (1 Thread); 405 s |
| Sonden-CLI `probe` (Paare suchen, 5 Snapshots, Richtungen, untrainierter Zwilling, Kriterium) | fanout: je 4 Paare; dao: 391 / 434 Paare | 23 s / 28 s; 70 s / 82 s (1 Thread) |
| Sonden-Kontrollen des ersten Laufs (200 Zufallsrichtungen, 100 Permutationen, Längenbereinigung) | alle vier Sondensets | 111 s; 1 862 s |
| `pair` | 300 Paare | 12 s; 11 s |
| Layer-Matching (2 Quellen × 3 Ridge-α + Längenkontrolle, Pythia-14m) | 300 Paare, 5 × 7 Layer | 85 s (1 Thread); 43 s |
| 2016-Profil: Download / Serialisierung (erster Lauf) | 22 Chunks (90 neue Dateien, 42 MB) / 46 Chunks | 23 s / 104 s |

¹ Im ersten Lauf wurden die Shards nach einer Präzisierung der fan-out-Definition in
`labels.py` (ökonomisch, nur Transfers mit Wert > 0) mit `--overwrite` neu geschrieben; nach
Fix-Runde 1 (§12) ein zweites Mal (Netto-Flüsse der wirksamen Transfers, Zuordnung nach
Machbarkeit). Alle Zahlen in diesem Dokument stammen aus den Shards der Fix-Runde.
² Der Benchmark lief gleichzeitig mit einem zweiten Benchmark-Prozess; das Training selbst
erreichte mit `small` im ersten Lauf effektiv 495 Tokens/s (unter Fremdlast), in der
Wiederholung 1 154 Tokens/s. Die `blockgpt`-CPU-Zahl ist entsprechend eher zu niedrig.

Training (Wiederholung): 1 154 Tokens/s inklusive der 11 Zwischenvalidierungen (64 Blöcke),
etwa 2 s pro Schritt, ununterbrochen. Im ersten Lauf (495 Tokens/s, 6–9 s pro Schritt) wurde
das Training nach dem Checkpoint bei Schritt 20 abgebrochen und mit `--steps 200`
fortgesetzt; weil Schritt 20 noch im linearen Warm-up liegt und die Batch-Reihenfolge eine
reine Funktion von Seed und Schritt ist, war der fortgesetzte Lauf identisch mit einem
ununterbrochenen (ein Test der Resume-Logik).

## 5. Korpus, Vokabular, Training

**Korpus.** 225 854 Transaktionen aus 31 Chunks (Blocklevel-Zeilen ausgeschlossen), Ären:
pre_dao (die 757 Blöcke von Chunk 1 428 000 vor der Erzeugung der DAO in Block 1 428 757),
dao_active, drain_crisis, fork (die 10 Fork-Blöcke), post_fork. Länge in ITR-Tokens:
Median 14, 90 % ≤ 40, 99 % ≤ 230, Maximum 6 953; 0,26 % der Transaktionen sind länger als
512 und werden im Training auf 512 Tokens abgeschnitten (593 Transaktionen; auch die
DAO-Drain-Transaktion mit 1 409 Tokens). Die beiden Drain-Chunks liefern 29 % aller
Tokens (Split-Angriffe und Nachahmer). Transaktionen je Ära: pre_dao 2 949, dao_active
79 560, drain_crisis 70 263, fork 110, post_fork 72 972. DAO-beteiligt (berührt eine
Adresse der EIP-779-Liste — DAO, extraBalance, Kindes-DAOs und deren extraBalances —, den
Creator oder WithdrawDAO): 11 221 (dao_active 3 503, drain_crisis 3 260, fork 8, post_fork
4 450), davon 1 732 mit WithdrawDAO; 46 weitere nennen eine DAO-Adresse nur als
Call-Argument. Label-Verteilung (Netto-fan-out/-fan-in über die wirksamen Transfers, k = 5;
§12): `single` 208 040 (92,1 %), `none` 15 775 (7,0 %), `other` 1 417, `many_to_many` 618
(466 davon rekursive Splits der Drain-Chunks mit 3 Netto-Quellen), `distribution` 4,
`pooling` 0. Netto-fan-out: 0 → 20 705, 1 → 203 157, 2 → 1 919, 3 → 66, 4 → 3, ≥ 5 → **4**
Transaktionen (alle vier `distribution`; brutto über die wirksamen Transfers wären es 49,
nach der Zählung des ersten Laufs 106). Netto-fan-out ≥ 2 haben 1 992 Transaktionen, 71 %
davon mit genau einer Quelle (ein Vertrag verteilt eine Einzahlung). Hoher fan-out ist 2016
also noch seltener als im ersten Lauf angenommen: die meisten „Mehrfach-Empfänger“ des
ersten Laufs waren durchgereichte ETH, Rückzahlungen oder zurückgerollte Versuche.

**Vokabular** (nur aus dem Trainingsteil, 203 304 Transaktionen, 6,36 M Tokens):

| Teil | Tokens | davon im Training benutzt |
|---|---|---|
| geschlossen: Wert-Tokens `V:N…` | 7 030 | 2 899 |
| geschlossen: Gas-Tokens `G:N…` | 7 030 | 268 |
| geschlossen: nackte Zahlen (Argumente) `N…` | 7 030 | 1 453 |
| geschlossen: Tiefe `[D0]`…`[D32]`, Slot-Buckets, Call-Typen, Fehler, Struktur | 105 | 87 |
| offen: Adressen (Top 20 000 von 65 108 verschiedenen; DAO-Adressen immer) | 20 000 | — |
| offen: Funktionsselektoren (alle) | 639 | — |
| offen: Event-Topics (alle) | 185 | — |
| **Summe** | **42 019** | |

78 % der geschlossenen Klasse (16 488 Tokens, fast alle Zahlentokens) kommen im Training
nie vor (erster Lauf: 16 840; seit §12 werden Argumente bis 2¹⁶⁰ als Zahl kodiert); sie kosten nur Embedding-Zeilen (bei `blockgpt` 2 × 8,6 M Parameter).

**OOV auf den zurückgehaltenen 10 %** (22 550 Transaktionen, 695 133 Tokens; Anteil der
Vorkommen der jeweiligen Klasse, die auf `[X_OOV]` fallen):

| | Adressen | Selektoren | Topics | alle offenen | `[UNK]` |
|---|---|---|---|---|---|
| Hold-out gesamt | **11,9 %** | **0,13 %** | **0,03 %** | 9,7 % | 0 |
| pre_dao (291 Tx) | 11,9 % | 0 | 0 | | |
| dao_active (7 931 Tx) | 13,9 % | 0,04 % | 0,06 % | | |
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
| 20 | 81 920 | 8,187 | 6,366 |
| 40 | 163 840 | 3,995 | 2,607 |
| 60 | 245 760 | 2,105 | 1,832 |
| 80 | 327 680 | 1,756 | 1,611 |
| 100 | 409 600 | 1,608 | 1,435 |
| 120 | 491 520 | 1,431 | 1,222 |
| 140 | 573 440 | 1,407 | 1,159 |
| 160 | 655 360 | 1,260 | 1,102 |
| 180 | 737 280 | 1,259 | 1,033 |
| 200 | 819 200 | 1,187 | 0,996 |

**Baselines** (nats pro Token, dieselben gepackten Blöcke, dieselben Zielpositionen):

| | alle 1 099 Val-Blöcke | 64 Monitor-Blöcke |
|---|---|---|
| gleichverteilt, ln V | 10,646 | 10,646 |
| Unigramm (Trainingszählung, Add-one) | 4,455 | 4,592 |
| Bigramm (interpoliert, λ = 0,95 auf Trainings-Dev gewählt) | 1,526 | 1,515 |
| **Modell nach 200 Schritten** | **1,175** | **0,996** |

Der Validierungsverlust liegt **klar unter der Unigramm-Baseline** (−3,28 nats) und auch
unter der Bigramm-Baseline (−0,35 nats); erster Lauf (vor §12): 1,121 gegen 4,422 / 1,490.
Aufschlüsselung nach Klasse des vorhergesagten Tokens (alle Val-Blöcke, nats):

| Zielklasse | Anteil | Modell | Bigramm | Unigramm |
|---|---|---|---|---|
| Struktur (`[START]`, `[END]`, `[CALL]`, …) | 46,4 % | 0,09 | 0,43 | 2,95 |
| Adresse im Vokabular | 12,9 % | 4,04 | 4,49 | 7,87 |
| Wert `V:` | 7,4 % | 4,00 | 3,89 | 7,41 |
| Tiefe `[D…]` | 5,8 % | 0,27 | 1,10 | 3,92 |
| Call-Typ | 5,8 % | 0,13 | 0,22 | 3,09 |
| Gas `G:` | 5,7 % | 1,70 | 2,54 | 6,03 |
| Selektor `F:` | 5,7 % | 0,69 | 1,02 | 4,77 |
| `[EOS]` (Transaktionsende) | 4,0 % | 0,01 | 0,06 | 3,24 |
| `[A_OOV]` | 2,5 % | 2,96 | 2,62 | 4,13 |
| Slot-Bucket | 1,7 % | 0,85 | 1,84 | 5,43 |
| Zahl-Argument | 1,2 % | 4,22 | 4,78 | 8,68 |
| Topic | 0,9 % | 1,49 | 2,28 | 7,49 |

Gelernt ist vor allem die Grammatik der Spuren (Struktur, Tiefe, Call-Typ, Gas, Selektor
nach Adresse). Bei Werten und bei `[A_OOV]` ist das Modell nach 0,16 Epochen noch
schlechter als das Bigramm; Adressen bleiben mit ~4 nats das Schwierigste. Der
Gesamtverlust liegt etwas über dem des ersten Laufs (1,175 gegen 1,121; vor allem Tiefe
0,27 gegen 0,15, Topic 1,49 gegen 0,77, Zahl-Argument 4,22 gegen 3,89). Vermutlich (nicht
getrennt geprüft) folgt das aus §12: Logs und Storage-Diffs stehen jetzt beim Rahmen, der
sie ausgelöst hat, statt gesammelt beim ersten Knoten mit der Adresse, was die Abfolge der
Blöcke weniger regelmäßig macht, und 7 370 frühere `[WORD]`-Argumente sind jetzt
Zahl-Tokens. Der Train-Loss
liegt über dem Val-Loss, weil er über die frühen, schlechteren Schritte eines Intervalls
mittelt und die 64 Monitor-Blöcke leichter als der Durchschnitt sind (alle Blöcke: 1,175).

## 6. Sondensets und Teilen-Richtung

**Definition** (`lsne.chain.labels.probe_sets`, CLI `probe`). *fanout*: Positive =
Netto-fan-out ≥ 5 (≥ 5 Adressen mit Netto-Zufluss über die wirksamen ETH-Transfers, §12),
Negative = Netto-fan-out = 1, beides auch im sichtbaren 512-Token-Fenster, alle Ären.
*dao*: Positive = DAO-beteiligt (EIP-779-Liste, Creator, WithdrawDAO), Negative = nicht
beteiligt und ohne DAO-Adresse als Call-Argument, nur drain_crisis und post_fork.
1:1-Matching innerhalb (Ära, Gas-Dezil) wie in der Spec (`--gas-bins 10`) und als
Sensitivität innerhalb (Ära, 1/1 000-Gas-Quantil) (`--gas-bins 1000`); Paare mit einer
schon vorhandenen sichtbaren Sequenz werden entfernt (§12). Richtung: Difference-of-Means
der über alle Positionen gemittelten Residuen je Snapshot 0–4 (0 = Eingang Block 0, also
Embeddings; 4 = Ausgang des letzten Blocks), Sequenzen auf 512 Tokens gekappt.
Trennschärfe: **Split-Half-AUC** (Richtung auf der einen Hälfte der Paare geschätzt, AUC der
Projektionen auf der anderen, Mittel über beide Richtungen; `lsne.chain.source.holdout_auc`).

Kontrollen (Skript, Schritt `probe`): (a) 200 zufällige Einheitsrichtungen auf denselben
Hälften, AUC gefaltet als max(a, 1 − a) (großzügig für die Null); (b) Label-Permutation
(Positiv/Negativ innerhalb zufälliger Paare vertauscht, 100 ×, 95-%-Quantil der
Split-Half-AUC der neu geschätzten Richtung); (c) **untrainiertes Modell** (gleiche
Architektur, gleiches Vokabular, gleicher Seed); (d) **Oberflächen-Baseline**
(`surface_baseline_auc`: Tokenzahl, `[CALL]`-Zahl, `V:`-Zahl, Calls mit Wert, derselbe
Split); (e) **Längenbereinigung**: log(Tokenzahl) linear aus den gepoolten Zuständen
herausregressiert, dann dieselbe Split-Half-AUC.

**Größe und Matching-Qualität:**

| Sondenset | Paare (vor Entfernen der Duplikate) | Pool pos / neg | Ären der Paare | Gas: Median-Faktor pos/neg | Anteil innerhalb Faktor 2 | SMD log Gas | Tokens (≤ 512) pos / neg | Oberflächen-AUC (kombiniert) |
|---|---|---|---|---|---|---|---|---|
| fanout, Gas-Dezile (Spec) | 4 (4) | 4 / 203 155 | drain 3, dao_active 1 | 2,97 | 0 % | 3,79 | 137 / 28 | 1,000 |
| fanout, 1 000 Gas-Strata | 4 (4) | 4 / 203 155 | drain 3, dao_active 1 | 1,02 | 100 % | −0,00 | 137 / 86 | 1,000 |
| dao, Gas-Dezile (Spec) | 391 (500) | 7 710 / 135 486 | post_fork 222, drain 169 | 1,67 | 53 % | −0,79 | 57 / 62 | 0,747 |
| dao, 1 000 Gas-Strata | 434 (500) | 7 710 / 135 486 | post_fork 253, drain 181 | 1,01 | 100 % | −0,01 | 61 / 72 | 0,685 |

Das Ära-Matching ist überall exakt. **fanout ist nicht auswertbar:** mit der korrigierten
Definition gibt es in der Stichprobe nur 4 Transaktionen mit Netto-fan-out ≥ 5 (alle vier
`distribution`: eine Quelle, 6–10 Empfänger; 3 aus den Drain-Chunks). Mit 4 Paaren hat jede
Split-Hälfte 2 Paare; trainiertes Modell, untrainiertes Modell, Zufallsrichtungen und
Oberflächen-Baseline erreichen alle AUC 1,0 (Tabellen in
`runs/trace_e2e/probe_fanout_g*/separability.csv`). Im ersten Lauf hatte dieselbe Sonde 106
Positive, die alle im obersten Gas-Dezil lagen und zu 95 % länger als 512 Tokens waren; 57
davon erreichten fan-out ≥ 5 nur über einen zurückgerollten Call, die übrigen über
durchgereichte ETH (§12). Bei *dao* ist das Matching mit 1 000 Strata praktisch perfekt
(Gas balanciert, Länge 61 gegen 72 Tokens); das Entfernen wiederholter Sequenzen kostet 22 %
bzw. 13 % der Paare.

**Trennschärfe pro Layer** (Split-Half-AUC; „untrainiert“ = Kontrolle (c);
„längenbereinigt“ = Kontrolle (e), trainiert / untrainiert):

*dao, Gas-Dezile (391 Paare)* — Oberflächen-Baseline 0,747

| Layer | AUC trainiert | AUC untrainiert | Zufallsrichtungen Mittel / p95 / max | Label-Permutation p95 | längenbereinigt trainiert / untrainiert |
|---|---|---|---|---|---|
| 0 | 0,999 | 0,999 | 0,70 / 0,91 / 0,98 | 0,55 | 0,999 / 0,999 |
| 1 | 1,000 | 1,000 | 0,77 / 0,98 / 0,99 | 0,55 | 1,000 / 1,000 |
| 2 | 1,000 | 0,996 | 0,75 / 0,98 / 1,00 | 0,54 | 1,000 / 0,997 |
| 3 | 1,000 | 0,998 | 0,73 / 0,97 / 0,99 | 0,55 | 1,000 / 0,998 |
| 4 | 1,000 | 0,994 | 0,72 / 0,98 / 0,99 | 0,54 | 1,000 / 0,994 |

*dao, 1 000 Gas-Strata (434 Paare)* — Oberflächen-Baseline 0,685

| Layer | AUC trainiert | AUC untrainiert | Zufallsrichtungen Mittel / p95 / max | Label-Permutation p95 | längenbereinigt trainiert / untrainiert |
|---|---|---|---|---|---|
| 0 | 0,977 | 0,977 | 0,65 / 0,84 / 0,95 | 0,55 | 0,980 / 0,979 |
| 1 | 0,996 | 0,984 | 0,75 / 0,97 / 0,99 | 0,55 | 0,997 / 0,986 |
| 2 | 0,998 | 0,987 | 0,73 / 0,95 / 0,99 | 0,54 | 0,998 / 0,987 |
| 3 | 0,996 | 0,988 | 0,72 / 0,94 / 0,98 | 0,55 | 0,996 / 0,989 |
| 4 | 0,994 | 0,986 | 0,71 / 0,94 / 0,98 | 0,55 | 0,995 / 0,988 |

Die entrauschte Variante (eine Within-Class-Hauptkomponente entfernt) ändert nichts am Bild
(`probe_*/probe_summary.csv`).

**Ehrliche Lesart.** Formal ist die DAO-Richtung in jedem Layer hoch trennscharf
(Split-Half-AUC 0,98–1,00, weit über der Label-Permutations-Null von 0,54–0,55). Das ist
aber **kein Befund über gelernte Repräsentationen**:
- Das **untrainierte** Modell trennt fast genauso gut (0,98–1,00), und schon Snapshot 0
  (gemittelte Embeddings) trennt. Die Trennung kommt aus der Eingabe, nicht aus dem
  Training; der kleine Vorsprung des trainierten Modells (≤ 0,012) ist ein
  Identitätseffekt der DAO-Adress-Tokens (siehe „Kontrollierte Sonden“).
- *dao* ist eine **Adress-Identitätssonde**: die DAO-Adressen stehen immer im Vokabular,
  schon ein zufälliges Embedding macht sie im Mittelwert sichtbar (Längenbereinigung ändert
  nichts, Oberflächen-Baseline nur 0,69–0,75). Selbst zufällige Richtungen erreichen im
  95-%-Quantil 0,84–0,98.
- *fanout* — die Sonde, die „Teilen“ meinen soll — hat mit der korrigierten Definition in
  dieser Stichprobe keine auswertbaren Paare; der erste Lauf hatte statt Teilen
  Transaktionslänge und Zählartefakte gemessen.
- Eine Teilen-Richtung zählt erst, wenn sie die untrainierte Kontrolle, die
  Oberflächen-Baseline und die längenbereinigte Null **gleichzeitig** schlägt. Das tut sie
  in diesem Lauf nirgends. Die Rohrleitung (Sondensets → Zustände → Richtung →
  `sharing_directions.pt` → Kontrollen) funktioniert; die Frage selbst ist offen.

## 7. Alignment Spurenmodell → Pythia-14m

300 zufällig gezogene Transaktionen, die ins 512-Token-Fenster passen (Seed 0; §12: längere
Transaktionen werden nicht gepaart, ihr Text beschreibt Knoten außerhalb des Fensters;
dadurch ist es eine andere Stichprobe als im ersten Lauf): 284 `single`, 15 `none`,
1 `other`; Ären dao_active 116, drain_crisis 93, post_fork 84, pre_dao 6, fork 1; ITR
(Median 14, max 472 Tokens) ins Spurenmodell, `render_text` (Median 125 Zeichen) in
`EleutherAI/pythia-14m@step98000` (d = 128, 6 Layer). `lsne.transplant.layer_matching`,
Mean-Pooling, Ridge, 20 % Hold-out (60 Paare), Null = Hold-out-Ziele permutiert.
Spurenmodell d = 256, Snapshots L0–L4; Pythia L0–L6.

*Trainiertes Spurenmodell, Ridge α = 1 (Voreinstellung von `fit_alignment`):*

r2_holdout

| Quelle \ Ziel | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| L0 | 0,022 | 0,029 | 0,041 | 0,061 | 0,060 | 0,058 | 0,050 |
| L1 | 0,349 | 0,358 | 0,474 | 0,634 | 0,661 | 0,623 | 0,534 |
| L2 | 0,349 | 0,352 | 0,466 | 0,626 | 0,653 | 0,619 | 0,530 |
| L3 | 0,333 | 0,378 | 0,472 | 0,639 | **0,666** | 0,629 | 0,527 |
| L4 | 0,323 | 0,373 | 0,471 | 0,612 | 0,648 | 0,618 | 0,530 |

r2_permuted

| Quelle \ Ziel | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| L0 | −0,023 | −0,014 | −0,013 | −0,008 | −0,009 | −0,010 | −0,015 |
| L1 | −0,504 | −0,492 | −0,561 | −0,662 | −0,688 | −0,676 | −0,661 |
| L2 | −0,564 | −0,541 | −0,626 | −0,713 | −0,739 | −0,728 | −0,735 |
| L3 | −0,592 | −0,589 | −0,678 | −0,865 | −0,872 | −0,832 | −0,778 |
| L4 | −0,632 | −0,627 | −0,712 | −0,925 | −0,919 | −0,869 | −0,796 |

Bestes Paar nach r2_holdout: Quelle L3 → Pythia L4 (0,666 gegen −0,872 permutiert);
`layer_matching` wählt nach der Lücke `r2_holdout − r2_permuted` L4 → L4 (0,648 gegen
−0,919). Kontrollen (jeweils bestes r2_holdout, fast immer bei Pythia L4):

| Quelle | α = 1 | α = 100 | α = 10 000 |
|---|---|---|---|
| Spurenmodell trainiert | 0,666 (L3 → L4) | 0,590 (L4 → L4) | 0,123 |
| Spurenmodell untrainiert | 0,628 (L4 → L4) | 0,240 (L4 → L3) | −0,003 |
| nur zwei Längenmerkmale (log ITR-Tokens, log Text-Tokens) | 0,642 (α = 0,001) | | |

**Ehrliche Lesart.** Die Hold-out-R² liegen weit über der Permutations-Null, aber diese
Null ist hier schwach: Eine Abbildung, die überhaupt etwas vorhersagt, wird mit
vertauschten Zielen *schlechter* als der Mittelwert, deshalb ist r2_permuted stark
negativ und die „Lücke“ überzeichnet. Die aussagekräftigen Vergleiche sind die Kontrollen:
**zwei Längenmerkmale erklären allein 0,64** der Varianz der gemittelten Pythia-Zustände
(Pythia-L4), das untrainierte Spurenmodell 0,63, das trainierte 0,67 (erster Lauf, andere
300 Paare: 0,56 / 0,61 / 0,67). Das Training fügt also +0,04 gegenüber untrainiert und
+0,02 gegenüber der reinen Länge hinzu; bei stärkerer Regularisierung (α = 100) behält das
trainierte Modell 0,59 (≈ Länge), das untrainierte fällt auf 0,24 — die gelernten Zustände
kodieren Größe und Struktur der Transaktion kompakter. Mit 60 Hold-out-Paaren (Streuung
grob ±0,05–0,1), 240 Trainingspaaren bei d_S = 256 (α = 1 ist fast interpolierend) und
schablonenhaften Texten ist das ein Funktionsnachweis der Kette `TraceLMSource` →
`layer_matching`, **kein Nachweis einer inhaltlichen Entsprechung** zwischen Spurenmodell
und Pythia.

### Kontrollierte Sonden (Runde 2)

Stand 26.10.2026, nach Fix-Runde 1 (§12) neu gerechnet: dasselbe neu trainierte Modell
(`trace_lm_small`, 200 Schritte) und dieselben neu serialisierten Shards wie oben. Anlass sind
die vier Befunde aus §6/§7: das untrainierte Modell trennt die Sondensets
so gut wie das trainierte; fanout folgt der Transaktionslänge; die DAO-Sonde folgt den
DAO-Adress-Tokens; zwei Längenmerkmale erklären den Großteil des Alignment-R².

**Was neu ist** (`lsne/chain/labels.py`, `source.py`, `cli.py`; Tests in
`tests/test_chain_data.py`, `tests/test_chain_model.py`):

1. **Längen-Matching** (`probe_sets(..., match="length")`, CLI `--match length`): jede
   Positive bekommt 1:1 eine Negative derselben Ära, deren ITR-Tokenzahl innerhalb des
   Faktors 1,25 liegt (`|log n_pos − log n_neg| ≤ log 1,25`), nächste Länge zuerst, ohne
   Zurücklegen, schwierigste Positive zuerst; Positive ohne zulässigen Partner bleiben
   ungepaart (kein stiller Rückfall). Balance wird als standardisierte Mittelwertdifferenz
   (SMD) für Länge, Gas, Knoten, Calls mit Wert, Tiefe und State-Writes berichtet
   (`match_table.attrs["balance"]`, `probe_report.json`).
2. **Identitätsmaskierung** (`mask_tokens`, CLI `--mask dao|all_addresses`): vor dem
   Kodieren werden in **beiden** Sets die Tokens der DAO, von extraBalance, Creator und
   WithdrawDAO (`dao`) bzw. alle Adress-Tokens (`all_addresses`) durch `[A_OOV]` ersetzt.
   Die Oberflächen-Baseline zählt dafür zusätzlich `[A_OOV]` (`n_a_oov`).
3. **Sonden-Kriterium** (`probe_criterion`, `criterion_table`, CLI `probe` schreibt
   `criterion.csv`): Split-Half-AUC des trainierten Modells minus dieselbe AUC des
   **untrainierten Zwillings** (`untrained_twin`: gleiche Konfiguration, gleiches Vokabular,
   Initialisierung mit dem Trainings-Seed 0 — bitgleich mit `trace_lm_untrained` und mit
   den Anfangsgewichten des Trainings), gleicher Split. KI: Bootstrap über gematchte Paare
   (1 000 Ziehungen, beide Hälften getrennt mit Zurücklegen gezogen, Richtung jedes Mal neu
   geschätzt, dieselben Ziehungen für beide Modelle; zum Vergleich auch nur die
   Auswertungshälfte gezogen, `delta_ci_eval_*`). Dazu Label-Permutation (200 ×, Positive
   und Negative innerhalb zufälliger Paare getauscht), 200 Zufallsrichtungen und die
   Oberflächen-Baseline. **Eine Richtung zählt nur, wenn das 95-%-KI von Δ = AUC_trainiert −
   AUC_untrainiert über 0 liegt und die trainierte AUC über dem 95-%-Quantil der
   Permutations-Null** (Bonferroni über die 5 Snapshots zusätzlich: `counts_adj`). Die
   Regel ist seit dem ersten Runde-2-Lauf unverändert. Ob sie im Code stand, *bevor* jener
   Lauf startete, ist aus dessen Herkunftsangaben nicht prüfbar (die Manifeste nannten
   einen Commit ohne `--threads` und ohne `delta_ci_eval`, also nicht den ausgeführten
   Code); seit Fix-Runde 1 trägt jeder Bericht und jedes Manifest `code_sha256` (Inhalt von
   `lsne/chain` und der importierten `lsne`-Module, `lsne.chain.cli.code_fingerprint`), die
   Läufe unten nennen alle denselben Wert `4e0ec556023a906ce6119451`. Der Stand danach
   (`62086f63cb32c406bcb88faf`) unterscheidet sich davon nur im Docstring und in der
   Fehlermeldung von `lsne.chain.pairs.resolve_shards` (Hinweis auf die Optionssyntax von
   Stufe L), nicht in einer gerechneten Zahl.
4. **Alignment-Kontrollen** (`length_covariates`, `residualize`, `alignment_controls`, CLI
   `align`): Quell- und Zielzustände werden auf die Längenmerkmale log ITR-Tokens (sichtbar,
   ≤ 512), log1p Knoten, log1p Calls mit Wert (`len3`; Variante `len3+text` zusätzlich log
   Pythia-Tokens des Texts) residualisiert, Koeffizienten nur aus den Trainingspaaren des
   Splits von `fit_alignment`. Berichtet wird r2_holdout roh, residualisiert, für die
   Längenmerkmale allein und für die untrainierte Quelle, der Gewinn trainiert − untrainiert
   und dessen Bootstrap-KI über die Hold-out-Paare (Abbildungen fest).

5. **Fix-Runde 1:** fan-out-Sonde auf Netto-Flüssen der wirksamen Transfers und im
   sichtbaren Fenster (`fan_out_eth_visible`, Match-Tabelle `pos/neg_fan_out_eth_visible`),
   DAO-Sonde mit der EIP-779-Liste und ohne Negative, die eine DAO-Adresse nur als Argument
   nennen; Paare mit wiederholter sichtbarer (maskierter, abgeschnittener) Sequenz werden vor
   Split und Bootstrap entfernt (`n_pairs_dropped_duplicates`; im ersten Lauf hatten z. B.
   19 % der Positiven der DAO-Sonde mit allen Adressen maskiert eine exakte Kopie in der
   anderen Hälfte); `--split heldout` beschränkt Sonden und Paare auf den Hold-out des
   Spurenmodells, `lm_split` im Bericht zählt die Zugehörigkeit.

```bash
O=runs/trace_e2e/r2; M=runs/trace_e2e/trace_lm_small; I=runs/trace_e2e/itr
python -m lsne.chain.cli probe --model $M --in $I --kind fanout --k 5 --match length --n 500 \
    --threads 1 --out $O/probe_fanout_len_k5            # ebenso --k 3, --k 2; --split heldout
python -m lsne.chain.cli probe --model $M --in $I --kind dao --match length --mask dao --n 500 \
    --threads 1 --out $O/probe_dao_len_maskdao          # ebenso --mask none, --mask all_addresses
python -m lsne.chain.cli pair --in $I --n 2000 --seed 0 --out $O/pairs_2000.jsonl
python -m lsne.chain.cli pair --in $I --n 2000 --seed 0 --split heldout --model $M --out $O/pairs_2000_heldout.jsonl
python -m lsne.chain.cli align --model $M --pairs runs/trace_e2e/pairs.jsonl \
    --target EleutherAI/pythia-14m@step98000 --alphas 1,100 --threads 1 --out $O/align_300
```

(Skript: `runs/trace_e2e/r2/run_probes.sh`.) **Laufzeit** (CPU, 1 Thread pro Prozess, bis zu
drei Prozesse parallel): Sonden 23 s (k ≥ 5), 33 s (k ≥ 3), 121 s (k ≥ 2), 46 s (k ≥ 2,
Hold-out), 61–75 s je DAO-Variante; Alignment 29 s (300 Paare) bzw. 115–121 s (2 000 Paare).
Auf geteilter CPU `--threads 1` (im ersten Lauf lief die Voreinstellung mit 4 Threads pro
Prozess unter Fremdlast rund 50-mal langsamer, bei identischen Ergebnissen).
**Unterstichproben:** höchstens 500 gematchte Paare je Sonde, danach werden Paare mit einer
schon vorhandenen sichtbaren Sequenz entfernt; Alignment auf den 300 Paaren aus §7, auf
2 000 Paaren und auf 2 000 Paaren nur aus dem Hold-out des Spurenmodells.

**Matching-Qualität** (SMD Positive − Negative vor dem Entfernen der Duplikate; „Calls mit
Wert“ = wirksame Wert-Calls):

| Sondenset | Paare (gematcht; Duplikate) | Pool pos / neg | SMD log Länge | SMD log Gas | SMD log Knoten | SMD Calls mit Wert | Oberflächen-AUC (bestes Einzelmerkmal) |
|---|---|---|---|---|---|---|---|
| fanout k ≥ 5 | 4 (4; 0) | 4 / 203 155 | −0,03 | −0,56 | 1,30 | 4,29 | 1,000 (`[CALL]`-Zahl 1,000) |
| fanout k ≥ 3 | 15 (20; 5) | 49 / 203 155 | −0,01 | 0,52 | 0,21 | 2,99 | 1,000 (Calls mit Wert 1,000) |
| fanout k ≥ 2 | 471 (500 von 1 210; 29) | 1 985 / 203 155 | 0,07 | 0,14 | 0,03 | 0,20 | 0,841 (Calls mit Wert 0,910) |
| fanout k ≥ 2, nur Hold-out | 106 (109; 3) | 183 / 20 342 | 0,08 | 0,23 | 0,01 | 0,25 | 0,864 (Calls mit Wert 0,910) |
| dao, ohne Maske | 450 (500 von 6 030; 50) | 7 710 / 135 486 | −0,03 | −0,36 | −0,44 | −1,10 | 0,795 (Calls mit Wert 0,726) |
| dao, DAO-Adressen maskiert | 450 (dieselben 500; 50) | | | | | | 0,988 (`[A_OOV]`-Zahl 0,983) |
| dao, alle Adressen maskiert | 356 (dieselben 500; 144) | | | | | | 0,668 (Calls mit Wert 0,651) |
| dao, alle maskiert, nur Hold-out | 365 (500 von 609; 135) | 764 / 13 542 | −0,07 | −0,46 | −0,55 | −1,00 | 0,801 (Calls mit Wert 0,629) |

Pool und Paare der fan-out-Sonde verlangen das Label auch im sichtbaren Fenster (k ≥ 3: 24
Positive fallen dadurch weg). Das Längen-Matching gelingt überall (|SMD| ≤ 0,08). Mit dem
Netto-fan-out ist die Oberflächen-Abkürzung schwächer (k ≥ 2: 0,84 statt 0,998 im ersten
Lauf): die Zahl der Wert-Calls bestimmt das Label nicht mehr, Negative mit fan-out 1 haben
oft mehrere Wert-Calls (Durchreichen). Bei der DAO-Sonde bleibt der Wert-Konfundierer
(DAO-Transaktionen sind meist Token-Aufrufe ohne ETH). Das Entfernen wiederholter Sequenzen
kostet bis zu 29 % der Paare (DAO, alle Adressen maskiert: viele gleichförmige
Token-Aufrufe sehen ohne Adressen identisch aus). Elemente aus dem Hold-out des
Spurenmodells: 10 % (k ≥ 2: 95 von 942, dao: 89 von 900), in den `heldout`-Zeilen 100 %.

**AUC trainiert / untrainiert** (Split-Half, Seed 0):

| Sondenset | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| fanout k ≥ 5 (4) | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 / 1,000 |
| fanout k ≥ 3 (15) | 0,716 / 0,706 | 0,720 / 0,675 | 0,792 / 0,760 | 0,781 / 0,750 | 0,730 / 0,760 |
| fanout k ≥ 2 (471) | 0,799 / 0,936 | 0,846 / 0,927 | 0,738 / 0,917 | 0,719 / 0,891 | 0,718 / 0,868 |
| fanout k ≥ 2, Hold-out (106) | 0,782 / 0,879 | 0,719 / 0,903 | 0,721 / 0,886 | 0,795 / 0,868 | 0,804 / 0,865 |
| dao, ohne Maske (450) | 0,997 / 0,996 | 0,999 / 0,998 | 0,999 / 0,996 | 0,999 / 0,996 | 0,998 / 0,995 |
| dao, DAO maskiert (450) | 0,992 / 0,990 | 0,990 / 0,990 | 0,977 / 0,984 | 0,962 / 0,984 | 0,945 / 0,983 |
| dao, alle maskiert (356) | 0,901 / 0,879 | 0,894 / 0,943 | 0,865 / 0,936 | 0,878 / 0,940 | 0,868 / 0,940 |
| dao, alle maskiert, Hold-out (365) | 0,893 / 0,871 | 0,916 / 0,940 | 0,868 / 0,939 | 0,884 / 0,942 | 0,878 / 0,940 |

**Δ = trainiert − untrainiert [95-%-Bootstrap-KI]** (fett: die Regel „KI über 0 und AUC über
der Permutations-Null“ ist erfüllt):

| Sondenset | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| fanout k ≥ 5 | 0 (nicht auswertbar: 2 Paare je Hälfte) | 0 | 0 | 0 | 0 |
| fanout k ≥ 3 | +0,010 [−0,061; +0,135] | +0,044 [−0,221; +0,222] | +0,032 [−0,181; +0,215] | +0,032 [−0,207; +0,235] | −0,030 [−0,175; +0,213] |
| fanout k ≥ 2 | −0,136 [−0,180; −0,017] | −0,081 [−0,223; −0,058] | −0,179 [−0,246; −0,085] | −0,171 [−0,219; −0,090] | −0,150 [−0,200; −0,068] |
| fanout k ≥ 2, Hold-out | −0,097 [−0,171; +0,050] | −0,184 [−0,276; −0,027] | −0,165 [−0,255; −0,033] | −0,073 [−0,246; −0,026] | −0,061 [−0,222; +0,001] |
| dao, ohne Maske | +0,001 [−0,000; +0,003] | **+0,001 [+0,000; +0,003]** | **+0,003 [+0,002; +0,006]** | **+0,002 [+0,001; +0,004]** | **+0,004 [+0,002; +0,007]** |
| dao, DAO maskiert | **+0,002 [+0,001; +0,004]** | 0,000 [−0,005; +0,003] | −0,007 [−0,016; −0,001] | −0,022 [−0,040; −0,012] | −0,038 [−0,061; −0,020] |
| dao, alle maskiert | **+0,022 [+0,000; +0,028]** | −0,050 [−0,082; −0,021] | −0,071 [−0,090; −0,047] | −0,062 [−0,084; −0,043] | −0,072 [−0,115; −0,047] |
| dao, alle maskiert, Hold-out | **+0,022 [+0,004; +0,027]** | −0,024 [−0,053; +0,002] | −0,071 [−0,090; −0,048] | −0,058 [−0,080; −0,037] | −0,062 [−0,097; −0,039] |

Nullen: 95-%-Quantil der Label-Permutation 0,74–0,77 (k ≥ 3, 15 Paare; p bis 0,095),
0,53–0,54 (k ≥ 2), 0,55–0,57 (k ≥ 2, Hold-out), 0,54–0,55 (dao); außer bei k ≥ 3 liegt jede
trainierte AUC darüber (p ≤ 0,005). Zufallsrichtungen (gefaltet, 95-%-Quantil, trainiert):
0,84–0,88 (k ≥ 3), 0,67–0,69 (k ≥ 2), 0,71–0,73 (k ≥ 2, Hold-out), 0,91–0,99 / 0,85–0,93 /
0,77–0,83 (dao ohne / DAO / alle maskiert). Mit Bonferroni über die 5 Snapshots
(`counts_adj`) bleiben dao ohne Maske L1–L4, dao/DAO maskiert L0 und dao/alle maskiert
(Hold-out) L0; dao/alle maskiert L0 auf dem ganzen Korpus fällt weg. Das Refit-KI ist
konservativ (enthält die Unsicherheit der Richtungsschätzung); das KI nur über die
Auswertungshälfte ist enger, ändert aber an keiner Stelle eine Entscheidung. Rohdaten:
`runs/trace_e2e/r2/probe_*/criterion.csv`, `probe_report.json`.

*Alignment mit Kontrollen* (die Zelle, die das rohe Alignment wählt — bei 300 Paaren und
α = 1 L3 → L4, sonst L4 → L4; „residualisiert“ = Anteil der **längenfreien** Zielvarianz):

| Paare | Kovariaten | α | Zelle | roh trainiert | roh untrainiert | Gewinn roh [95-%-KI] | nur Länge | resid. trainiert | resid. untrainiert | Gewinn resid. [95-%-KI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | len3 | 1 | L3 → L4 | 0,666 | 0,627 | +0,039 [−0,025; +0,122] | 0,572 | 0,168 | 0,091 | +0,077 [−0,181; +0,297] |
| 300 | len3 | 100 | L4 → L4 | 0,590 | 0,238 | +0,352 [−0,000; +0,491] | 0,572 | 0,090 | 0,000 | +0,090 [+0,039; +0,140] |
| 300 | len3+text | 1 | L3 → L4 | 0,666 | 0,627 | +0,039 [−0,025; +0,122] | 0,610 | 0,238 | 0,138 | +0,099 [−0,027; +0,215] |
| 300 | len3+text | 100 | L4 → L4 | 0,590 | 0,238 | +0,352 [−0,000; +0,491] | 0,610 | 0,094 | 0,002 | +0,092 [+0,030; +0,153] |
| 2 000 | len3 | 1 | L4 → L4 | 0,785 | 0,756 | +0,029 [+0,005; +0,053] | 0,586 | 0,514 | 0,464 | +0,050 [+0,009; +0,091] |
| 2 000 | len3 | 100 | L4 → L4 | 0,668 | 0,437 | +0,231 [+0,182; +0,274] | 0,586 | 0,308 | 0,103 | +0,205 [+0,141; +0,260] |
| 2 000 | len3+text | 1 | L4 → L4 | 0,785 | 0,756 | +0,029 [+0,005; +0,053] | 0,672 | 0,473 | 0,416 | +0,058 [+0,030; +0,085] |
| 2 000 | len3+text | 100 | L4 → L4 | 0,668 | 0,437 | +0,231 [+0,182; +0,274] | 0,672 | 0,247 | 0,102 | +0,145 [+0,110; +0,179] |
| 2 000 Hold-out | len3 | 1 | L4 → L4 | 0,670 | 0,603 | +0,067 [+0,033; +0,105] | 0,464 | 0,421 | 0,348 | +0,074 [+0,017; +0,127] |
| 2 000 Hold-out | len3 | 100 | L4 → L4 | 0,528 | 0,299 | +0,228 [+0,175; +0,273] | 0,464 | 0,268 | 0,085 | +0,183 [+0,133; +0,237] |
| 2 000 Hold-out | len3+text | 1 | L4 → L4 | 0,670 | 0,603 | +0,067 [+0,033; +0,105] | 0,559 | 0,399 | 0,341 | +0,058 [+0,020; +0,096] |
| 2 000 Hold-out | len3+text | 100 | L4 → L4 | 0,528 | 0,299 | +0,228 [+0,175; +0,273] | 0,559 | 0,231 | 0,086 | +0,144 [+0,116; +0,176] |

Die 300-Paar-Zeilen reproduzieren §7 (0,666 / 0,627). Über alle 28 Zellen mit
Quell-Layer 1–4 liegt das KI des residualisierten Gewinns bei 300 Paaren in 3–6 (α = 1)
bzw. 25–27 (α = 100) Zellen über 0, bei 2 000 Paaren in 24–28 (α = 1) bzw. allen 28
(α = 100), bei 2 000 Hold-out-Paaren in 27–28 bzw. 28. Von den 400 Hold-out-Paaren der
2 000er-Stichprobe stammen 45 aus dem Hold-out des Spurenmodells, in der Hold-out-Stichprobe
alle 400. Rohdaten: `runs/trace_e2e/r2/align_{300,2000,2000_heldout}/align_controls.csv`,
`align_summary.json`.

**Ehrliche Lesart (Runde 2, nach Fix-Runde 1).**
- **Keine Teilen-Richtung zählt.** Die eigentliche Schwelle k ≥ 5 hat mit der korrigierten
  Definition nur 4 Positive und ist nicht auswertbar; k ≥ 3 hat nach Sichtbarkeit,
  Längen-Matching und Entfernen der Duplikate 15 Paare (KI-Breite ≈ 0,4). Bei k ≥ 2 (471
  balancierte Paare: Transaktionen, in denen ETH netto an mindestens zwei Adressen geht,
  meist ein Vertrag, der eine Einzahlung verteilt) trennt das **untrainierte** Modell in
  allen Snapshots besser als das trainierte (Δ −0,08 bis −0,18, KI unter 0), auch nur auf
  dem Hold-out des Spurenmodells (106 Paare, Δ −0,06 bis −0,18). Die
  Oberflächen-Baseline (0,84) liegt über den trainierten AUCs (0,72–0,85) — die Information
  steht in der Eingabe, das Training ordnet sie nicht entlang dieser Richtung.
- **DAO-Sonde:** Ohne Maske ist sie eine Identitätssonde (beide 0,995–0,999). Das trainierte
  Modell liegt dort in L1–L4 um 0,001–0,004 vorn, und das KI liegt formal über 0 (auch mit
  Bonferroni) — an der Decke einer Sonde, die die DAO-Adress-Tokens direkt sieht, ist das
  ein Effekt gelernter Adress-Embeddings, kein Merkmal „Teilen“. Das Maskieren der
  DAO-Adressen allein erzeugt die bekannte `[A_OOV]`-Abkürzung (Oberfläche 0,983); mit allen
  Adressen maskiert fällt die Oberfläche auf 0,67, beide Modelle trennen noch 0,87–0,94, das
  trainierte in L1–L4 wieder schlechter (Δ −0,05 bis −0,07). Formale Ausnahme bleibt
  Snapshot 0 (Mittel der Eingabe-Embeddings, ohne Kontext): Δ +0,022, auf dem Hold-out auch
  mit Bonferroni — eine Eigenschaft gelernter Token-Embeddings, keine kontextuelle
  Repräsentation.
- Dass das trainierte Modell in tieferen Layern *schlechter* trennt als zufällige Gewichte,
  heißt nicht, dass es die Information verliert: die Difference-of-Means-Richtung der
  gemittelten Zustände ist ein grober Leser, und das Sprachmodell-Training ordnet die
  Varianz nach Vorhersagbarkeit des nächsten Tokens, nicht nach diesen Labels. Für die
  Stufe-L-Frage ist aber genau diese Richtung `v_src`; sie ist in diesem Modell nicht besser
  als die einer Zufallsinitialisierung.
- **Alignment:** Die Längenmerkmale erklären 0,46–0,67 der Varianz der Pythia-L4-Zustände.
  Nach Residualisierung erklärt das trainierte Spurenmodell 0,47 (len3+text, α = 1, 2 000
  Paare) der längenfreien Restvarianz, das untrainierte 0,42; der Gewinn +0,06 [+0,03; +0,09]
  ist mit 2 000 Paaren in fast jeder Zelle signifikant, mit den 300 Paaren aus §7 bei α = 1
  nicht. Auf 2 000 Paaren nur aus dem Hold-out des Spurenmodells ist er gleich groß
  (+0,06 [+0,02; +0,10]) — kein Hinweis auf Auswendiglernen der Trainingsspuren. Bezogen
  auf die Gesamtvarianz sind das etwa 0,02 (0,33 × 0,058). Bei α = 100 ist der Gewinn größer
  (+0,14 bis +0,21), weil zufällige Features viele Dimensionen brauchen — derselbe
  Kompaktheitseffekt wie in §7. Das trainierte Modell trägt also eine kleine,
  längenunabhängige, linear mit Pythias Zuständen des Begleittexts geteilte Information über
  die Transaktion, die zufällige Features nicht haben. *Welche* (Call-Typen, als Hex
  wiedergegebene Selektoren, Wertgrößen) ist offen; ein Merkmal „Teilen“ ist es nach den
  Sonden oben nicht.
- Grenzen der Kontrollen: lineare Residualisierung (nichtlineare Längeneffekte bleiben,
  stehen aber beiden Quellen gleich zur Verfügung); das Alignment-KI erfasst nur die
  Hold-out-Stichprobe, nicht die Schätzunsicherheit der Abbildung; eine einzige
  Split-Saat; 0,16 Epochen eines kleinen Modells. Für den A100-Lauf heißt das: Sonden nur
  mit `--match length` und auf dem Hold-out (`--split heldout`), DAO nur mit
  `--mask all_addresses` (oder besser innerhalb der DAO-Transaktionen kontrastieren). Der
  Netto-fan-out ≥ 5 ist 2016 so selten, dass die eingefrorene Sonde auch auf dem ganzen Jahr
  (≈ 60-mal mehr Transaktionen, hochgerechnet ≈ 240 Positive vor Sichtbarkeit und Matching)
  knapp bleibt; k = 2 trifft jetzt echte Verteilung (ein Vertrag verteilt an mehrere
  Empfänger), nicht mehr Durchreichen. Eine Änderung von k wäre eine Änderung des
  eingefrorenen Merkmals (Präregistrierung 18) und vor dem A100-Lauf festzulegen. Das
  Kriterium Δ > 0 mit KI bleibt Bedingung, bevor eine Richtung in die Transplantation geht.

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

**Stufe L mit dem Spurenmodell.** `transplant_experiment` nimmt Prompt-*Paare*
(`lsne.transplant.PromptPair`: die Quelle kodiert eine Trägertransaktion, Pythia den
Stimulus), und Stufe L hat einen eigenen Pfad für `--transplant-source trace:<modell>`
(`lsne.experiments.stage_l_transplant`, Abschnitt „Trace source“; Helfer
`lsne/chain/pairs.py`): eingefrorene fan-out-Sonde mit untrainiertem Zwilling,
längenresidualisierte Alignments je Ziel-Layer, Transplantation A/B/C mit einer
K=0-Variante je Familie und ihrem K=8/C1-Partner auf derselben Trägertransaktion,
B/T-Effekte mit Familien-Bootstrap, `trace_verdict`. Die Optionen stehen auf der
Kommandozeile hinter dem Modellverzeichnis:

```bash
python -m lsne.cli run --stage L --model EleutherAI/pythia-160m --revision step98000 --out runs_trace \
    --transplant-source "trace:runs/trace_lm,shards=data/traces/itr,k=5,standin=runs/L/pythia-160m@step98000"
```

`standin` zeigt auf den Stand-in-Lauf (Kontrast 18, `verdict.csv`); ohne bestandene
Stand-in-Validierung setzt die Stufe `counts_for_prereg = False` (Präregistrierung 18).
Mit den korrigierten Labels (§12) hat die eingefrorene Sonde (Netto-fan-out ≥ 5) in der
CPU-Stichprobe nur 4 Positive; die Stufe fällt dann auf eine gelockerte Sonde zurück und
behauptet nichts (`feature_frozen = False`).

## 10. Grenzen

- **Maßstab:** `small`, 0,16 Epochen, 31 von 672 Chunks der DAO-Zeit; die beiden
  Drain-Chunks stellen 29 % der Tokens. Kein Befund über ein trainiertes BlockGPT.
- **Keine Baum-Positionsembeddings** (BlockGPT): die Position im Call-Baum steckt in
  `[D{Tiefe}]`-Tokens und der Breitensuche-Reihenfolge; Rotary nur über die Sequenz.
- **Keine ABI-Dekodierung:** Argumente sind die ersten 4 32-Byte-Wörter (Adresse, Zahl
  < 2^160 oder `[WORD]`); Selektoren bleiben unbenannte 4-Byte-IDs.
- **OOV-Adressen:** 11,9 % der Adressvorkommen im Hold-out (post_fork 16 %); Adressen sind
  Identitäten, das Modell lernt Verträge, nicht Rollen. Für 2016 vollständig: Budget
  ≥ 50 000–100 000.
- **Packing über Transaktionsgrenzen** ohne Attention-Maske (Standard-GPT-Packing).
- **Kappen auf 512 Tokens** pro Transaktion (Training und `TraceLMSource`): gerade die
  vielteiligen Transaktionen (Drain: 1 409 Tokens) werden abgeschnitten; die
  Mean-Pooling-Zustände sehen nur die ersten 512 (Breitensuche: von der Drain-Transaktion
  19 der 81 Calls, maximale Tiefe 5 statt 28, alle 32 Storage-Diffs, 8 der 31 Logs). Seit §12 verlangt die fan-out-Sonde ihr
  Label auch im sichtbaren Fenster, und Paare (ITR, Text) enthalten nur Transaktionen, die
  ins Fenster passen (der Text beschreibt sonst Knoten, die das Spurenmodell nie sieht).
- **Zuordnung von Logs und Storage-Diffs** ist rekonstruiert (Xatu speichert keinen
  Knotenzeiger): nach Adresse, Ausführungsreihenfolge und Gas machbar (§12), aber bei
  mehreren machbaren Rahmen derselben Adresse nicht eindeutig.
- **Keine Storage-Reads** (`storage_reads` nicht geladen).
- **Teilen-Labels:** Netto-fan-out ≥ 5 kommt in der Stichprobe nur 4-mal vor (nach der
  Zählung des ersten Laufs 106-mal, §12); die eingefrorene Sonde ist damit auf der
  CPU-Stichprobe nicht auswertbar. Die DAO-Sonde ist zum großen Teil eine
  Adress-Identitätssonde (DAO-Adressen stehen immer im Vokabular).
- **In-Sample-Elemente:** Sonden und Paare stammen standardmäßig aus dem ganzen Korpus,
  also zu ≈ 90 % aus dem Trainingsteil des Spurenmodells; `--split heldout` beschränkt sie
  auf den Hold-out (Kontrolle in „Kontrollierte Sonden“).
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
1. **Längen-/Strukturkontrolle in den Sondensets:** `--match length` (Notebook 12), die
   Oberflächen-Baseline (`surface_baseline_auc`) und die untrainierte Kontrolle bleiben
   Pflicht. Eine Teilen-Richtung zählt erst, wenn sie die untrainierte Kontrolle und die
   Oberflächen-Baseline schlägt.
2. **Hold-out:** Sonden und Paare mit `--split heldout` (bzw. `split=heldout` in Stufe L),
   damit trainiert − untrainiert kein In-Sample-Vergleich ist.
3. **DAO-Sonde ohne Adress-Abkürzung:** mit allen Adressen maskiert bzw. innerhalb der
   DAO-Transaktionen (z. B. Split vs. Transfer).
4. **Ausgewogene Paare** (`pair --balance`) und mehr davon (≥ 2 000), Ziel Pythia-160m
   (Stufe L), Alignment gegen die Längen-Kontrolle; nur Transaktionen, die ins Fenster
   passen (Voreinstellung von `pair`).
5. **Seltenheit des Merkmals:** Netto-fan-out ≥ 5 hatte in der CPU-Stichprobe 4 Positive
   (§12); auch über ganz 2016 sind nur einige hundert zu erwarten. Ob k bleibt, ist vor dem
   Lauf festzulegen (`frozen_feature.json`, Präregistrierung 18).
6. **Stufe L** mit `standin=<Stand-in-Lauf>` (Präregistrierung 18) und einer K=0-Variante
   je Familie (Voreinstellung), damit der Familien-Bootstrap ≥ 3 Cluster hat.

## 12. Fix-Runde 1: Datensemantik korrigiert, alles neu gemessen

Stand 26.10.2026 (nach der adversariellen Prüfung des ersten Laufs). Die Prüfung fand
Fehler in der Datensemantik, die Labels, Texte und Tokens betreffen; alle Zahlen in §4–§8
und in „Kontrollierte Sonden“ stammen deshalb aus **neu serialisierten Shards** (`serialize
--overwrite`, 37 s mit 3 Prozessen, gleiche 225 854 Transaktionen und 7 051 538 Tokens —
die Korrekturen verschieben Tokens, ändern aber keine Längen), einem **neu trainierten**
`small`-Modell (gleiche Einstellungen, gleicher Seed) und neu gerechneten Sonden und
Alignments. Die Shards und Modelle des ersten Laufs liegen unter `runs/trace_e2e/*_round0`.

| Befund | Korrektur | gemessene Wirkung (31 Chunks) |
|---|---|---|
| Logs und Storage-Diffs hingen am **ersten** Knoten mit der Zieladresse, oft einem Aufruf, der sie nicht bezahlen konnte (WithdrawDAO: die DAO-Schreibvorgänge an `balanceOf`, 788 Gas) | Zuordnung nach Machbarkeit (`itr.attach_effects`): Kandidaten sind die Rahmen, deren Code auf den Speicher der Adresse wirkt (`call`/`create` an die Adresse, `delegate_call`/`call_code` *von* ihr), nie Rahmen in gescheiterten Teilbäumen; das eigene Gas (`gas_used` minus Kinder) muss die Mindestkosten tragen (SSTORE ≥ 5 000, LOG ≥ 375 + 375 je Topic + 8 je Datenbyte); Logs monoton in `log_index` entlang der Ausführungsreihenfolge (Euler-Tour) | alte Regel: 22 566 von 97 803 Schreibvorgängen (23,1 %) und 25 280 von 90 125 Logs (28,0 %) an Rahmen, deren eigenes Gas nicht einmal einen Eintrag bezahlt; neue Regel: 0, alle 187 928 Einträge machbar und monoton platziert (kein Rückfall); 26 543 Schreibvorgänge (27 %) und 53 228 Logs (59 %, vor allem die rekursiven Drain-Splits) sitzen an einem anderen Knoten als vorher |
| `native_transfers` enthält eine Zeile je Trace-Knoten, auch zurückgerollte Versuche und den Kontextwert von `call_code`/`delegate_call` (bewegt kein ETH) | Flags `reverted` / `phantom` je Transferzeile (exakter `internal_index`-Zeiger); `itr.effective_transfers` ist die eine Definition für Labels **und** Text | 2 595 zurückgerollte Wertzeilen in 2 566 Transaktionen, 467 Phantomzeilen; `total_eth` > 10⁹ ETH: 66 → 0 (die 2²⁵⁶-Wei-Zeile in 0x5ed540cf… zählt nicht mehr) |
| fan-out/fan-in zählten Durchreich-Verträge und Rückzahlungen als Empfänger und Sender | **Netto-Flüsse** (`itr.net_flows`): Adressen mit Netto-Änderung 0 fallen weg; fan-out = Adressen mit Netto-Zufluss, fan-in = mit Netto-Abfluss | `many_to_many` 12 874 → 618 (10 669 → `single`, 1 364 → `other`, 223 → `none`); 699 `single` → `none` (gescheiterte Transaktionen); fan-out ≥ 5: 106 → **4** |
| 512-Token-Fenster: die Wert-Transfers langer Transaktionen liegen hinter dem Fenster | Merkmale zusätzlich auf dem sichtbaren Fenster (`fan_out_eth_visible`, …); die fan-out-Sonde verlangt das Label auch dort; Match-Tabelle mit sichtbaren Empfängern; Paare (ITR, Text) nur für Transaktionen ≤ 512 Tokens (`pair --max-len`) | k ≥ 3: 73 Positive, davon 49 im Fenster sichtbar; k ≥ 2: 1 992 / 1 985 |
| Text zählte `call_code`-Wertzeilen, nannte Adressen nach dem Verhalten in *dieser* Transaktion „account“ | gleiche wirksame Transfers wie die Labels, Netto-Satz („Net, … ETH goes from … sources to … recipients“, wenn er vom Brutto abweicht); „contract“/„account“ aus Code-Evidenz im ganzen Chunk (`itr.address_kinds`), sonst „address“ | 15 228 Texte geändert; `many_to_many` mit Text „from 1 sender to 1 receiver“/„No ETH moves“: 1 477 → 36 (alle 36 aus ERC-20-Flüssen) |
| DAO-Menge: 4 Adressen statt der EIP-779-Liste; Kindes-DAOs als Negative; Ära `dao_active` begann 757 Blöcke vor der DAO | EIP-779-Liste `L` (116 Konten, identisch mit go-ethereums `DAODrainList`) + Creator + WithdrawDAO; reine Argument-Erwähnungen (`dao_mentioned`) aus den Negativen der DAO-Sonde ausgeschlossen; `dao_active` ab Block 1 428 757 (Erzeugung der DAO) | DAO-beteiligt 11 207 → 11 221; 46 Transaktionen erwähnen eine DAO-Adresse nur als Argument (39 davon in drain_crisis/post_fork, jetzt ausgeschlossen); 2 949 Transaktionen von `dao_active` nach `pre_dao` |
| Argumente zwischen 2⁶⁴ und 2¹²⁸ → `[WORD]`; `CallTooDeep` → `[ERR:other]`; rekursives `to_dict` scheitert an Tiefe 1 025 | Zahlen-Token für alle Werte < 2¹⁶⁰, Adresse nur bei ≥ 2¹²⁸ oder bekannter Adresse der Transaktion; `CallTooDeep` → `[ERR:call_depth]`; iteratives `to_dict`/`from_dict` | 7 370 Argumente in 6 185 Transaktionen werden von `[WORD]` zu Zahl oder Adresse; `CallTooDeep` kommt nur im DoS-Chunk 2 378 000 (Profil) vor. Mit der Umordnung der Logs/Diffs hat sich die Tokenfolge von 12 101 Transaktionen (5,4 %) geändert, 5 916 davon nur in der Reihenfolge |
| Sonden- und Paar-Elemente stammen zu ≈ 90 % aus dem Trainingsteil des Spurenmodells; identische Sequenzen in beiden Split-Hälften; `git_sha` ohne den tatsächlichen Code | `probe/pair/align --split heldout` (Hash-Split des Trainings) und die Zugehörigkeit jedes Elements im Bericht (`lm_split`); Paare mit wiederholter sichtbarer Sequenz werden vor Split und Bootstrap entfernt (`n_pairs_dropped_duplicates`); jeder Bericht/Manifest trägt `code_sha256` (Inhalt von `lsne/chain` und den importierten Modulen) und die Richtungsdatei `model_fingerprint` | siehe „Kontrollierte Sonden“ |

Für Stufe L (Notebook 12) kamen hinzu: Transplantations-Prompts über alle Familien (eine
K=0-Variante je Familie statt der ersten drei Varianten einer Familie, sonst kann der
Familien-Bootstrap nie ≥ 3 Cluster erreichen), eine gespeicherte Richtung zählt nur, wenn
sie von genau der eingefrorenen Sonde stammt (`direction_spec_ok`), KV-Pflanzung verlangt
gute Abbildungen auf **allen** gepflanzten Layern, die Zufallskontrolle benutzt je
Trägertransaktion eine Richtung, `counts_for_prereg` verlangt eine bestandene
Stand-in-Validierung (`verdict.csv` des Notebook-11-Laufs), und die Spuren-Optionen sind
über `--transplant-source "trace:<dir>,shards=…,k=…,standin=…"` auf der Kommandozeile
erreichbar (`lsne.cli run` hat keine `--trace-*`-Flags). Prüflauf ohne `--smoke` (dieses
Modell, `pythia-14m@step98000`, Familien DEV01–DEV04,
`trace:runs/trace_e2e/trace_lm_small,limit=40000,n_probe=200,n_pairs=200,k=2`, 3,5 min):
8 Prompts aus 4 Familien, Familien-Bootstrap mit 4 Clustern (`enough_clusters`), für `kv`
und `both` verfehlen 4 der 7 gepflanzten Abbildungen das Alignment-Kriterium (Ziel-Layer 0,
1, 2, 6), kein Anspruch, und `claim` nennt die fehlende Stand-in-Validierung.
