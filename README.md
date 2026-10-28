# LS-NEQ — Sprachwechsel → Nicht-Gleichgewicht auf State 2143

Kontrollierte Sprachwechsel-Experimente (Englisch ↔ Deutsch) an **State 2143**
(`EleutherAI/pythia-160m`, 208-Token-Kontext, Ziel „bey“) mit den Diagnostiken der
Nicht-Gleichgewichts-Gitterphysik. Python-Paket `lsne`, Colab-Notebooks `notebooks/`,
Versuchsdesign `docs/DESIGN.md`, Präregistrierung `docs/PREREGISTRATION.md`,
Modulverträge `docs/SPEC.md`.

Ergebnisse und Prüfung des ersten Colab-Laufs (pythia-160m@step98000, vorläufig): `docs/RESULTS_RUN1.md`;
präregistrierte Auswertung nach den Neuläufen (Notebook 13): `docs/RESULTS_PREREG.md`.

Das Dossier „State 2143“ zeigt einen reproduzierbaren lokalen Eingriffseffekt (ein
Schritt der Dosis ε entlang des Entropiegradienten am Eingang von Block 11), aber keinen
Sprachwechselmechanismus: der Originalkontext enthält null Sprachwechsel. Dieses
Repository macht Sprachwechsel zu einer **kontrollierten, skalierbaren Eigenschaft des
Kontexts** — diskret über die Wechselzahl K und die Strukturstufe C, kontinuierlich über
eine Dosis ε entlang einer Sprachachse oder eines SAE-Features — und prüft drei Fragen.

## Die drei Fragen

1. **Counterfactual-Frage (Im et al. 2026).** Welche Positionen vermitteln die Wirkung
   eines Sprachwechsels auf die Zielvorhersage? Tragen Leerzeichen-, Satzzeichen- und
   Grenz-Tokens einen überproportionalen Anteil, wie es Im et al. für Prompt-
   Interventionen fanden (Patching der Delimiter allein stellt 11–16 % des Effekts
   wieder her)? → Stufe C, `lsne.patching`.
2. **Achsen-Frage (Tagliabue et al. 2026, „Pain Axis“).** Gibt es eine lineare
   Sprachachse (Difference-of-Means D − E), wie steht sie geometrisch zur
   State-2143-Entropierichtung, zu einer Häufigkeitsachse und zu Zufallsrichtungen,
   und welche Tokens befördert sie durch die Unembedding-Matrix? → Stufe B,
   `lsne.directions`.
3. **Nicht-Gleichgewichts-Frage (Gitterphysik).** Erreicht die Dynamik auf dem
   (Layer × Position)-Gitter beim Hochskalieren von Sprachwechseln Zustände, die sich mit
   Standard-Diagnostiken von einem relaxierenden Regime unterscheiden: wachsende
   Störungsausbreitung, Hysterese, langsame Relaxation, nicht-monotone Konvergenz der
   Zwischenschicht-Vorhersagen, Bruch der linearen Antwort? → Stufe D/E,
   `lsne.dynamics`, `lsne.generation`, `lsne.tn_lattice`.

Primärer, präregistrierter Endpunkt: der Wachstumsexponent **λ** der
Störungsausbreitung entlang der Tiefe als Funktion von (K, ε, Richtung) auf dem
gemeinsamen englischen Schlussabschnitt, mit isotropen Zufallsrichtungen gleicher
RMS-Dosis als Nullmodell (`docs/PREREGISTRATION.md` §2).

**Erweiterungsstufen G–L** (`docs/DESIGN.md` §4.4–4.9, Tabellen und Endpunkt-Karte in
`docs/STAGES.md`): Sprachwechsel bleibt das Mittel, mit dem lokale Verdichtung der
semantischen Geometrie kontrolliert gestellt wird (G, *Meaning Contracts*), die
Eingriffe werden zu λ-Pfaden zwischen Trennschärfe und Mimikry verallgemeinert (H,
Biggio et al.), die Steuerrichtung wird am Domänen-Kompass und am schlechtesten Head
gewählt (I), Differenzen statt Einzeldomänen werden erklärt und der Gegenpol
(Proteinfehlfaltung) empirisch per Pile-Schwierigkeitsrangfolge geprüft (J), ein
DAO-Fork-Korpus wird per Test-Time-Training gegen drei Kontrollarme injiziert (K) und
die Transplantation eines Quellmodell-Merkmals in den Residualstrom bzw. KV-Cache wird
mit einem Stand-in validiert (L). Die Stufen laufen mit derselben Stufenschnittstelle
(`run()`), schreiben ihre Tabellen in dasselbe Layout und fangen Fehler pro Substufe ab.

## Der Anker: State 2143

| Merkmal | Wert |
|---|---|
| Modell | `EleutherAI/pythia-160m` (GPT-NeoX, 12 Blöcke, d = 768, 12 Heads, parallele Residuen) |
| Datensatz | `EleutherAI/pile-standard-pythia-preshuffled`, Sequenzlänge 2049, Batch 1024 |
| Globaler Sequenzindex | 5 025 183 = 4907 · 1024 + 415 |
| Kontext / Ziel | erste 208 Tokens / Token 22905 „bey“ |
| SHA256 (2049 Tokens, uint16 LE) | `bdd4bbdd2b8138437e052b2f0e3aeb212a563c25fe447c920b0e9c6ee76f55d4` |
| SHA256 (208 Kontext-Tokens) | `84c8f1b6a7a933f53f3b21e353c0c3af219d3ccfbcb7092584a9c6ad1035a750` (Dossier-Präfix `84c8f1b6a7a933f5`) |
| Eingriffsstelle | Eingang Block 11 (nullbasiert), Position 207; δ = ε · RMS(h) · g/RMS(g), ε = 0,01 |
| Checkpoints | step98000, step99000 (Eingriff); step143000 (SAE); 27 Dossier-Checkpoints (`lsne.config.CHECKPOINT_TRACE`, Verlauf) |

**Wie der Anker geholt wird.** Die Sequenz wird *nicht* aus lokalen Dateien gelesen und
*nicht* ins Repository geschrieben, sondern zur Laufzeit per HTTP-Range-Request direkt
aus dem Datensatz geladen (`lsne.pile_fetch.fetch_sequence`): 4098 Bytes ab Byte-Offset
5 025 183 · 4098 im Shard `document-00000-of-00020.bin` (die `.bin`-Dateien sind rohe
little-endian-uint16-Tokenströme ohne Header). Danach prüft `lsne.anchor.verify_state2143`
beide Hashes und die Ziel-ID; bei Abweichung bricht der Lauf ab (`strict=True`). Der
Download wird unter `LSNE_CACHE/state2143` (Standard `./cache/state2143`) gecacht. Alle
Konstanten stehen in `lsne.config.STATE2143`.

Hook-Konventionen (verifiziert mit transformers 5.20): `hidden_states[ℓ]` ist der
*Eingang* von Block ℓ, `hidden_states[12]` die Ausgabe der finalen LayerNorm; Block-Ausgänge
werden per Forward-Hook abgegriffen. Der SAE-Hookpoint `layers.10` der EleutherAI-SAEs ist
identisch mit „Eingang Block 11“ (`HookSite(11, "block_input").sae_hookpoint == "layers.10"`).

## Stimulus-Design: Sprachwechsel als kontrollierte Variable

Jede Familie (`data/stimuli/families/F01A.json` …) besteht aus **10 Bedeutungseinheiten**
mit je drei Segmenten `subj | pred | adj`, jeweils in Englisch (E) und Deutsch (D):

```
"Lena" | "plans a new exhibition"      | "at the museum"
"Lena" | "plant eine neue Ausstellung" | "im Museum"
```

Daraus entstehen die drei **Strukturstufen C** eines Wechsels an der Grenze Einheit k → k+1,
ohne separate Texte zu schreiben:

| Stufe | Realisierung |
|---|---|
| C1 Satzgrenze | Einheit k vollständig in m_k, Einheit k+1 vollständig in m_{k+1} |
| C2 Teilstruktur | `subj_k pred_k` in m_k, `adj_k` in m_{k+1} (Wechsel vor dem Adjunkt) |
| C3 Abhängigkeit | `subj_k` in m_k, `pred_k adj_k` in m_{k+1} (Subjekt-Verb-Kongruenz über die Sprachgrenze) |

Die **Wechselzahl K** wird durch die Sprachmasken des Dossiers gesetzt (Einheit 1 und 10
immer E, Einheit 9 in gemischten Bedingungen immer D, letzter Wechsel stets vor Einheit 10):

| K | Maske (Einheiten 1–10) |
|---|---|
| 0 | `E E E E E E E E E E` |
| 2 | `E E E E D D D D D E` |
| 4 | `E E D D E E D D D E` |
| 8 | `E D E D E D E D D E` |
| 16, 28 | Masken auf Segmentebene (Wechsel an fast jeder Segmentgrenze), geführt als `C = "seg"` |

Jede Familie hat einen festen englischen Schlussabschnitt `tail_E` und ein Zieltoken
`target` (ein einzelnes Pythia-Token, nicht im Kontext enthalten). Gemessen wird
log p(target | Kontext). Pro Variante werden geprüft: identische Ziel-ID, identische
Token-IDs des Schlussabschnitts (BPE-Grenzen), kein Ziel-Leak. Umfang: 12 Testfamilien
F01–F12 × Formulierungen A/B + 4 Entwicklungsfamilien DEV01–DEV04 (nur für Dosis- und
Schwellenwahl); pro Formulierung 1 + 3·3 + 2 = 12 Varianten; zusätzlich die
**Original-Fallstudie** (der 208-Token-Kontext in 7 Einheiten, Einheiten 1–6 mit
deutschen Übersetzungen, Ziel bleibt „bey“). Jedes Token trägt eine Annotation (Sprache,
Einheit, Tokenklasse `ws`/`punct`/`word_start`/`word_cont`/`boundary`, Abstand zum
letzten Wechsel, F100 = 100·K/(N−1)); deutsche Segmente erzeugen mehr Subwort-Tokens
pro Wort, die Längen werden daher immer mitberichtet (Konfundierung, nicht wegnormiert).

## Richtungen (Eingriffsvektoren)

| Kürzel | Konstruktion | Rolle |
|---|---|---|
| `ent` | ∂H/∂h am Eingang des letzten Blocks, letzte Kontextposition (Dossier-Konstruktion); optional orthogonal zu einer ARI-Achse aus `axes.npz` | Replikation / Transferarm |
| `lang_dom` | Difference-of-Means D − E über die letzten Tokens gepaarter Segmente, oberste gemeinsame Hauptkomponente herausprojiziert („denoised“) | Sprachachse |
| `lang_probe` | logistische Regression E/D auf denselben Aktivierungen | Kontrolle für `lang_dom` |
| `sae_D*`, `sae_E*` | Decoder-Zeilen sprachselektiver TopK-SAE-Features (`EleutherAI/sae-pythia-160m-32k`, Hookpoint `layers.10`); FVU wird gemessen, > 0,4 = „nicht zertifiziert“ | Feature-Steering |
| `freq` | mittlere Unembedding-Zeile der 100 häufigsten Pile-Tokens minus Gesamtmittel (Unigramm-Zählung aus per Range-Request gezogenen Sequenzen) | Häufigkeitsachse |
| `rand_00…31` | 32 isotrope Zufallsrichtungen, jeweils ± | Nullmodell |

Alle Richtungen sind Einheitsvektoren; die Dosis ist δ = ε · RMS(h) · dir/RMS(dir)
(`lsne.model.make_dose_delta`), die Soll/Ist-Abweichung des geschriebenen Vektors wird
protokolliert (`Injector.audit`).

## Nicht-Gleichgewichts-Diagnostiken (Stufe D, kurz)

Gitter: Sites (ℓ, t) mit ℓ = 0…L (Block-Eingänge + Ausgang), t = Position; „Zeit“ =
Tiefe ℓ, zweite Zeit = autoregressiver Schritt, langsame Achse = Trainings-Checkpoint.

| Diagnostik | Messgröße | Nicht-Gleichgewichts-Signatur |
|---|---|---|
| D1 Störungsausbreitung | D(ℓ,t) = ‖h′−h‖/‖h‖ nach Dosis an (ℓ₀,t₀); λ = Steigung von ln D über ℓ; Ausbreitungsfront | λ > 0, wachsend mit K bzw. ε |
| D2 Tiefen-Monotonie | KL(p_final ‖ p_ℓ) der Logit-Lens; Nichtmonotonie-Index | Überschwingen, späte Umkehr |
| D3 Hysterese | f_D(ε) bei Dosisrampe 0 → ε_max → 0 in der Generierung; Fläche vs. Neustart-Null | A_hyst > Null-Verteilung |
| D4 Relaxation | Dosissprung bei Token t_on; τ aus Exponentialfit von f_D(t) | τ wächst nahe einer Schwelle |
| D5 Wechselstatistik | Abstände zwischen Sprachwechseln im generierten Text | schwere Ränder statt geometrisch |
| D6 Suszeptibilität | χ(ε) = d f_D/dε, Varianz über Samples | Peak / Sprung |
| D7 Fluktuation–Antwort | T_eff = Var₀(m)/χ für m = Projektion auf `lang_dom` (explorativ) | bedingungsabhängig |
| D8 Lineare Antwort | Abweichung der gemessenen Störung vom linearen Modell (JVP bzw. Tensornetz, Stufe E) | groß schon bei kleinem ε |
| D9 CA-Komplexität | Symbolfeld S(ℓ,t) ∈ {N,E,D} aus der Logit-Lens-Sprachmasse; Entropien, Aktivität, LZ76 | steigend mit K, ε |

Phasendiagramme jeder Kenngröße über (ε, K) und (ε, C), getrennt für `ent`, `lang_dom`,
`sae_*`, `rand`. Alle Signaturen sind operationalisierte, vorab definierte Kenngrößen mit
Nullmodellen — Analogien aus der Gitterphysik, keine physikalische Identität.

## Die Rolle von TNCO

TNCO (Google Research; Abanin et al. 2025) ist ein **Optimierer für
Kontraktionsreihenfolgen von Tensornetzwerken**, kein Interpretierbarkeitsverfahren.
Softmax, LayerNorm und GELU lassen sich nicht wegkontrahieren; was sich als Tensornetz
schreiben lässt, ist der am Arbeitspunkt **linearisierte** Transformer: bei eingefrorenen
Attention-Mustern und eingefrorener LN-Skalierung ist die Jacobi-Kette der Blöcke eine
Transfer-Matrix-Kette auf dem (Layer × Position)-Gitter, in faktorisierter Form (pro Block:
Skip ⊕ Σ_Heads A^h ⊗ M^h ⊕ positionsweise MLP-Blöcke mit Branch-Index) ein echtes
Tensornetz mit nicht-trivialer Kontraktionsreihenfolge. Genau diese Reihenfolge
optimiert TNCO (`lsne.tn_lattice.optimize_path`, Fallback `opt_einsum`); das Ergebnis
ist die lineare Antwortfunktion des Netzes, die in Stufe E mit der exakten linearen
Antwort (Autograd-JVP) und der gemessenen nichtlinearen Störung verglichen wird.
Details, Herleitung und Grenzen: `docs/TNCO_INTEGRATION.md`.

## Ausführen

### Colab-Notebooks (in dieser Reihenfolge)

| Notebook | Stufe | Inhalt |
|---|---|---|
| `notebooks/00_setup_and_anchor.ipynb` | A | Setup, Drive, Anker-Verifikation, Stimulusbank, Tokenklassen |
| `notebooks/01_baseline_and_directions.ipynb` | B | Baselines, Richtungsbank, Kosinus-Heatmap, Unembedding, P_freq vs. Entropie |
| `notebooks/02_switching_grid.ipynb` | C | T(K,C) mit Familien-Bootstrap, B(x)-Dosiskurven, Counterfactual-Attribution |
| `notebooks/03_nonequilibrium.ipynb` | D | Damage-Maps, λ(ε,K)-Phasendiagramm, Lens, Hysterese, Relaxation, CA |
| `notebooks/04_tn_linear_response.ipynb` | E | gemessen vs. linear vs. faktorisiert, TNCO vs. opt_einsum |
| `notebooks/05_scale_and_checkpoints.ipynb` | F | Shard-Auswahl, Checkpoint-Trace / Skalierungsarm, Resume, Merge |
| `notebooks/06_geometry_trajectory.ipynb` | G | Realisierungswolken, G0-Gate, T1 Tiefen-Trajektorie, G1 Kontraktion (n-angepasst), T2 Position, T3 Steering-Validität, T5 Kopplung mit Stufe D |
| `notebooks/07_lambda_dynamics.ipynb` | H | λ-Pfade (g, Dichte, F), Endpunkt-Größen vs. λ, Spearman mit Bootstrap, SAE-Feature-Kandidaten, B(x)(λ) für State 2143 |
| `notebooks/08_compass_heads.ipynb` | I | Domänenprofile (Atlas-Proxy/-Import), Domänenachsen, Gegensteuern × Familiendomäne, Head-Attribution, schlechtester Head, Head-Arm |
| `notebooks/09_rdx_sharing.ipynb` | J | Pile-Schwierigkeitsrangfolge (Gegenpol), RDX-z-Karten pro Kontrast × Basis × Layer, Stabilität, Sharing-Index und Netzwerkaktivität |
| `notebooks/10_ttt_injection.ipynb` | K | Korpora (Platzhalter markiert), Memorisationsgeschwindigkeit, Trainingsverlauf, Vorher/Nachher-Effekte gegen drei Kontrollarme |
| `notebooks/11_transplant.ipynb` | L | Funktionaler Transfertest pro Layer, Layer-Matching, Residual-Substitution / KV-Pflanzung / beides gegen Kontrollen |
| `notebooks/12_trace_source.ipynb` | L (A100) | Spurenmodell 2016 als Quelle für Stufe L: Xatu-Download, Serialisierung, Vokabular mit Hold-out-OOV, Training `blockgpt` (bf16), Evaluation, Sonden, Alignment, Stufe L mit `trace:<modell>` (`docs/RUNBOOK_COLAB.md` §10) |
| `notebooks/14_token_control.ipynb` | D, Kontrolllauf (A100) | Kontrolle der Sprache an der Injektionsstelle (Nachtrag 6.11 der Präregistrierung): reduzierter D1-Lauf mit Bank, Richtungen und Dosen des Hauptlaufs von 03, eigene Wurzel `runs_ctl`; nach 03 und vor 13, auf der Plattform des Hauptlaufs (`docs/RUNBOOK_COLAB.md` §12) |
| `notebooks/13_prereg_evaluation.ipynb` | C, D, G–L (CPU) | Präregistrierte Auswertung der vorhandenen Läufe beider Konten (`lsne.prereg_eval`): Primärendpunkt mit der Kontrolle aus 14, Sekundärendpunkte mit BH, G0-Gate, übernommene 11–18; läuft zuletzt (`docs/RUNBOOK_COLAB.md` §11) |

Jedes Notebook klont den Branch `claude/language-switching-non-equilibrium-ejpz0q` dieses
Repositories nach `/content/lsne-repo`, installiert `requirements.txt` + `pip install -e .` und
bindet Google Drive ein (`OUT_ROOT = /content/drive/MyDrive/LSNE`). Die Notebooks 00–11 rufen
die Stufenfunktion `lsne.experiments.<stage>.run()` direkt auf und zeichnen die Kerntabellen;
12 und 14 starten die Kommandozeile in eigenen Prozessen (Logs auf Drive), 13 wertet nur aus.
Mit `SMOKE = True` läuft jedes Notebook außer 13 (es hat keinen Rauchtest und braucht ohnehin
nur die CPU) in wenigen Minuten auf CPU mit pythia-14m; Notebook 12 nimmt dazu den
Xatu-Testausschnitt, Notebook 14 baut seinen eigenen Rauch-Hauptlauf von Stufe D unter
`OUT_ROOT/smoke_14` (Notebook 03 ist dafür nicht nötig). Die Notebooks 06–11 haben
zusätzlich eine Parameterzelle für die vom Auftraggeber gelieferten Eingaben
(`ATLAS_DIR`, `TTT_INPUTS`/`TTT_CORPUS`/`TTT_CHECKPOINTS`, `TRANSPLANT_SOURCE`), einen
Substufen-Bericht aus dem Manifest und im Kopf die präregistrierten Kontraste mit den
Tabellen, die sie tragen. Die Notebooks werden von `scripts/build_notebooks.py` erzeugt
(`tests/test_notebooks.py` prüft, dass sie nicht abdriften, führt 00, 06, 12 und 14 im
Rauchtest kopflos aus und 13 auf einer kleinen synthetischen Laufwurzel sowie auf den
Rauchläufen von 14). Schritt-für-Schritt-Anleitung für zwei Colab-Konten:
`docs/RUNBOOK_COLAB.md`.

### Kommandozeile

```bash
pip install -r requirements.txt && pip install -e .
# eine Stufe (A|B|…|L) oder mehrere; Ausgabe unter --out/<stage>/<modell>@<revision>/
python -m lsne.cli run --stage A,B,C,D,E --model EleutherAI/pythia-160m --revision step98000 \
    --dtype float32 --device auto --out runs --families F01,F02
# Erweiterungsstufen G-L auf demselben Lauf (Stufe K mit dem DAO-Korpus, Stufe L mit einem HF-Quellmodell)
python -m lsne.cli run --stage G,H,I,J,K,L --model EleutherAI/pythia-160m --revision step98000 --out runs \
    --ttt-corpus data/ttt/dao_fork --transplant-source EleutherAI/pythia-70m@step143000 [--atlas-dir data/atlas]
# Checkpoint-Trace (27 Dossier-Checkpoints) mit Stufen B und D, auf 2 GPUs aufgeteilt
python -m lsne.cli run --stage F --pairs trace --f-stages B,D --shard-index 0 --shard-count 2 --out runs
# Memorisationsgeschwindigkeit über drei Checkpoints (Shard 7)
python -m lsne.cli run --stage K --model EleutherAI/pythia-160m --revision step98000 --out runs_ttt \
    --ttt-corpus data/ttt/dao_fork --ttt-checkpoints step4000,step98000,step143000
# Rauchtest auf CPU, alle Stufen A-L (~1-2 min unbelastet)
python -m lsne.cli run --stage all --model EleutherAI/pythia-14m --revision step98000 \
    --device cpu --families DEV01 --out runs/smoke --smoke
# Kontrolllauf von Stufe D nach Nachtrag 6.11 (Notebook 14): eigene Wurzel, der Hauptlauf wird nur gelesen;
# gültig nur auf der Plattform des Hauptlaufs (A100, dieselbe Torch- und transformers-Version)
python -m lsne.cli run --stage D --d-control token_language --d-main-run runs/D/pythia-160m@step98000 \
    --out runs_ctl --model EleutherAI/pythia-160m --revision step98000 --dtype float32
# Tabellen aller Läufe zusammenführen
python -m lsne.cli merge --root runs --out runs/merged
```

Weitere Optionen: `--use-sae`, `--axis-npz`, `--k-levels`, `--c-levels`, `--k-extreme`,
`--n-random-dirs`, `--gen-n-new`, `--gen-n-samples`, `--tn-rank`, `--tn-window`,
`--limit-variants`, `--limit-families`, `--seed`; für G–L `--ttt-corpus`,
`--ttt-checkpoints`, `--transplant-source`, `--atlas-dir`; für den Kontrolllauf von Stufe D
`--d-control`, `--d-main-run`, `--d-allow-platform-mismatch` (`docs/STAGES.md`;
`python -m lsne.cli run --help`).
Umgebungsvariablen: `LSNE_CACHE` (Anker-/Unigramm-/Pile-NLL-Cache), `LSNE_RUNS`
(Standard-`--out` außerhalb von Colab), `LSNE_OFFLINE=1` (keine Netzzugriffe; `freq` aus
der Stimulusbank, Stufe J ohne Schwierigkeitsrangfolge), `HF_TOKEN` (nur bei Rate-Limits).
`--stage all` läuft A B C D E F G H I J K L in dieser Reihenfolge auf einem geladenen
Modell; die Stufen G–L fangen Fehler pro Substufe ab (`manifest.json → substages`,
`substage_errors.txt`), eine fehlgeschlagene Substufe lässt ihre Tabelle leer und die
Stufe läuft weiter (`docs/STAGES.md`).

### Eingaben, die der Auftraggeber liefert

Alle optional; ohne sie laufen die Stufen mit Platzhaltern bzw. Proxys und vermerken das
im Manifest. Nichts davon wird eingecheckt (`.gitignore`), nur die Ablagenotizen.

| Ablage | Stufe | Inhalt | Ohne die Datei |
|---|---|---|---|
| `data/ttt/dao_fork/` (+ `neutral/`, `nofork/`) | K | DAO-Fork-Korpus 2016 und zwei Kontrollkorpora (`*.txt` / `*.jsonl`, `data/ttt/README.md`); `--ttt-corpus`, im Notebook `TTT_INPUTS` (Drive-Ordner wird in den Checkout kopiert) | gemeinfreier Platzhalter, jede Zeile `is_placeholder = True` (Funktionstest, kein Befund) |
| `data/atlas/*.txt` | I | gespeicherte Extraktion des Pythia-Suszeptibilitätsatlas (Cluster-Schlüsselwörter); `--atlas-dir` / `ATLAS_DIR` | nur der lokale Logit-Lens-Proxy, `atlas_overlap` leer |
| `data/s4/axes.npz` | B | ARI-Achsen aus S4, gegen die `ent` orthogonalisiert wird; `--axis-npz` / `AXIS_NPZ` | `ent` ohne Orthogonalisierung |
| Quellmodell | L | BlockGPT-artiges Spurenmodell über die `SourceModel`-Schnittstelle (`docs/BLOCKGPT_FEASIBILITY.md`, `docs/TRACE_SOURCE_SPEC.md`, Paket `lsne.chain`); bis dahin `--transplant-source EleutherAI/pythia-70m@step143000` | Stand-in `rotated_self` (Validierung der Maschinerie, kein Transferbefund) |

### Tests

```bash
python -m pytest tests -q -p no:warnings              # CPU, pythia-14m step98000 (gecacht)
python -m pytest tests -q -m "not slow and not network"
python -m pytest tests/test_notebooks.py -q -m "not slow"   # Notebooks regenerieren + Drift-Check; 13 und 14 auf synthetischen Läufen
python -m pytest tests/test_notebooks.py -q -m slow         # Notebooks 00, 06, 12 und 14 (danach 13 auf dessen Rauchläufen) kopflos im Rauchtest (nbclient, ipykernel)
```

## Rechenplan: 7 A100-Shards (`jobs/shards.yaml`)

| Shard | Konto | Modell @ Revision | Stufen | Zweck |
|---|---|---|---|---|
| 1 | A | pythia-160m @ step98000 | A–E, G–L | Kernarm, Eingriffs-Checkpoint |
| 2 | A | pythia-160m @ step99000 | A–E, G–L | Kernarm, zweiter Eingriffs-Checkpoint |
| 3 | A | pythia-160m @ step143000 | A–E, G–L, `--use-sae` | SAE-Arm (Feature-Richtungen; H/J/K mit publiziertem SAE) |
| 4 | B | pythia-160m, 27 Checkpoints | F (B, D limitiert) | Verlauf über das Training |
| 5 | B | pythia-410m @ step143000 | A–E | Skalierungsarm (SAE nur MLP-Hookpoints) |
| 6 | B | pythia-1b, pythia-1.4b @ step143000 | F (B, C, D limitiert; nur C1) | Skalierungsarm |
| 7 | B | pythia-160m @ step98000 (+ step4000, step143000) | K, `--ttt-checkpoints` | Memorisationsgeschwindigkeit über die Dossier-Checkpoints (`runs_ttt`) |

Schätzung (A100): ein Forward von 160M auf ≤ 320 Tokens ≈ 3 ms; Stage C ≈ 3 000 Forwards,
Stage D1/D2 ≈ 50 000 Forwards (≈ 5 min), Generierungsstufen D3–D7 ≈ 300 000 Token-Schritte
gebatcht (≈ 30–60 min); Stufen G–L zusammen 1–2 h (Stufe H dominiert: ≈ 110 000
Forward+Backward-Pässe der λ-Pfade im vollen Budget; Stufe J zieht 2 000 Pile-Sequenzen
per Range-Request, gecacht). Shards 1–3 je 2–5 h, Shard 5 1–3 h, Shard 4 2–4 h, Shard 6
3–6 h, Shard 7 1–2 h. Jede Shard schreibt eigene Laufverzeichnisse; `python -m lsne.cli
merge` fügt sie zusammen. Shards 1–3 nennen `--stage A,B,C,D,E,G,H,I,J,K,L` statt `all`,
weil `all` auch Stufe F (den Checkpoint-Sweep) anstoßen würde.

## Ausgabe-Layout

```
<OUT_ROOT>/                         Colab: /content/drive/MyDrive/LSNE ; sonst $LSNE_RUNS oder ./runs
├── runs/
│   ├── A/pythia-160m@step98000/    manifest.json, families.csv, variants.csv, checks.csv, tokens.csv, ...
│   ├── B/pythia-160m@step98000/    baseline.csv, cosines.csv, unembed.csv, directions.npz, directions_meta.json, ...
│   ├── C/pythia-160m@step98000/    grid.csv, contrasts.csv, contrasts_boot.csv, attribution.csv, grad_x_act.csv, ...
│   ├── D/pythia-160m@step98000/    damage.csv, damage_phase.csv, lens.csv, hysteresis.csv, relaxation.csv, ca.csv, ...
│   ├── E/pythia-160m@step98000/    tn_compare.csv, tn_summary.csv, tn_paths.csv, tn_basis.csv
│   ├── F/pythia-160m@step143000/   plan.csv, runs.csv
│   │   └── shard0of1/<stage>/<modell>@<revision>/   Teilläufe des Sweeps (gleiches Schema)
│   ├── G/pythia-160m@step98000/    clouds, geometry, pair_ordering, g0_gate, depth_trajectory, contraction(_boot), position_traj, steering(_boot), coupling
│   ├── H/pythia-160m@step98000/    lambda_paths, lambda_steps, lambda_corr, lambda_boot, lambda_features(_summary), lambda_original
│   ├── I/pythia-160m@step98000/    compass(_summary), atlas_overlap, domain_axes, direction_domain(_T/_boot), head_effects, head_direct, worst_head, head_steering(_boot), directions_domain.npz
│   ├── J/pythia-160m@step98000/    pile_difficulty, pile_ranking, pile_rank_check, rdx, rdx_summary, rdx_stability, sharing
│   ├── K/pythia-160m@step98000/    ttt_corpora, ttt_memorisation(_boot), ttt_training, ttt_before_after, ttt_effects
│   ├── L/pythia-160m@step98000/    transfer_test, transfer_summary, layer_matching, alignments, transplant, transplant_summary
│   ├── merged/                     Ausgabe von `lsne.cli merge` (Spalten run_id, run_stage, run_model, run_revision)
│   └── run_report_<modell>_<rev>_<dtype>.json
├── runs_ttt/K/…                    Shard 7 (Stufe K über drei Checkpoints), getrennt vom Kernlauf
├── runs_trace/L/<modell>@<rev>/    Notebook 12: Stufe L mit dem Spurenmodell, getrennt vom Stand-in-Lauf in runs/L/…
├── runs_ctl/
│   ├── D/<modell>@<rev>/           Notebook 14: Kontrolllauf von Stufe D (Nachtrag 6.11): damage_ctl, damage_ctl_profile,
│   │                               damage_ctl_checks, damage_ctl_contrasts, manifest.json (control = token_language)
│   └── logs/                       Logs der CLI-Aufrufe von Notebook 14 (Vorabprüfung, Kontrolllauf)
├── prereg_eval/<modell>@<rev>/     Notebook 13: prereg_results.{csv,md}, prereg_runs.csv, prereg_meta.json
│   └── archive/<zeitstempel>/      frühere Auswertungen; ein neuer Lauf verschiebt die vorige hierher, statt sie zu überschreiben
├── cache/                          state2143/ (Anker), unigram/ (Zählungen), pile_nll/ (Stufe J) — LSNE_CACHE
├── figures/<notebook>/             PNG-Abbildungen der Notebooks
└── summary_<notebook>_*.json       kurze Laufzusammenfassung je Notebook
```

Jede `manifest.json` enthält Git-SHA, Paketversion, Modellspezifikation, dtype, Hook-Site,
Anker-Hashes, Zeitstempel, Tabellengrößen und Laufzeit; ein Stufenfehler hinterlässt
`error.txt` und `status: "error"`, die übrigen Stufen laufen weiter. Die Manifeste der
Stufen G–L enthalten zusätzlich `substages` (Laufzeiten und Fehler pro Substufe) und
`prereg` (welche Tabelle und welche Zeilen jeden präregistrierten Kontrast tragen;
vollständige Karte Endpunkt → CSV → Spalten in `docs/STAGES.md`). Platzhalter sind nie
stumm: Zeilen aus einem synthetischen SAE tragen `sae_source = "synthetic_random"`,
TTT-Zeilen auf dem Platzhaltertext `is_placeholder = True`.

## Präregistrierung

`docs/PREREGISTRATION.md` wird vor dem ersten Bestätigungslauf eingefroren (Git-Tag
`prereg-v1`): primärer Endpunkt λ(K, ε, dir) mit H1 „λ(K>0, lang_dom) − λ(K=0, lang_dom) > 0,
monoton in K und größer als derselbe Kontrast mit `rand`“, Familien-Cluster-Bootstrap
(10 000 Ziehungen), Mindesteffekt Δλ ≥ 0,05 pro Layer; sekundäre Endpunkte (T(K),
K × C, Delimiter-Recovery, Hysteresefläche, Nichtmonotonie, Transfer von `ent`,
Geometrie-Gate G0, Kontraktion G1, Steering-Validität T3, λ-Dynamik und B(x)(λ) für
State 2143, Domänen-Kompass, Richtung × Domäne, Head-Arm, RDX-Anteil,
Schwierigkeitsrangfolge, Test-Time-Training, Transplantations-Validierung — Kontraste
1–18) BH-korrigiert; Ausschlussregeln vor Sichtung der Modellwerte. Alles darüber hinaus
(Phasendiagramme, Checkpoint-Sweeps, Skalierungsarme, SAE-Feature-Auswahl, FDT) ist
explorativ. `docs/STAGES.md` ordnet jedem Kontrast die Stufe, die Tabelle und die Spalten
zu, die ihn tragen. Nach Lauf 1 kamen datierte Nachträge hinzu (`docs/PREREGISTRATION.md` §6),
zuletzt 6.11 vom 28.10.2026: die Kontrolle der Sprache an der Injektionsstelle mit einem eigenen
Kontrolllauf von Stufe D (Notebook 14); ohne gültigen Kontrolllauf ist der Primärendpunkt
höchstens „offen“.

## Grenzen

- Eine Sprachachse ist eine lineare Messkonstruktion, keine „innere Sprache“.
- Nicht-Gleichgewichts-Signaturen sind Analogien aus der Gitterphysik; sie werden als
  vorab definierte Kenngrößen mit Nullmodellen geprüft, nicht als physikalische Identität
  behauptet.
- Familien sind die statistische Einheit; Checkpoints derselben Trainingsbahn sind keine
  Replikationen; Effekte auf Entwicklungsfamilien sind keine Bestätigung.
- Das Tensornetz ist eine Linearisierung erster Ordnung mit eingefrorenen Attention-
  Mustern und LN-Skalen in einem PCA-Unterraum (r = 64) und Positionsfenster (32 Tokens);
  TNCO verbessert nur die Kontraktionskosten, nie die Näherung.
- Die SAEs wurden auf dem Endcheckpoint trainiert (Dossier: FVU 0,88 an Layer 11 bei
  98k); SAE-Analysen laufen primär auf step143000, bei FVU > 0,4 gelten sie als nicht
  zertifiziert.
- Deutsche Segmente haben mehr Subwort-Tokens pro Wort; Längen werden als Kovariate
  mitgeführt, nicht wegnormiert.
- Die Literaturstellen „Axis of Ignorance/Uncertainty“ und „The Geometry of Pain“ konnten
  unter diesen Titeln nicht identifiziert werden; verwendet werden die
  operationalisierbaren Kerne (`freq`-Achse, Difference-of-Means-Achse).

## Literatur

- Biderman, S. et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across
  Training and Scaling.* ICML 2023. — Modelle, Checkpoints, deterministische Datenreihenfolge.
- Im, S. et al. (2026). *Behavioral Counterfactuals* (Counterfactual-Attribution von
  Prompt-Interventionen; Delimiter-Anteil). arXiv:2610.30630.
- Tagliabue, F. et al. (2026). *The Pain Axis* (lineare Achse per Difference-of-Means,
  Denoising über die oberste Hauptkomponente, Unembedding-Projektion). arXiv:2609.16247.
- Godoy Ortiz, A. (2026). *Meaning Contracts* (lokale Geometrie, trunkierte Mahalanobis-
  Distanz, Log-Volumen; Stufe G/`lsne.geometry`). arXiv:2610.30715.
- Biggio, B. et al. (2013/2017). *Evasion attacks against machine learning at test time*
  (λ-Mimikry-Dynamik, `lsne.lambda_dynamics`). arXiv:1708.06131.
- Gai, Y., Zhou, L., Qin, K., Song, D. & Gervais, A. (2023). *BlockGPT* (Causal-LM auf
  EVM-Ausführungsspuren; Quellmodell-Vorbild für Stufe L, keine Gewichte verfügbar:
  `docs/BLOCKGPT_FEASIBILITY.md`). arXiv:2304.12749.
- EleutherAI (2024). *sparsify* — TopK-SAEs für Pythia (`EleutherAI/sae-pythia-160m-32k`,
  `EleutherAI/sae-pythia-410m-65k`). https://github.com/EleutherAI/sparsify
- Google Research (2025). *TNCO — Tensor Network Contraction Optimizer.*
  https://github.com/google-research/tnco ; Abanin, D. et al. (2025), *Nature* —
  Kontraktionsreihenfolgen großer Tensornetzwerke.
- Zhang, F. & Nanda, N. (2024). *Towards Best Practices of Activation Patching in Language
  Models: Metrics and Methods.* ICLR 2024. — Patching-Metriken (Log-Prob-Recovery, Nullmodelle).
- Guzmán, G. A. et al. (2017). *Metrics for Modeling Code-Switching Across Corpora.*
  Interspeech 2017. — Code-Switching-Metriken (M-Index, I-Index, Burstiness; hier F100
  und Wechselabstände).

## Repository-Struktur

```
lsne/            Paket: config, anchor, pile_fetch, model, stimuli, lang_id, directions, sae, patching,
                 dynamics, generation, tn_lattice, geometry, trajectory, lambda_dynamics, atlas, heads,
                 rdx, pile_domains, sharing, ttt, transplant, stats, io, cli,
                 prereg_eval (präregistrierte Auswertung vorhandener Läufe, CPU)
lsne/chain       Quelladapter für ein BlockGPT-artiges Spurenmodell (Xatu-Traces 2016, docs/TRACE_SOURCE_SPEC.md)
lsne/experiments Stufen A-L (run() je Stufe) + stage_d_control (Kontrolllauf von Stufe D, Nachtrag 6.11)
                 + common (Limits, Stimulusbank, Richtungsbank, Substufen, Familiengruppen)
data/stimuli     Familien F01-F20, DEV01-DEV04 (Formulierungen A-D), original_2143.json, SCHEMA.md, review.csv
data/lexicon     Funktionswort-Lexikon (Sprachklassifikation), domains.json (Domänenlexika E/D)
data/ttt, data/atlas, data/s4   Ablagen für die Eingaben des Auftraggebers (nur die Notizen sind eingecheckt)
docs/            DESIGN.md, PREREGISTRATION.md, SPEC.md, STAGES.md, RUNBOOK_COLAB.md, TNCO_INTEGRATION.md,
                 BLOCKGPT_FEASIBILITY.md, TRACE_SOURCE_SPEC.md, TRACE_SOURCE.md, RESULTS_RUN1.md,
                 RESULTS_PREREG.md, LITERATUR.md, ROUND2_ITEMS.md, berichte/ (PDF-Berichte)
jobs/shards.yaml Rechenplan (7 Shards)
notebooks/       generierte Colab-Notebooks 00-14
scripts/         build_notebooks.py
tests/           pytest (CPU, pythia-14m); test_experiments_smoke.py = Rauchtest A-L über die CLI
```

Lizenz: Apache-2.0.
