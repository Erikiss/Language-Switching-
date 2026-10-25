# LS-NEQ: Sprachwechsel → Nicht-Gleichgewicht auf State 2143

Versuchsdesign, Stand 25.10.2026. Grundlage: Forschungsdossier „State 2143“ (25.10.2026),
Kurzfassung „TNCO × Pythia-410M (UIP)“, Im et al. 2026 (Behavioral Counterfactuals,
arXiv:2610.30630), Tagliabue et al. 2026 (The Pain Axis, arXiv:2609.16247).

Dieses Dokument legt fest, **was gemessen wird, warum, und was als Nachweis zählt**.
Es enthält keine Messergebnisse. Code: Paket `lsne/`, Notebooks: `notebooks/`.

---

## 0. Kurzfassung

Ergänzung 25.10.2026 (zweite Rückmeldung des Auftraggebers): Der Zwischenschritt
„Verdichtung der lokalen semantischen Geometrie“ (Godoy Ortiz 2026, *Meaning Contracts*,
arXiv:2610.30715) wird als eigene Stufe G aufgenommen (§4.4) und die darauf aufbauende
Trajektorien-Analyse als §4.5. Sprachwechsel ist dabei das Mittel, Verdichtung
kontrolliert zu stellen; der Gegenstand ist nicht die Tokenisierung seltener Wörter.

Das Dossier zeigt für State 2143 (Pythia-160M, 208-Token-Kontext, Ziel „bey“) einen
reproduzierbaren lokalen Eingriffseffekt, aber keinen Sprachwechselmechanismus: der
Originalkontext enthält null Sprachwechsel. Das hier aufgesetzte Programm macht
Sprachwechsel zu einer **kontrollierten, skalierbaren Eigenschaft** des Kontexts
(diskret über Wechselzahl K und Strukturstufe C; kontinuierlich über eine Dosis ε
entlang einer Sprachachse oder eines SAE-Features) und prüft drei Fragen:

1. **Counterfactual-Frage (Im et al.):** Welche Positionen vermitteln die Wirkung eines
   Sprachwechsels auf die Zielvorhersage? Insbesondere: tragen Leerzeichen-, Satzzeichen-
   und Grenz-Tokens („Füllwörter“/Delimiter) einen überproportionalen Anteil, wie es
   Im et al. für Prompt-Interventionen fanden (Attribution konzentriert auf Whitespace
   und Delimiter; Patching dieser Tokens allein stellt 11–16 % des Effekts wieder her)?
2. **Achsen-Frage (Pain Axis):** Gibt es eine lineare Sprachachse (Difference-of-Means
   E↔D), wie steht sie geometrisch zur State-2143-Entropierichtung, zu einer
   „Häufigkeitsachse“ (Masse auf hochfrequenten Tokens) und zu Zufallsrichtungen, und
   welche Tokens befördert sie durch die Unembedding-Matrix?
3. **Nicht-Gleichgewichts-Frage (Gitterphysik):** Wenn man Sprachwechsel hochskaliert,
   erreicht die Dynamik auf dem (Layer × Position)-Gitter Zustände, die sich mit den
   Standard-Diagnostiken der Nicht-Gleichgewichts-Gitterphysik von einem relaxierenden
   („Gleichgewichts-“) Regime unterscheiden: wachsende Störungsausbreitung
   (damage spreading), Hysterese und langsame Relaxation in der Generierungsdynamik,
   nicht-monotone Konvergenz der Zwischenschicht-Vorhersagen, Bruch der linearen
   Antworttheorie?

Primärer Endpunkt (präregistriert, siehe `docs/PREREGISTRATION.md`): der
Wachstumsexponent λ der Störungsausbreitung entlang der Tiefe als Funktion von
(K, ε, Richtung), ausgewertet auf dem gemeinsamen englischen Schlussabschnitt, mit
isotropen Zufallsrichtungen gleicher RMS-Dosis als Nullmodell.

---

## 1. Verankerung: Was State 2143 hier exakt ist

| Merkmal | Wert | Prüfung im Code |
|---|---|---|
| Modell | `EleutherAI/pythia-160m` (GPT-NeoX, 12 Blöcke, d=768, 12 Heads, parallele Residuen) | `lsne.config.STATE2143` |
| Datensatz | `EleutherAI/pile-standard-pythia-preshuffled`, Sequenzlänge 2049, Batch 1024 | — |
| Globaler Sequenzindex | 5 025 183 = 4907 · 1024 + 415 | `lsne.anchor` |
| Kontext / Ziel | erste 208 Tokens / Token 22905 „bey“ | SHA256-Prüfung |
| SHA256 (2049 uint16 LE) | `bdd4bbdd2b8138437e052b2f0e3aeb212a563c25fe447c920b0e9c6ee76f55d4` | verifiziert 25.10.2026 |
| SHA256 (208 Kontext-Tokens, uint16 LE) | `84c8f1b6a7a933f53f3b21e353c0c3af219d3ccfbcb7092584a9c6ad1035a750` (Dossier: Präfix `84c8f1b6a7a933f5`) | verifiziert 25.10.2026 |
| Eingriffsstelle (Dossier) | Eingang Block 11 (nullbasiert), Position 207, 768 Komponenten | `HookSite(layer=11, site="block_input")` |
| Dosis | δ = ε · RMS(h) · g/RMS(g), ε = 0,01 | `lsne.model.make_dose_delta` |
| Checkpoints | step98000, step99000 (Eingriff); step143000 (SAE); 4000/5000 (Exposition) | HF-Revisionen |

**Neu gegenüber dem Dossier:** Der exakte Kontext wird nicht aus lokalen Dateien,
sondern per HTTP-Range-Request direkt aus dem Datensatz geholt (4098 Bytes ab
Byte-Offset 5 025 183 · 4098 im Shard `document-00000-of-00020.bin`) und gegen beide
Hashes geprüft. Damit ist der Anker ohne private Dateien reproduzierbar. Der Text
wird nicht ins Repository geschrieben.

**Hook-Konventionen (verifiziert mit transformers 5.20 an pythia-14m):**
`output_hidden_states` liefert `hidden_states[ℓ]` = Eingang von Block ℓ (ℓ = 0…11)
und `hidden_states[12]` = Ausgabe der *finalen LayerNorm*, nicht den rohen Residualstrom.
Block-Ausgaben werden deshalb per Forward-Hook abgegriffen. Die SAE-Hookpoints
`layers.N` der EleutherAI-SAEs sind Block-Ausgänge: die Dossier-Eingriffsstelle
„Eingang Block 11“ ist identisch mit dem SAE-Hookpoint `layers.10`, nicht `layers.11`.
Beide werden berichtet.

---

## 2. Stimuli: Sprachwechsel als kontrollierte Variable

### 2.1 Bedeutungseinheiten mit drei Segmenten

Jede Familie besteht aus 10 Bedeutungseinheiten. Jede Einheit hat drei Segmente,
jeweils in Englisch (E) und Deutsch (D):

```
subj | pred | adj        z. B.  "Kraft" | "reads the report" | "in the morning"
                                "Kraft" | "liest den Bericht" | "am Morgen"
```

Daraus werden die drei Strukturstufen des Dossiers konstruiert, **ohne** separate
Texte zu schreiben:

| Stufe | Realisierung eines Wechsels an der Grenze Einheit k → k+1 |
|---|---|
| C1 Satzgrenze | Einheit k vollständig in Sprache m_k, Einheit k+1 vollständig in m_{k+1} |
| C2 Teilstruktur | `subj_k pred_k` in m_k, `adj_k` in m_{k+1} (Wechsel vor dem Adjunkt; Satzrahmen bleibt offen, keine markierte Zielabhängigkeit überspannt) |
| C3 Abhängigkeit | `subj_k` in m_k, `pred_k adj_k` in m_{k+1} (Subjekt-Verb-Kongruenz über die Sprachgrenze) |

Die Wechselzahl K wird durch die Sprachmasken des Dossiers gesetzt (Einheit 1 und 10
immer E, Einheit 9 immer D in gemischten Bedingungen, letzter Wechsel stets vor
Einheit 10):

```
K=0 : E E E E E E E E E E
K=2 : E E E E D D D D D E
K=4 : E E D D E E D D D E
K=8 : E D E D E D E D D E
```

**Skalierung über das Dossier hinaus** (explorativ, für die Nicht-Gleichgewichtsfrage):
Masken auf Segmentebene mit K = 16 und K = 28 (Wechsel an fast jeder Segmentgrenze).
Diese Bedingungen haben keine eindeutige C-Stufe und werden als `C="seg"` geführt.

### 2.2 Ziel und englischer Schlussabschnitt

Jede Familie hat einen festen englischen Schlussabschnitt (`tail_E`), der in allen
Varianten identisch ist, und ein Zieltoken (`target`), das *nicht* Teil des Kontexts
ist. Gemessen wird log p(target | Kontext). Der Code prüft für jede Variante:
Ziel-Token-ID identisch, Token-IDs des Schlussabschnitts identisch (BPE-Grenzen),
Zieltoken erscheint nicht im Kontext (kein Ziel-Leak).

### 2.3 Umfang

- 12 Testfamilien × 2 Formulierungen (A/B) + 4 Entwicklungsfamilien (nur für
  Dosis-/Schwellenwahl, nie für Bestätigung).
- Pro Familie und Formulierung: 1 (K=0) + 3·3 (K∈{2,4,8} × C1–C3) + 2 (K=16, 28) = 12 Varianten.
- Zusätzlich die **Original-Fallstudie**: der 208-Token-Kontext von State 2143 wird
  in 7 Einheiten segmentiert (Token-Grenzen in `data/stimuli/original_2143.json`),
  Einheiten 1–6 erhalten deutsche Übersetzungen, Einheit 7 (`…Brecht's _Dream of Fewkoom`)
  bleibt englisch; Ziel bleibt „bey“ (22905). Der englische Originaltext wird zur
  Laufzeit aus dem Datensatz rekonstruiert.

### 2.4 Annotation pro Token

Für jede Variante: Sprache des Segments (E/D), Einheitsindex, Tokenklasse
(`ws` reiner Whitespace/Zeilenumbruch, `punct`, `word_start` mit führendem Leerzeichen,
`word_cont` Subwort-Fortsetzung, `boundary` erstes Token nach einem Wechsel),
Abstand zum letzten Wechsel, F100 = 100·K/(N−1), Längen in Wörtern und Tokens.
Deutsche Segmente erzeugen systematisch mehr Subwort-Tokens pro Wort; die
Tokenlängen werden deshalb immer mitberichtet (Konfundierung, nicht wegnormiert).

---

## 3. Richtungen (Eingriffsvektoren)

| Kürzel | Konstruktion | Rolle |
|---|---|---|
| `ent` | ∂H/∂h am Eingang Block 11, Position 207 (bzw. letzte Kontextposition), Dossier-Konstruktion; optional ARI-orthogonal, wenn `axes.npz` aus S4 vorliegt | Replikation / Transferarm |
| `lang_dom` | Difference-of-Means D − E über die letzten Tokens gepaarter Segmente, pro Layer; „denoised“: Projektion der obersten gemeinsamen Hauptkomponente heraus (Pain-Axis-Vorgehen) | Sprachachse |
| `lang_probe` | logistische Regression E/D auf denselben Aktivierungen | Kontrolle für `lang_dom` |
| `sae_f` | Decoder-Zeilen sprachselektiver TopK-SAE-Features (`EleutherAI/sae-pythia-160m-32k`, k=32, 32 768 Latente, Hookpoints `layers.10` und `layers.11`) | Feature-Steering |
| `freq` | mittlere Unembedding-Zeile der 100 häufigsten Pile-Tokens minus Gesamtmittel (Unigramm-Zählung aus 2 000 per Range-Request gezogenen Sequenzen) | „Häufigkeitsachse“ |
| `rand` | 32 isotrope Zufallsrichtungen, jeweils ± | Nullmodell |

Geometrie-Bericht: Kosinusmatrix aller Richtungen pro Layer; Projektion jeder
Richtung durch finale LN + Unembedding (beförderte/unterdrückte Tokens).
Hinweis zum SAE: Die SAEs wurden auf dem Endcheckpoint trainiert; das Dossier meldet
FVU 0,88 an Layer 11 bei 98k (Gate nicht bestanden). SAE-Analysen laufen daher
primär auf `step143000`; auf 98k/99k wird FVU vor jeder Nutzung erneut gemessen und
bei FVU > 0,4 als „nicht zertifiziert“ markiert.

---

## 4. Messgrößen

Notation wie im Dossier: p₀(x) unverändert, p_δ(x) mit Eingriff; NLL = −ln p;
B(x) = ln p_δ − ln p₀ (Interventionsnutzen); T = ln p₀(x_gemischt) − ln p₀(x_einsprachig)
(reiner Texteffekt); I = B(x_gemischt) − B(x_einsprachig).

### 4.1 Counterfactual-Attribution (Frage 1)

Für jedes Paar (x_mono, x_mixed) derselben Familie/Formulierung werden auf dem
gemeinsamen Schlussabschnitt (positionsgleich vom Ende her) Aktivierungen von x_mono
in den Lauf von x_mixed gepatcht (Block-Eingang, Layer-Raster {2,5,8,11}), getrennt
nach Tokenklasse. Wiederherstellungsanteil R_c = (ln p_patched − ln p_mixed)/(ln p_mono − ln p_mixed).
Nullmodell: gleich viele zufällig gewählte Positionen (10 Ziehungen). Zusätzlich
Gradient×Aktivierung je Position. Erwartung nach Im et al.: R_ws+punct+boundary > R_random.

### 4.2 Nicht-Gleichgewichts-Diagnostiken (Frage 3)

Gitter: Sites (ℓ, t), ℓ = 0…12 (Block-Eingänge + Ausgang), t = Position.
„Zeit“ = Tiefe ℓ (deterministische Abbildung mit nicht-lokaler kausaler Kopplung);
zweite Zeit = autoregressiver Schritt; langsame Achse = Trainings-Checkpoint.

| Diagnostik | Definition | Gleichgewichts-Erwartung | Nicht-Gleichgewichts-Signatur |
|---|---|---|---|
| D1 Störungsausbreitung | δ an (ℓ₀,t₀); D(ℓ,t) = ‖h′−h‖/‖h‖; λ = Steigung von ln D über ℓ (Fit ab ℓ₀+1); Ausbreitungsfront über t | λ ≤ 0 (Störung relaxiert) | λ > 0, wachsend mit K bzw. ε; Front breitet sich über Positionen aus |
| D2 Tiefen-Monotonie | KL(p_final ‖ p_ℓ) über ℓ (Logit-Lens); Nichtmonotonie-Index = Σ positiver Zuwächse / Gesamtvariation | monoton fallend | Überschwingen, späte Umkehr |
| D3 Hysterese | Generierung (64 Tokens, 8 Samples) mit Dosisrampe 0→ε_max→0 in 8 Stufen; f_D(ε) aufwärts vs. abwärts; Fläche A_hyst | A ≈ 0 wie bei unabhängigen Neustarts | A > Null-Verteilung (Gedächtnis durch generierte Tokens, keine Relaxation im Fenster) |
| D4 Relaxation | Dosissprung (Quench) 0→ε bei Token 16; f_D(t), Entropie(t); τ aus Exponentialfit | kurzes, dosisunabhängiges τ | τ wächst nahe einer Schwelle ε_c (kritische Verlangsamung) |
| D5 Wechselstatistik | Abstände zwischen Sprachwechseln in generiertem Text | geometrisch (Markov) | schwere Ränder (Lawinen) |
| D6 Suszeptibilität | χ(ε) = d f_D/dε; Varianz von f_D über Samples | glatt | Peak / Sprung (erste Ordnung) |
| D7 Fluktuation–Antwort | T_eff = Var₀(m)/χ für Ordnungsparameter m = Projektion auf `lang_dom` | konstant über Bedingungen | bedingungsabhängig (FDT-Bruch) — explorativ |
| D8 Lineare Antwort | Abweichung der gemessenen Störung vom linearen Vorhersagemodell (Jacobi-Kette bzw. Tensornetzwerk, Abschnitt 5) | klein, ∝ ε² | groß, bereits bei kleinem ε |
| D9 CA-Komplexität | Symbolisierung S(ℓ,t) ∈ {N,E,D} über Logit-Lens-Sprachmasse; Raum-/Zeit-Entropie, Aktivität, LZ76 | niedrig/stabil | steigend mit K, ε |

Phasendiagramme: jede Kenngröße über (ε, K) und (ε, C), getrennt für `ent`, `lang_dom`,
`sae_f`, `rand`.

### 4.4 Zwischenschritt: Lokale Geometrie („Meaning Contracts“) unter Sprachwechsel

Godoy Ortiz (2026, arXiv:2610.30715) zeigt für vier Modellfamilien (darunter Pythia-2.8B):
(i) nach Entfernen der dominanten Varianzrichtungen (Top-k-PCA-Deflation) ordnen sich
Satzpaare nach semantischer Verwandtschaft; (ii) exakte Übersetzungen bilden unter
trunkierter Mahalanobis-Distanz den **invarianten Kern** einer Bedeutung, freie
Realisierungen die weite, anisotrope Streuung darum; (iii) semantische Verfeinerung ist
**gerichtete Verschiebung in engere Nachbarschaften** (Abnahme des Log-Volumens der
lokalen Realisierungswolke, über Kontrollen hinaus); (iv) Mean-Pool-Readout zeigt die
Kontraktion auf Satzebene, Last-Token-Readout schärfere anisotrope Endpunkte (niedrigere
Participation Ratio); die Pro-Position-Inkremente des Log-Volumens wachsen über späte
Präfixpositionen. Das Paper manipuliert nicht; die Brücke zur kausalen Intervention ist
dort explizit offen.

**Was hier geprüft wird (der vom Auftraggeber benannte Zwischenschritt):** Findet man
eine Größe, die mit dieser Verdichtung korreliert und die man kontrolliert stellen kann?
Educated guess: Sprachwechsel. Konkret wird die lokale Geometrie des Papers auf dem
eigenen Stimulusraster **unter Sprachwechsel als Störgröße** gemessen und erst danach die
Trajektorie betrachtet (§4.5). Es geht ausdrücklich **nicht** um seltene, buchstabierte
Namen wie „Fewkoom|bey“: deren Tokenisierung ist eine Konfundierung, die kontrolliert
wird (Zielwörter der neuen Familien sind häufige Ein-Token-Wörter; die Original-Fallstudie
bleibt getrennt), nicht der Forschungsgegenstand.

Messgrößen (Modul `lsne.geometry`, Definitionen wie im Paper):

| Größe | Definition | Readout |
|---|---|---|
| Deflation | X^(−k) = X̃ − (X̃ V_kᵀ) V_k, k ∈ {0, 1, 5, 10}, V_k aus allen gepoolten Zuständen eines Layers | beide |
| Trunkierte Mahalanobis | d_M² = (x−y)ᵀ U_r Λ_r⁻¹ U_rᵀ (x−y), Σ_ℓ aus allen gepoolten Zuständen des Layers, r = min(128 bzw. 256, N−1, d) | Mean / Last |
| Log-Volumen | Σ_i log max(λ_i, ε) der Kovarianz der lokalen Realisierungswolke (n Realisierungen) | beide |
| Participation Ratio | (Σλ_i)² / Σλ_i² | beide (ordinaler Indikator, wie im Paper) |
| Verschiebung | d(mean(Bedingung), mean(K=0-Bedingung)) euklidisch und Mahalanobis | beide |
| Pro-Position-Inkrement | Log-Volumen der Wolke an Position t (vom Ende her ausgerichtet auf dem gemeinsamen Schlussabschnitt; Präfix in relativen Positionsbins) | Last-Token je Position |

**Realisierungswolke pro Bedingung (Familie, K, C):** Formulierungen × alle zulässigen
Masken mit genau K Wechseln (Einheit 1 und 10 E, Einheit 9 D). Für K=2 gibt es 8
zulässige Einheitsmasken, für K=4 und K=8 mehr; mit zwei Formulierungen sind das ≥ 16
Realisierungen wie im Paper (n = 16). Für K=0 liegen nur die Formulierungen vor; deshalb
werden die Familien um die Formulierungen C und D erweitert (Stufe A, Nachtrag), sodass
auch K=0 eine Wolke mit n ≥ 4 hat und die Dosis-Trajektorie (§4.5) an jeder
Realisierung gemessen werden kann.

Vorab definierte Tests:
- G0 *Replikation am Raster:* exakte Übersetzungspaare (A_E, A_D) sind unter
  Mahalanobis näher als freie Realisierungen (A_E, B_E) und diese näher als
  Fremdfamilien — die Kern-und-Streuung-Ordnung des Papers muss auf Pythia-160M mit
  unserem Material sichtbar sein, sonst ist Kontraktion dort nicht interpretierbar.
- G1 *Korrelation:* Δ log-Vol(K) − Δ log-Vol(K=0) und ΔPR als Funktion von K und C,
  pro Layer und Readout; Vorzeichen offen (Kontraktion **oder** Expansion sind beides
  informative Befunde; „Atmen“ = Vorzeichenwechsel über die Tiefe, §4.5).
- G2 *Kontrollen:* einsprachige Umbaukontrolle (gleiche Segmentierung, alles E),
  Längen-Kovariate (deutsche Segmente haben mehr Tokens), Deflationsstufe k,
  Rang r, Familien-Bootstrap.
- G3 *Positivkontrolle (optional):* die veröffentlichten Verfeinerungsketten des Papers
  (Repo `adgodoyo/Meaning_Contracts`) auf Pythia-160M nachgerechnet; zeigt, ob das
  160M-Modell die Verfeinerungskontraktion überhaupt trägt.

### 4.5 Trajektorie: Verdichtung als Ordnungsparameter der Dynamik

Erst wenn G1 ein Signal liefert, wird die **Trajektorie** betrachtet (Modul
`lsne.trajectory`). Gemeint ist die Bahn der lokalen Geometrie durch die drei Achsen,
die der Q-Learning-Analogie des Auftraggebers entsprechen: Zustand s ↔ lokale Geometrie
(Log-Volumen, PR, Verschiebung), zustandsbeschreibender Parameter ↔ Gitterkoordinaten
(Tiefe ℓ, Position t), θ ↔ Trainings-Checkpoint. Konkret:

| Trajektorie | Messung | Frage |
|---|---|---|
| T1 Tiefe | Log-Vol(ℓ), PR(ℓ), Verschiebung(ℓ) je Bedingung; Atem-Index = Zahl der Vorzeichenwechsel von Δ_ℓ log-Vol und Amplitude | kontrahiert der Raum monoton oder „atmet“ er, und ändert Sprachwechsel das Muster? |
| T2 Position | Pro-Position-Inkremente (Paper, Abb. 6) entlang des Präfixes und des gemeinsamen Schlussabschnitts, relativ zur letzten Wechselstelle | wandert die Verdichtung mit den Wechselstellen? |
| T3 Dosis | Steering mit `lang_dom`/`sae_f` bei Dosis ε: Verschiebung des gesteuerten Zustands relativ zur K=0-Wolke (Mahalanobis) und Kosinus zwischen Steuerverschiebung und der diskreten K-Verschiebung mean(K) − mean(0) | reproduziert kontinuierliches Steuern die Geometrie echter Wechsel? (Voraussetzung dafür, Sprachwechsel „hochzuskalieren“) |
| T4 Training | T1–T3 über die Dossier-Checkpoints | wann im Training entsteht das Muster? |
| T5 Kopplung | Korrelation von Δ_ℓ log-Vol mit der Störungswachstumsrate λ_ℓ (D1) und der Lens-KL (D2) über Layer | ist Verdichtung der Ordnungsparameter der Nicht-Gleichgewichts-Signaturen? |

Auswertung: Familien-Bootstrap; Trajektorien werden als Kurven mit Konfidenzbändern
berichtet, Phasenlage (Layer des stärksten Inkrements) als Kennzahl. T3 ist der
entscheidende Validitätstest für die gesamte Steering-Skalierung: fällt er negativ aus,
werden Dosis-Phasendiagramme (§4.2) nur noch als Eingriffs-, nicht als
Sprachwechselbefunde interpretiert.

### 4.3 Dossier-Kernkontraste (Frage 2 und Replikation)

T(K, C), B(x) pro Richtung, I(K, C). Wechselwirkung K × C, Transfer `ent` von
State 2143 auf neue Familien vs. je Familie neu berechnete Richtung (getrennte Arme).

---

## 5. Tensornetzwerk / TNCO: wo es tatsächlich hingehört

Die Kurzfassung stellt klar: TNCO optimiert Kontraktionsreihenfolgen, es ist kein
Interpretierbarkeitsverfahren; Softmax, LayerNorm und GELU lassen sich nicht
wegkontrahieren. Beides wird hier respektiert:

- Der **linearisierte** Transformer (Jacobi-Matrix jedes Blocks am Arbeitspunkt, bei
  eingefrorenen Attention-Mustern und eingefrorener LN-Skalierung) ist eine
  Transfer-Matrix-Kette auf dem (Layer × Position)-Gitter. In faktorisierter Form
  (pro Block: Skip ⊕ Σ_Heads A^h ⊗ M^h ⊕ positionsweise MLP-Blöcke, mit Branch-Index)
  ist das ein echtes Tensornetzwerk, dessen Kontraktionsreihenfolge nicht trivial ist —
  genau das Problem, das TNCO löst (`lsne.tn_lattice`). Ergebnis: die lineare
  Antwortfunktion („Green-Funktion“) χ[(ℓ₀,t₀) → (L,t)] des Netzes.
- Drei Ebenen werden verglichen: (a) gemessene nichtlineare Störung (D1),
  (b) exakte lineare Antwort per Autograd-JVP, (c) faktorisiertes Tensornetzwerk mit
  TNCO-Pfad (Fallback `opt_einsum`). (a)−(b) misst Nichtlinearität (D8),
  (b)−(c) misst den Fehler der Faktorisierungsnäherung.
- Für Pythia-160M wird das Netz standardmäßig in einem r = 64-dimensionalen
  PCA-Unterraum des Residualstroms und einem Positionsfenster von 32 Tokens
  aufgebaut; auf A100 ist die volle Breite (d = 768) optional.
- Das 60482-Grammatik-Design der Kurzfassung (Pythia-410M, Heads L13H7/L15H7/L15H2)
  wird **nicht** vermischt; Pythia-410M läuft nur als Skalierungsarm derselben
  Sprachwechselaufgabe (SAE: `EleutherAI/sae-pythia-410m-65k`, nur MLP-Hookpoints).

---

## 6. Kontrollen und Konfundierungen

- Jede Textvariante hat ihren eigenen unveränderten Ausgangslauf (p₀).
- Richtungen: Vorzeichenumkehr, 32 isotrope Zufallsrichtungen, `freq`, `lang_probe`.
- Positionen: zufällige Positionsmengen gleicher Größe (Patching), zufällige
  Injektionspositionen (Störungsausbreitung).
- Texte: Formulierung A/B, Entwicklungs- vs. Testfamilien, einsprachige Umbaukontrolle
  für C2/C3 (gleiche Segmentierung, alles Englisch).
- Numerik: FP32 Standard; separater BF16-Lauf mit eigener Null-Dosis-Baseline;
  Soll/Ist-Prüfung des geschriebenen Vektors (`lsne.model.Injector.audit`).
- Längen: Tokens pro Segment und Gesamtlänge werden immer gespeichert; Analysen mit
  und ohne Längenkovariate.

---

## 7. Modelle, Rechenplan, Shards

| Arm | Modell | Revisionen | Zweck |
|---|---|---|---|
| Kern | pythia-160m | step98000, step99000 | Replikation, Grid, Nicht-Gleichgewicht |
| SAE | pythia-160m | step143000 | Feature-Steering, Vergleich mit Achsen |
| Verlauf | pythia-160m | 27 Dossier-Checkpoints (4k…143k) | Suszeptibilität über Training |
| Skala | pythia-410m, pythia-1b, pythia-1.4b | step143000 | Skalierung der Signaturen |

Rechenaufwand (Schätzung, A100): ein Forward von 160M auf ≤ 320 Tokens ≈ 3 ms.
Stage C (Grid, 6 Arme) ≈ 3 000 Forwards; Stage D1/D2 ≈ 50 000 Forwards (≈ 5 min);
Generierungsstages (D3–D7) ≈ 300 000 Token-Schritte gebatcht (≈ 30–60 min);
Checkpoint-Sweep ×27; Skalierungsarm ×(3…10). Alles passt auf 6 A100 in Stunden.
Shards: `jobs/shards.yaml` teilt nach (Modell, Revision, Familien-Teilmenge); jede
Shard schreibt eigene Ergebnisdateien, `lsne.cli merge` fügt zusammen.

---

## 8. Gestufter Plan (Dossier-Stufen A–F, hier konkretisiert)

| Stufe | Inhalt | Notebook | Abbruch-/Entscheidungskriterium |
|---|---|---|---|
| A Stimuli | Familien, Masken, Tokenisierung, Ziel-Leaks, Annotation | `00_setup_and_anchor` | Variante wird ausgeschlossen, wenn Ziel-ID oder Tail-IDs abweichen |
| B Pilot/Richtungen | Baselines, `ent`, `lang_dom`, SAE, Geometrie | `01_baseline_and_directions` | Dosis & Layerraster werden nur auf Entwicklungsfamilien gewählt |
| C Grid + Counterfactual | T, B, I über K × C; Positions-Patching | `02_switching_grid` | Primärkontrast T(K) auf Testfamilien, Familien-Bootstrap |
| D Nicht-Gleichgewicht | D1–D9, Phasendiagramme | `03_nonequilibrium` | Primärer Endpunkt λ (Präregistrierung) |
| G Geometrie + Trajektorie | G0–G3 (lokale Geometrie unter Sprachwechsel), T1–T5 (Trajektorie) | `06_geometry_trajectory` | G0 muss bestehen, sonst keine Interpretation; T3 entscheidet über die Steering-Lesart |
| E Tensornetz | lineare Antwort, TNCO vs. opt_einsum, Nichtlinearität | `04_tn_linear_response` | Berichtet Kosten/Laufzeit/Fehler; kein Hypothesentest |
| F Skala & Verlauf | 410M/1B/1.4B, Checkpoint-Sweep; BF16-Vergleich | `05_scale_and_checkpoints` | Shards, Drive-Export |

Stufe F (Lernen/Replay) des Dossiers wird hier nicht aufgesetzt; sie setzt eine
präzise Lernhypothese voraus, die erst aus C/D folgen kann.

---

## 9. Was dieses Design nicht behauptet

- Eine Sprachachse ist eine lineare Messkonstruktion, keine „innere Sprache“.
- Nicht-Gleichgewichts-Signaturen sind Analogien aus der Gitterphysik; sie werden als
  operationalisierte, vorab definierte Kenngrößen mit Nullmodellen geprüft, nicht als
  physikalische Identität behauptet.
- Ein Effekt auf Entwicklungsfamilien ist keine Bestätigung. Familien sind die
  statistische Einheit; Checkpoints derselben Trainingsbahn sind keine Replikationen.
- Die Literaturstellen „Axis of Ignorance/Uncertainty“ und „The Geometry of Pain“
  konnten unter diesen Titeln nicht identifiziert werden; verwendet werden die
  operationalisierbaren Kerne (Unterdrückung hochfrequenter Tokens bei Unsicherheit →
  `freq`-Achse und P_freq-vs-Entropie-Messung; lineare Achse per Difference-of-Means →
  Tagliabue et al. 2026). Falls andere Arbeiten gemeint sind, werden die Achsen
  entsprechend ersetzt; die Messgrößen bleiben.
