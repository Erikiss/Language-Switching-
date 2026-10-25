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

### 4.6 λ-Dynamik: Eingriffe zwischen Trennschärfe und Mimikry (Biggio et al.)

Biggio et al. (2013/2017, *Evasion attacks against machine learning at test time*,
arXiv:1708.06131) beschreiben den Gradientenabstieg gegen eine Diskriminante g(x) mit
einem Dichte-Term („Mimikry“), gewichtet durch λ:

F(x) = ĝ(x) − (λ/n) Σ_{i: y_i = Zielklasse} k((x − x_i)/h),  s. t. d(x, x⁰) ≤ d_max  (Alg. 1).

λ = 0: der Pfad überquert die Entscheidungsgrenze schnellstmöglich, endet aber in
Regionen ohne Datenstützung (p(x) ≈ 0); das Ergebnis ist „trennscharf“, aber unecht.
λ > 0: der Pfad wird langsamer und länger, endet aber in der dicht besetzten Region der
Zielklasse. Neuronale Netze haben flache Regionen (∇g ≈ 0), in denen reiner Abstieg stehen
bleibt; dort ist der Mimikry-Term entscheidend. Eine engere **Umschließung** (enclosure)
der legitimen Klasse zwingt zur Mimikry — das ist dieselbe Geometrie wie die Kontraktion
in §4.4: enge lokale Nachbarschaften erzwingen hohes λ.

**Übertragung auf dieses Projekt.** Alle bisherigen Eingriffe (Dossier: ein Schritt der
Größe ε entlang ∂H/∂h; hier: Sprachachse, SAE-Feature) sind λ = 0-Eingriffe mit einem
einzigen Schritt. Die λ-Dynamik verallgemeinert sie zu einem Pfad im Aktivierungsraum
(Modul `lsne.lambda_dynamics`):

| Element | Umsetzung |
|---|---|
| Punkt x | Residualvektor h an (Layer, Position) der Eingriffsstelle |
| Diskriminante ĝ | (a) lineare Sprach-Probe E↔D (scharf, SVM-Analogon); (b) −log p(Ziel) bzw. −H der nächsten Tokenverteilung (Modell-eigen, mit flachen Regionen); (c) −Aktivierung eines SAE-Features |
| Zielklassen-Dichte p̂ | KDE über **echte** Aktivierungen der Zielbedingung an derselben Stelle (z. B. Zustände der K>0-Varianten oder deutscher Segmente); Metrik ℓ₂ oder Mahalanobis (gewhitet mit der Layer-Kovarianz aus §4.4) |
| λ-Sweep | λ_rel ∈ {0, 0.1, 0.3, 1, 3, 10, 30}, skaliert so, dass λ/(n·h) mit dem Wertebereich von ĝ vergleichbar ist (Paper §3.2) |
| Pfad | Alg. 1: Einheitsschritt entlang ∇F, Projektion auf d(h, h⁰) ≤ d_max, Abbruch bei ΔF < ε |

Pro λ wird der Pfad aufgezeichnet und an jedem Schritt gemessen:
1. **Monosemantische Repräsentation:** TopK-SAE-Features (aktive Indizes, Aktivierungen)
   und SAE-Rekonstruktionsfehler (FVU) als Indikator für „unbesetzte Region“.
2. **Geometrie des Pfads** (hier die operationalisierte „Geometrie der Kette“, da
   Pythia-Basismodelle keine explizite Chain-of-Thought erzeugen): Pfadlänge,
   Nettoverschiebung, Tortuosität, mittlerer Drehwinkel, Schritte bis zur Grenzüberquerung
   (g < 0), Mahalanobis-Abstand zur Zielwolke, k-NN-Abstand zu echten Zuständen,
   lokales Log-Volumen (§4.4) am Endpunkt; zusätzlich die Tiefen-Trajektorie des Endpunkts
   durch die restlichen Layer (Lens-KL, Störungsausbreitung D1).
3. **Wirkung:** log p(Ziel), Entropie, p_freq nach Einsetzen des Endpunkts; für State 2143
   damit B(x) als Funktion von λ — bleibt der 98k/99k-Nutzen bestehen, wenn der Eingriff
   auf die Datenmannigfaltigkeit gezwungen wird?

**Was gesucht wird:** diejenigen SAE-Features und diejenigen Pfadgrößen, deren Werte am
Endpunkt monoton mit λ korrelieren (Spearman über den Sweep, Familien-Bootstrap), getrennt
für die drei Diskriminanten. Erwartung aus dem Paper: Bei (b) stehen λ = 0-Pfade in
flachen Regionen still (Schritte bis Grenzüberquerung → ∞), bei λ > 0 nicht; Features,
die nur auf λ > 0-Pfaden aktiv werden, sind Kandidaten für die „echte“ Repräsentation
eines Sprachwechselzustands, Features nur auf λ = 0-Pfaden Kandidaten für Artefakte.
Sprachwechsel bleibt das Mittel: Die Zielklassen-Dichte stammt aus echten
Sprachwechselzuständen, und K, C modulieren deren Umschließung (§4.4), also das λ, das
nötig ist, um sie zu erreichen.

### 4.7 Wohin steuern? Domänen-Kompass, Suszeptibilitätsatlas, schlechtester Head

Präzisierung des Auftraggebers (25.10.2026, vierte Rückmeldung): Die Atlas-Abfrage
(Schlüsselwörter einzelner Tokens im Pythia-Suszeptibilitätsatlas) hatte State 2143 in die
Nähe von **Jura** (Recht) gelegt. Soll der Kontext in Richtung *mehr* Sprachwechsel
bewegt werden, muss die Richtung **weg von Jura und hin zu Kultur/Politik** zeigen,
gegebenenfalls erst gegensteuern (Jura-Anteil entfernen), dann steuern. Der
ChronoCulture-Benchmark (Infografik, 10 360 Zeitpaare aus 72 Ländern, WVS/ISSP) zeigt,
dass Sprachmodelle genau dort am schwächsten sind: kulturelle Dynamik in Politik und
Wirtschaft (Demokratie 17 %, Ungleichheit 20 % Genauigkeit), stabile Einstellungen
(Familie, Wohlbefinden) deutlich besser (≈ 40 %), nicht-westliche Regionen doppelt so
schlecht. Paradigma: Modellqualität verbessert man am schlechtesten Head. Wo das Modell am
schwächsten ist, hat lineares Steuern die höchste Treffsicherheit.

Umsetzung (Module `lsne.atlas`, `lsne.heads`; Daten `data/lexicon/domains.json`):

| Baustein | Umsetzung |
|---|---|
| Domänen-Lexika | Schlüsselwortbänke E/D für `law` (Jura), `politics`, `economy`, `culture_values`, `family`, `religion`, `wellbeing`, `science`, `everyday`; nur Ein-Token-Wörter (mit führendem Leerzeichen) werden für Projektionen verwendet |
| Suszeptibilitäts-Abfrage (lokaler Proxy) | Für einen Zustand h oder eine Richtung d: beförderte/unterdrückte Tokens unter Logit-Lens (h + ε·d vs. h) und Unembedding-Projektion der SAE-Decoderzeilen; Aggregation zu einem **Domänenprofil** (Masse pro Domäne, z-standardisiert gegen Zufallsrichtungen) |
| Atlas-Import | Loader für die gespeicherte Atlas-Extraktion des Auftraggebers (D7, `data/atlas/*.txt`): Cluster-Schlüsselwörter werden eingelesen und mit demselben Domänenprofil verglichen; ohne Datei läuft nur der Proxy |
| Domänenachsen | `dom_law`, `dom_culture_politics` usw.: Difference-of-Means der Residuen auf Domänensätzen (gleiche Satzschablonen, nur Domänenwörter variieren) pro Layer; **Gegensteuern** = Projektion des Jura-Anteils aus h, dann Schritt entlang `dom_culture_politics`; als λ-Dynamik (§4.6) mit Zielklassen-Dichte aus Kultur/Politik-Zuständen |
| Schlechtester Head | Head-weise Zerlegung der Attention-Ausgabe (W_O-Slices); pro Head: direkter Beitrag zum Ziel-Logit, Ablations-Effekt (Mean-Ablation) auf T (Sprachwechsel-Effekt) und auf B; „schlechtester Head“ = größter Verlust auf Kultur/Politik-Familien relativ zu stabilen Familien; Steering am Head-Ausgang statt am Residualstrom als eigener Arm |
| Stimuli | Neue Familien F13–F18: Demokratie, Ungleichheit, Werte/Geschlechterrollen, Religion/Migration (ChronoCulture-Schwachdomänen) sowie zwei Jura-Familien (Nähe zu State 2143) als Domänenkontrolle; bestehende Familien = Alltag/stabil |

Vorab definierte Vergleiche: (a) Domänenprofil der State-2143-Richtung `ent` und des
Zustands h₂₀₇ (erwartet laut Atlas: Jura-nah); (b) T(K) und B(x) für `dom_culture_politics`
vs. `dom_law` vs. `lang_dom` vs. Zufall, getrennt nach Familiendomäne; (c) Interaktion
Richtung × Domäne der Familie (ist Steuern in der Schwachdomäne treffsicherer?);
(d) Head-Arm: Steering am schlechtesten Head vs. am Residualstrom bei gleicher Dosis.
Alles explorativ, bis die Bestätigung auf Testfamilien läuft; der Atlas bleibt die
Instanz für die Richtungswahl, wenn Befunde mehrdeutig sind.

### 4.8 RDX: Differenz statt Einzeldomäne, Gegenpol Proteinfehlfaltung, Sharing-Index, Test-Time-Training

Präzisierung des Auftraggebers (25.10.2026, fünfte Rückmeldung): Nach dem Prinzip der
**Representational Difference Explanation (RDX)** ist eine *Differenz* aussagekräftiger
als eine einzelne Domäne. Kultur/Politik allein genügt also nicht; es braucht einen
Gegenpol. Vorschlag: Pile enthält PubMed, PubMed enthält Proteinliteratur, und dort sind
**fehlgefaltete Proteine** (Prionen, Amyloid, β-Faltblätter) vermutlich das am schlechtesten
vorhersagbare Teilgebiet — strukturell analog zur Schwachstelle des Kultur-Benchmarks, aber
aus entgegengesetzter Richtung. Dritter Baustein: der **Sharing**-Parameter aus Metas
*Meaningful Social Interactions*, der die Netzwerkaktivität steigern sollte; da Pythia
keinen solchen Parameter hat, soll Test-Time-Training auf Ethereum-Daten (DAO-Fork 2016,
also eine Spaltung) als strukturelle Injektion dienen, unter Ausnutzung der Phasen, in
denen Textstücke sehr schnell memorisiert werden (wie State 60482 zeigte).

Umsetzung in drei Teilen, jeder mit einer eigenen Prüfung statt einer Annahme
(Module `lsne.rdx`, `lsne.pile_domains`, `lsne.sharing`, `lsne.ttt`):

**(a) RDX-Prinzip.** Nicht absolute Repräsentationen werden erklärt, sondern
Differenzvektoren zwischen kontrollierten Bedingungen. Kontrastpaare:
Kultur/Politik ↔ Proteinfehlfaltung (die beiden mutmaßlichen Schwachpole),
Jura ↔ Kultur/Politik (Dossier-Achse aus §4.7), Englisch ↔ Deutsch (Sprachachse),
K=0 ↔ K=8 (Sprachwechsel). Für jede Differenz Δ = mean(A) − mean(B) an jedem
(Layer, Readout) wird der **RDX-Anteil** berechnet: Welcher Bruchteil von ‖Δ‖² lässt sich
durch eine Kandidatenbasis erklären (Domänenachsen, SAE-Decoderzeilen, Head-Beiträge,
Hauptkomponenten)? Gegen Zufallsbasen gleicher Größe z-standardisiert. Eine Basis erklärt
eine Differenz nur dann, wenn sie sie besser erklärt als eine Zufallsbasis **und** die
Erklärung über Familien hinweg stabil ist.

**(b) Gegenpol Proteinfehlfaltung — gemessen, nicht angenommen.** Neue Lexikondomänen
`biomed_protein` und `protein_misfolding`. Die Prämisse „Fehlfaltung ist das am
schlechtesten vorhergesagte Teilgebiet“ wird empirisch geprüft: Mit dem vorhandenen
Range-Request-Sampler werden N Trainingssequenzen aus dem Pile gezogen, per Schlüsselwort-
profil einer Domäne zugeordnet (inklusive PubMed-Marker) und ihre Token-NLL unter
Pythia-160M gemessen. Ergebnis ist eine **empirische Schwierigkeitsrangfolge der
Pile-Teilgebiete**; sie entscheidet, ob Fehlfaltung tatsächlich der passende Gegenpol ist
oder ob ein anderes Teilgebiet (z. B. Formeln, Code, seltene Nomenklatur) die Rolle besser
erfüllt. Dazu Stimulusfamilien F19–F20 (Prion/Amyloid-Kontext, Laborerzählung) für den
Sprachwechseltest in dieser Domäne.

**(c) Sharing und Netzwerkaktivität — operationalisiert.** „Sharing“ ist bei Meta eine
Produktkennzahl, kein Modellparameter; eine direkte Übertragung wird hier nicht behauptet.
Das modellinterne Analogon ist **repräsentationales Teilen**: Wie stark nutzen
verschiedene Domänen dieselben Einheiten? Messgrößen pro (Layer, Site):
Sharing-Index S = Jaccard-Überlappung der aktiven TopK-SAE-Features zwischen
Domänenbedingungen (paarweise und global), Kosinus der Head-Beitragsprofile,
Anteil gemeinsam genutzter Features an der Gesamtaktivierung;
Netzwerkaktivität A = Anteil der Features bzw. Heads über Schwelle, mittlere
Aktivierungsmasse, Participation Ratio der Feature-Aktivierungen. S und A sind vor und
nach dem Eingriff messbar und bilden den „dritten Parameter“ des Vorschlags.

**(d) Test-Time-Training als strukturelle Injektion.** Fortgesetztes Training eines
eingefrorenen Checkpoints auf einem kleinen Kontrastkorpus (Ethereum/DAO-Fork 2016,
vom Auftraggeber bereitzustellen unter `data/ttt/`; der Code liefert einen Loader und einen
kleinen gemeinfreien Platzhalter, damit Tests ohne das Korpus laufen). Gemessen wird:
1. **Memorisationsgeschwindigkeit** pro Textstück: NLL-Verlauf über Updateschritte, mit
   den Kontrollen des Dossiers (Batchnachbarn, angepasste Kontrolltexte, Platzhalter);
   „schnelle Phase“ = Schritte bis zum Unterschreiten einer vorab definierten NLL-Schwelle.
   Geprüft über mehrere Checkpoints (4k, 98k, 143k), da das Dossier Phasenabhängigkeit
   nahelegt.
2. **Wirkung auf alles Übrige:** Sprachwechselkontraste T und B, lokale Geometrie (§4.4),
   Sharing-Index und Netzwerkaktivität (c), schlechtester Head (§4.7) — jeweils vor/nach
   TTT auf demselben Stimulusraster.
3. **Kontrollarme:** gleiches Trainingsbudget auf (i) einem thematisch neutralen Korpus
   gleicher Größe, (ii) einem Korpus ohne Spaltungs-/Fork-Struktur, (iii) permutiertem
   Text des DAO-Korpus. Nur ein Effekt, der über diesen Kontrollen liegt, stützt die
   „strukturelle Injektion“.

**Was hier nicht behauptet wird.** Dass eine Produktkennzahl eines sozialen Netzwerks einem
Modellparameter entspricht; dass ein Fork-Text im Modell eine Spaltung erzeugt; dass
Fehlfaltung ohne Messung der schwierigste Pile-Teil ist. TTT verändert Gewichte, daher
gelten alle Befunde am eingefrorenen Modell danach nicht automatisch weiter — jede Größe
wird nach dem Training neu gemessen.

### 4.9 Cross-Modell-Transplantation: BlockGPT → Pythia (KV-Cache und Residualstrom)

Präzisierung des Auftraggebers (25.10.2026, sechste Rückmeldung): Die Blockchain-Idee war
nicht als Textkorpus gemeint, sondern als **Quellmodell**: BlockGPT (Gai, Zhou, Qin, Song,
Gervais 2023, arXiv:2304.12749) ist auf Ethereum trainiert; dort soll die Eigenschaft
„Teilen“ extrahiert und in Pythia übertragen werden, entweder in den **KV-Cache**
(Einpflanzen) oder in den **Residualstrom**, dort nicht als Addition, sondern als
**Aktivierungs-Substitution** (Austausch), am besten beides, da beide Orte unterschiedlich
zu behandeln sind. Erwartung: starkes Signal, Transfer sollte gelingen.

**Fakten zu BlockGPT, die den Aufbau bestimmen.** Eingabe sind baumkodierte
EVM-Ausführungsspuren (Call-, State-, Log-Trace als „ITR“), eigene Tokenisierung mit
100k-Vokabular (93 233 Adressen, 6 759 Funktionssignaturen), 8 Transformer-Layer,
Kontext 512, Training per Causal-LM auf 68 M Transaktionen der Blöcke 5 470 817 bis
15 000 000 (**19.04.2018 bis 21.06.2022**). Der DAO-Fork 2016 liegt also außerhalb der
Trainingsdaten. Es gibt **keine offiziellen Gewichte**; eine inoffizielle Grob-
Reimplementierung (dm4sec/BlockGPT: ITR-Builder, Trainer, Detector) setzt einen
Ethereum-Archive-Node voraus. Konsequenz: Der Transfer wird als **modellagnostische
Maschinerie mit steckbarem Quelladapter** gebaut und sofort mit einem Stand-in-Quellmodell
verifiziert; ein echtes Spurenmodell (selbst trainiert über dm4sec/BlockGPT oder vom
Auftraggeber bereitgestellt) wird dann nur noch als Adapter eingehängt.

**Maschinerie (Modul `lsne.transplant`):**

| Schritt | Umsetzung |
|---|---|
| Quelladapter | `SourceModel`-Schnittstelle: `encode(item) → ids`, `residual(ids, layer) → [T, d_S]`, `d_model`, `n_layers`; Adapter für HF-Causal-LMs (Stand-in: Pythia-70M/160M, Code-Modelle) und ein Trace-Adapter-Stub für BlockGPT-artige Modelle |
| Paarung | Alignment braucht **gepaarte** Eingaben: dieselbe Transaktion als Spur (Quelle) und als Textserialisierung (Pythia); beim Stand-in derselbe Text in beiden Modellen |
| Alignment | lineare Abbildung W: R^{d_S} → R^{d_T} per Ridge-Regression oder orthogonalem Procrustes nach PCA (Rang r); Gütemaß R² auf Hold-out-Paaren, Null = permutierte Paare; Layer-Zuordnung über die beste R²-Kombination (Layer-Matching) |
| Quellmerkmal „Teilen“ | im Quellmodell als Richtung oder Zustandsmenge: Difference-of-Means zwischen Eingaben mit hohem und niedrigem Teilen (bei Spuren: viele-zu-viele Transfers, viele unterschiedliche Gegenparteien, Fork-Ära vs. Normalbetrieb; beim Stand-in: ein bekanntes Merkmal wie die Sprachachse, damit der Transfer funktional prüfbar ist) oder als SAE-Feature des Quellmodells |
| Transfer | v_T = W v_S bzw. H_T = W H_S; Qualität: Kosinus zwischen transferierter Richtung und der nativ in Pythia bestimmten Richtung desselben Merkmals (wo vorhanden) |
| Injektion A: Residual-Substitution | `Injector(mode="replace")` an (Layer, Positionen) mit H_T; keine Addition; Soll/Ist-Audit |
| Injektion B: KV-Cache-Pflanzung | virtuelles Präfix aus n Platzhaltertokens, dessen Block-Eingänge auf **allen** Layern durch H_T ersetzt werden; die Keys/Values dieser Positionen entstehen damit aus den transplantierten Zuständen (inklusive Rotary an den Präfixpositionen), die echten Tokens sehen sie per Attention, ihr eigener Residualstrom bleibt unangetastet; Variante: nur ausgewählte Layer |
| Injektion C: beides | A auf den echten Positionen plus B im Präfix |
| Kontrollen | permutierte Paare im Alignment, Zufallsrichtungen gleicher Norm, Substitution mit Zuständen des Zielmodells selbst (Identitätskontrolle), virtuelles Präfix aus echten Zielmodell-Zuständen |

**Messungen:** T und B der Sprachwechselaufgabe, Sprachanteil in generiertem Text,
Sharing-Index und Netzwerkaktivität (§4.8c), Domänenprofil (§4.7), Störungsausbreitung
(§4.2) — jeweils für A, B, C gegen die Kontrollen. Funktionaler Transfertest beim
Stand-in: Die im Quellmodell bestimmte Sprachachse muss nach Transfer in Pythia den
Sprachanteil der Fortsetzungen in dieselbe Richtung verschieben wie die native Achse
(Vorzeichen und Mindestkorrelation), sonst ist die Maschinerie nicht validiert.

**Was hier nicht behauptet wird:** dass BlockGPT ein Merkmal „Teilen“ enthält, bevor es
im Quellmodell operationalisiert und gemessen wurde; dass ein 2018–2022-Modell den
Fork 2016 kennt; dass eine Transplantation ohne Alignment-Güte über der Permutationsnull
interpretierbar ist.

**Machbarkeitsergebnis (25.10.2026, `docs/BLOCKGPT_FEASIBILITY.md`):** Es existieren
keine BlockGPT-Gewichte und kein offizieller Code (von den BlockScan-Autoren unabhängig
bestätigt); die beiden inoffiziellen Repositories sind unvollständig und ohne Daten.
Empfohlener Weg: ein eigenes BlockGPT-artiges Causal-LM (8 Layer, d = 512, Kontext 512)
auf den frei verfügbaren Xatu-Parquet-Traces (ethPandaOps, CC BY 4.0, 2016 vollständig
abgedeckt, kein Node nötig) trainieren; „Teilen“-Labels aus den Transfer-Tabellen
(fan_out/fan_in pro Transaktion, DAO/WithdrawDAO-Beteiligung, Fork-Block 1 920 000)
ableiten und die Richtung im Quellmodell daraus bestimmen. Die Transplantationsmaschinerie
ist mit Stand-in-Quellmodellen validiert (20 Tests, adversarielle Prüfung, eine
Fix-Runde).

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
| H λ-Dynamik | Alg. 1 im Aktivierungsraum, λ-Sweep, SAE-Features und Pfadgeometrie vs. λ, B(x)(λ) | `07_lambda_dynamics` | Korrelationen mit λ auf Entwicklungsfamilien gefunden, auf Testfamilien bestätigt |
| I Kompass + Heads | Domänenprofile (Atlas-Proxy/-Import), Domänenachsen mit Gegensteuern, Head-Attribution, Steering am schlechtesten Head | `08_compass_heads` | Domänenprofil von `ent` dokumentiert; Richtung × Domäne-Interaktion auf Entwicklungsfamilien |
| J RDX + Sharing | Differenzerklärung gegen Zufallsbasen, empirische Schwierigkeitsrangfolge der Pile-Teilgebiete, Sharing-Index und Netzwerkaktivität | `09_rdx_sharing` | Gegenpol wird durch die NLL-Rangfolge bestimmt, nicht gesetzt |
| K Test-Time-Training | Kontrastkorpus-Training, Memorisationsgeschwindigkeit über Checkpoints, Vorher/Nachher auf allen Kenngrößen, drei Kontrollarme | `10_ttt_injection` | Effekt nur gültig, wenn er über allen drei Kontrollarmen liegt |
| L Transplantation | Quelladapter, Alignment mit Permutationsnull, Residual-Substitution, KV-Cache-Pflanzung, Stand-in-Validierung | `11_transplant` | Funktionaler Transfertest mit dem Stand-in muss bestehen, bevor ein Spurenmodell eingehängt wird |
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
