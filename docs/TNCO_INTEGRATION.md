# TNCO-Integration: das linearisierte Gitter als Tensornetzwerk

Was TNCO in diesem Projekt tut, was es nicht tut, und wie das Tensornetz aus dem
Transformer hergeleitet ist. Code: `lsne/tn_lattice.py`, Stufe E
(`lsne/experiments/stage_e_tn.py`), Notebook `04_tn_linear_response`. Grundlage: DESIGN §5
und die Kurzfassung „TNCO × Pythia-410M (UIP)“.

## 0. Einordnung in einem Satz

TNCO (Google Research; Abanin et al. 2025, *Nature*) ist ein Optimierer für die
**Kontraktionsreihenfolge** eines gegebenen Tensornetzwerks — es verkleinert die Kosten
einer Rechnung, die man ohnehin ausführen könnte. Es ist **kein**
Interpretierbarkeitsverfahren und erzeugt keine Näherung: welche Näherung gerechnet wird,
legt die Konstruktion des Netzes fest (Abschnitte 1–2), nicht TNCO.

## 1. Linearisierung des Transformers

GPT-NeoX (Pythia) verwendet parallele Residuen:

```
h_{l+1} = h_l + Attn_l(LN1_l(h_l)) + MLP_l(LN2_l(h_l))
Attn_l(x)_{t'} = Σ_h Σ_t A^h[t', t] (x_t W_V^{hT} + b_V^h) W_O^{hT} + b_O
MLP_l(x)_t = act(x_t W_in^T + b_in) W_out^T + b_out
```

Für eine kleine Störung δ_l des Residualstroms am Arbeitspunkt h_l (dem unveränderten
Lauf) gilt in erster Ordnung δ_{l+1} = J_l δ_l mit der Block-Jacobi-Matrix

```
J_l = I  +  Σ_h  A^h ⊗ (W_O^h W_V^h J_LN1)  +  diag_t( W_out diag(act'(z_t)) W_in J_LN2,t )
```

(Tensor-Notation: der erste Faktor wirkt auf die Positionsachse, der zweite auf die
d-dimensionale Residualachse). Drei bewusste Vereinfachungen machen daraus ein
faktorisiertes Netz:

1. **Eingefrorene Attention-Muster.** Die Softmax-Gewichte A^h werden aus dem
   unveränderten Lauf übernommen (`run_with_cache(..., need_attn=True)`); ihre erste
   Ableitung nach Query/Key wird ignoriert. Es bleibt nur der Wertpfad W_O^h W_V^h.
2. **Eingefrorene LayerNorm.** d LN(h) ≈ diag(γ)/σ; Zentrierung und die Projektion auf den
   normierten Zustand entfallen. Für die Heads wird σ über das Positionsfenster gemittelt
   (`sigma1_mean`), für das MLP positionsweise (`sigma2_t`). `ln_mode="exact"` liefert
   die volle LN-Jacobi-Matrix γ/σ (I − 11ᵀ/d − n nᵀ/d) zum Vergleich.
3. **GELU nur als Diagonale.** act'(z_t) ist die exakte elementweise Ableitung an der
   Vorsaktivierung z_t des Arbeitspunkts (per Autograd, `_activation_derivative`), d. h.
   das MLP ist eine positionsweise lineare Abbildung.

Softmax, LayerNorm und GELU lassen sich also nicht „wegkontrahieren“ — sie liefern am
Arbeitspunkt feste Zahlen (A^h, σ, act'(z)), die als Tensorfaktoren eingehen. Alles
jenseits erster Ordnung fehlt dem Netz per Konstruktion.

## 2. Faktor-Tensoren und Indexstruktur

Residualraum: orthonormale PCA-Basis P ∈ ℝ^{d×r} (`pca_basis`, Standard r = 64 aus
`cfg.tn_rank`), gewonnen aus den Block-Eingangs-Residuen der Stufe-E-Teilmenge an der
Injektionsstelle. Positionen: Fenster von T_w Tokens (`cfg.tn_window`, Standard 32) am
Ende des Kontexts; die Injektionsposition liegt im Fenster (dann ist die Einschränkung
durch Kausalität exakt, Positionen vor dem Fenster werden nie gestört).

Für jeden Block l im Bereich `layers = range(l0, L)` (l0 = Tiefen-Injektion `n_layers//3`)
entstehen zwei Tensoren mit einem **Branch-Index** b_l der Größe 2 + H
(0 = Skip, 1…H = Heads, H+1 = MLP):

| Tensor | Indizes | Inhalt |
|---|---|---|
| Quelle `v` | (t_{l0}, i_{l0}) | onehot(Injektionsposition − Fensterbeginn) ⊗ Pᵀ·Richtung |
| `G_l` | (b_l, t_{l+1}, t_l) | Skip: I_{T_w}; Head h: A^h auf das Fenster eingeschränkt (Zeilen = Query-Positionen); MLP: I_{T_w} |
| `K_l` | (b_l, t_l, i_{l+1}, i_l) | Skip: I_r; Head h: M^h = Pᵀ W_O^h W_V^h diag(γ₁)/σ₁ P (über t_l verteilt); MLP: M_t = Pᵀ W_out diag(act'(z_t)) W_in diag(γ₂)/σ₂,t P |

Die Abbildung eines Blocks ist Σ_{b_l, t_l} G_l[b, t', t] K_l[b, t, i', i]; `t_l` ist ein
**Hyper-Index**, der von G_l, K_l und G_{l−1} geteilt wird, `b_l` wird innerhalb des Blocks
summiert. Das Netz hat 1 + 2·|layers| Tensoren und die offenen Indizes (t_L, i_L) am
Ausgang des letzten Blocks. Speicher pro Block: (2 + H)·T_w·r² Gleitkommazahlen für K_l
(160M, H = 12, T_w = 32, r = 64: 14·32·4096·4 B ≈ 7,3 MB). Die Rücktransformation in den
vollen Residualraum ist y·Pᵀ; die Zahl `source_captured = ‖Pᵀ dir‖²` in `tn_summary`
sagt, welcher Anteil der Störung überhaupt im Unterraum liegt (Trunkierungsverlust).

Alle Näherungen stehen maschinenlesbar in `LatticeTN.meta["approximations"]`.

## 3. Was TNCO optimiert

`optimize_path(tn, backend="tnco")` übergibt das Netz als
`tnco.app.TensorNetwork([Tensor(inds, dims), ...], output_inds)` an
`tnco.app.Optimizer(method="sa")` (Simulated Annealing, `betas=(0, 1e4)`,
`n_steps = Limits.tn_n_steps = 500`, `n_runs = 4`, `fuse=0`, `decompose_hyper_inds=False`,
damit die Tensorliste unverändert bleibt und der zurückgegebene lineare Pfad (Liste von
Indexpaaren) direkt mit `opt_einsum.contract(expr, *arrays, optimize=path)` verwendbar
ist). Der beste der `n_runs` Pfade wird genommen. Gemeldet werden

- `cost_log10`: das Kostenmodell des jeweiligen Backends (TNCO zählt
  Multiplikations-Additionen pro paarweiser Kontraktion, opt_einsum FLOPs),
- `cost_log10_opt_einsum`: **jeder** Pfad noch einmal mit der opt_einsum-FLOP-Zählung
  bewertet, damit TNCO, `opt_einsum` (dp/greedy) und die Referenzen `reference_greedy`,
  `reference_dp` vergleichbar sind,
- `runtime_s` der Pfadsuche, `largest_intermediate_log10`, `fallback_reason`.

Bei `ImportError` oder jedem TNCO-Fehler fällt `optimize_path` auf opt_einsum zurück
(„dp“ für ≤ 12 Tensoren, sonst „greedy“); die Kontraktion selbst läuft immer über
opt_einsum (`contract`). `cost_reference` berechnet zusätzlich die greedy- und (bis 40
Tensoren) die dp-Referenzkosten. Für die Standardgröße (160M: 1 + 2·8 = 17 Tensoren,
Dimensionen 14 × 32 × 64) ist das Problem klein genug, dass die dp-Referenz und TNCO meist
denselben optimalen Pfad finden — der Mehrwert von TNCO liegt bei breiteren Netzen (volle
Breite d = 768 auf A100, längere Fenster, 410M mit H = 16 und 24 Blöcken), wo die
dp-Referenz entfällt und greedy deutlich schlechtere Pfade liefert. Stufe E berichtet
genau diesen Vergleich (`tn_paths`), ohne Hypothesentest.

## 4. Die drei Antwortebenen

Für jede Variante (K=0 und K=8/C1 der ersten `Limits.n_families_tn` Familien), jede
Richtung (`lang_dom`, `ent`) und jede Dosis ε ∈ `Limits.tn_doses` = (0,01, 0,1) wird derselbe
Dosisvektor δ = ε·RMS(h)·dir/RMS(dir) an (Block-Eingang l0, Position = letztes Token von
Einheit 9) angelegt und die Antwort am Ausgang des letzten Blocks auf dem Fenster
verglichen (`compare_responses`):

| Ebene | Berechnung | Enthält |
|---|---|---|
| (a) gemessen | `Injector` + Forward, h′ − h | alles (nichtlinear) |
| (b) exakt linear | `linear_response_jvp`: `torch.func.jvp` durch das Modell mit dem Injektions-Hook (Fallback: zentrale Differenzen mit Richardson-Extrapolation) | vollständige erste Ordnung inkl. Attention-Ableitungen und exakter LN |
| (c) faktorisiert | `build_factorized_lattice` → `optimize_path` → `contract` → ·Pᵀ | erste Ordnung mit eingefrorenen Mustern/LN im PCA-Unterraum |

Kenngrößen pro Fensterposition (`tn_compare`) und normgewichtet über die Positionen ab
der Injektion (`tn_summary`): `nonlinearity = ‖a − b‖/‖b‖` (D8: im linearen Regime ∝ ε,
also zwischen ε = 0,01 und 0,1 um einen Faktor ≈ 10 wachsend; ein großer Wert schon bei
kleinem ε ist die Nicht-Gleichgewichts-Signatur), `factorization_error = ‖b − c‖/‖b‖`
(Fehler der Faktorisierung; er hängt nicht von ε ab, weil beide Ebenen linear sind) und
die Kosinus `cos_measured_linear`, `cos_linear_factorized`. Zur Trennung der Fehlerquellen
kann man r = d setzen (keine Trunkierung) und `ln_mode="exact"` (keine LN-Näherung); was
dann bleibt, sind die eingefrorenen Attention-Muster.

## 5. Grenzen (Kurzfassung)

- **Softmax, LayerNorm, GELU** sind nicht Teil des Netzes, sondern am Arbeitspunkt
  eingefroren; das Netz ist eine Linearisierung erster Ordnung. Für Dosen ≥ 0,1 wird die
  Nichtlinearität in (a) sichtbar und das Netz beschreibt die Messung nicht mehr.
- **Attention-Muster eingefroren:** der Beitrag von δ über Query/Key (Umverteilung der
  Aufmerksamkeit) fehlt in (c), nicht in (b). Das ist in (b)−(c) enthalten und typisch der
  größte Term des Faktorisierungsfehlers.
- **Trunkierung auf r Hauptkomponenten** und **Fensterbegrenzung**: exakt für r = d und
  Fenster ab der Injektion; `tn_basis` zeigt die erklärte Varianz, `source_captured` den
  erfassten Anteil der Richtung.
- TNCO ändert an keiner dieser Grenzen etwas; es macht nur die Kontraktion des gegebenen
  Netzes billiger. Alle Aussagen über das Modell stammen aus den Ebenen (a) und (b).
- Das 60482-Grammatik-Design der Kurzfassung (Pythia-410M, Heads L13H7/L15H7/L15H2) wird
  nicht mit diesem Programm vermischt.

## 6. Aufbau eines 410M-Laufs (Skalierungsarm, Shard 5)

`EleutherAI/pythia-410m@step143000`: 24 Blöcke, d = 1024, 16 Heads. Besonderheiten:

1. **SAE nur an MLP-Hookpoints.** Die veröffentlichten SAEs
   (`EleutherAI/sae-pythia-410m-65k`) existieren nur für `layers.N.mlp`
   (`lsne.config.SAE_REPOS`). `HookSite(N, "mlp_output").sae_hookpoint == "layers.N.mlp"`;
   die Dossier-Stelle (Block-Eingang) hat dort keinen SAE, `direction_bank` lässt die
   SAE-Richtungen daher mit einer Warnung aus (`bank.meta["skipped"]["sae"]`), solange die
   Site ein Block-Eingang ist. Für Feature-Steering am 410M muss die Site auf einen
   MLP-Ausgang gelegt werden: eine Störung an `mlp_output` von Block N tritt im
   Residualstrom erst am Snapshot N + 1 auf (`_snapshot_index`), das Netz wird also mit
   `site_from = HookSite(N, "mlp_output")` und `layers = range(N + 1, L)` gebaut.
2. **Größe des Netzes.** Mit l0 = 24 // 3 = 8 sind es 16 Blöcke → 33 Tensoren; pro Block
   (2 + 16)·T_w·r² Zahlen: bei T_w = 32, r = 64 ≈ 9,4 MB, bei r = 128 ≈ 38 MB, bei voller
   Breite r = 1024 ≈ 2,4 GB pro Block — auf A100 (40 GB) bleibt r ≤ 256 praktikabel
   (`--tn-rank 128 --tn-window 32` empfohlen). Mit 33 Tensoren arbeitet der
   opt_einsum-Backend nur noch mit greedy (dp nur bis 12 Tensoren); die dp-Referenz wird
   bis 40 Tensoren noch berechnet, bei längeren Fenstern oder mehr Blöcken nicht mehr —
   hier trägt TNCO.
3. **Befehl** (identisch zu `jobs/shards.yaml`, Shard 5):
   ```bash
   python -m lsne.cli run --stage A,B,C,D,E --model EleutherAI/pythia-410m --revision step143000 \
       --dtype float32 --device auto --out /content/drive/MyDrive/LSNE/runs --seed 0 --tn-rank 128 --tn-window 32
   ```
   Im Notebook 04: `MODEL="EleutherAI/pythia-410m"`, `REVISION="step143000"`,
   `TN_RANK=128`, `TN_WINDOW=32`. Die Injektionsstelle wird automatisch aus der Tiefe
   abgeleitet (`C.depth_layer(mb) = n_layers // 3`), nichts ist auf 12 Blöcke oder d = 768
   festgeschrieben.
4. **Auswertung** wie beim 160M: `tn_summary` pro Variante/Richtung/Dosis,
   `tn_paths` mit TNCO vs. opt_einsum (greedy), `tn_basis`. Erwartung: der
   Faktorisierungsfehler wächst mit der Tiefe des Netzes (mehr eingefrorene
   Attention-Muster zwischen Injektion und Ausgang); die Nichtlinearität (a)−(b) ist die
   Messgröße D8 der Nicht-Gleichgewichtsfrage und wird wie beim 160M über (K, ε) berichtet.
