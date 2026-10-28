# Ergebnisse des ersten Colab-Laufs (Lauf 1, pythia-160m@step98000)

Stand 27.10.2026. Zusammenfassung der Prüfung aller Colab-Läufe der ersten Runde auf
`EleutherAI/pythia-160m@step98000` (FP32, A100), dazu die SAE-Läufe der Stufen H und J bei
`step143000` und der Zwischenstand des noch laufenden Checkpoint-Sweeps (Notebook 05).

Grundlage: Jede Stufengruppe (AB, C, D, EF, GH, IJ, KL) wurde aus den Drive-Tabellen von einem
ersten Prüfer ausgewertet und von einem zweiten gegengeprüft. Wo beide voneinander abweichen,
gelten Zahlen und Urteil der Gegenprüfung; das ist mit „(Gegenprüfung)“ markiert. Die
SAE-Auswertung bei 143k kommt aus einer eigenen Analyse der Tabellen von `runs/H` und `runs/J`
(`@step143000`).

**Dieses Dokument ist keine konfirmatorische Auswertung.** Die Lauftabellen von Lauf 1 poolen
Entwicklungs- und Testfamilien und rechnen mit 2 000 statt 10 000 Bootstrap-Ziehungen; der
Primärendpunkt hat in Lauf 1 gar keine Tabelle, und keine Tabelle enthält die BH-Korrektur. Die
formale Auswertung nach PREREGISTRATION §2–4 macht der neue Auswerteschritt
`lsne/prereg_eval.py` (Notebook 13, nur CPU, `docs/RUNBOOK_COLAB.md` §11). Er ist noch nicht auf
den Drive-Daten gelaufen. Alles unten Genannte ist deshalb vorläufig.

**Lesehinweise.**

* Zahlen mit Dezimalpunkt wie in den Tabellen. `[a, b]` = 95-%-Intervall des
  Familien-Cluster-Bootstraps.
* „Test“ = Bestätigungsmenge F01–F12 × Formulierungen {A, B} (PREREGISTRATION §1). „DEV“ =
  DEV01–DEV04. „Gepoolt“ = alle 24 Familien (DEV01–04, F01–F20) × Formulierungen A–D, so wie
  die Lauftabellen gerechnet sind.
* Test- und DEV-Zahlen sind Nachrechnungen der Prüfung aus den Lauftabellen (Mittel der
  Familienmittel, `lsne.stats.cluster_bootstrap(stat_level="cluster")`, 10 000 Ziehungen,
  seed 0), sofern nicht anders vermerkt.
* Konto A = Hauptkonto (Stufen A, B, C, D, L, Notebook 12, H und J bei 143k). Konto B = zweites
  Konto (E, F, G, H, I, J, K bei 98k).
* Urteile: **gestützt** (supported), **nicht gestützt** (not_supported), **offen** (open),
  **nicht getestet** (not_tested), **nicht interpretierbar** (not_interpretable), **Gate
  verfehlt** (gate_failed), **Dokumentation** (kein Test).
* „Diese Runde“ = die Korrekturen nach der Prüfung. Sie sind mit Commit `97d4104` auf dem Branch
  `claude/language-switching-non-equilibrium-ejpz0q` (Basis `76a4a5a`) eingespielt und gepusht
  (§5.0).

> **Stand 28.10.2026:** Die präregistrierte Auswertung nach den Neuläufen (Notebook 13, inklusive
> Spurenmodell) steht in `docs/RESULTS_PREREG.md`. Diese Datei beschreibt Lauf 1.

## Kurzfassung

* **Gültigkeit.** Der Anker ist verifiziert, 0 von 1 152 Varianten wurden ausgeschlossen. Die
  Soll/Ist-Prüfung der geschriebenen Vektoren (§4) ist für C und D noch offen, weil die Tabellen
  für die Prüfung zu groß waren. Notebook 13 rechnet sie aus den vollständigen Tabellen.
* **Primärendpunkt: formal nicht getestet.** Die vorläufigen C1-Zahlen zeigen in Richtung H1
  (Δλ(`lang_dom`, K=8, ε=0.05) = 0.0987, Differenz gegen `rand` 0.1083). Sie sind aber gepoolt
  und ohne Familien-Bootstrap gerechnet. Die korrigierte konservative Untergrenze (0.0457) liegt
  unter dem Mindesteffekt 0.05. Außerhalb von C1 ist der Effekt nicht `lang_dom`-spezifisch, und
  die Monotonie in K gilt nur für die Punktschätzungen. Im Checkpoint-Sweep ist Δλ ≥ 0.05 an
  20 von 20 Checkpoints (ε 0.01–0.1), ebenfalls gepoolt und ohne Intervall.
* **Robuste Sekundärbefunde** (vorbehaltlich der noch fehlenden BH-Korrektur):
  T(8, C1) = -0.3438, das heißt, Sprachwechsel senkt log p(Ziel). Die Hysteresefläche von
  `lang_dom` beträgt 0.2935. ρ(λ, Mahalanobis) = -0.5208, wobei das überwiegend den Unterschied
  zwischen λ = 0 und λ > 0 misst.
* **Nicht gestützt:** Transfer von `ent` (das Vorzeichen kehrt sich gegenüber State 2143 um) und
  Richtung × Domäne (Endpunkt 13).
* **G0-Gate verfehlt**, deshalb werden 8 und 9 nicht interpretiert.
* **Nicht in präregistrierter Form gelaufen:** Endpunkt 15 (Regel 16 ersetzt den Gegenpol durch
  `books`, das wurde nicht umgesetzt), 17 (Platzhalterkorpus), 18 (Stand-in im alten Design) und
  18T (Spurenmodell aus einer gescheiterten Pipeline).
* **SAE bei 143k (explorativ):** ρ(λ, SAE-FVU) = -0.8393 auf Test. Der Effekt kommt überwiegend
  vom λ=0-Pfad. Das FVU-Zertifikat wurde nicht gespeichert, und die `sae_feature`-Diskriminante
  ist degeneriert.

---

## 1. Was gelaufen ist

Alle Läufe auf `pythia-160m`, FP32, Ausgabe unter `MyDrive/LSNE/runs/<stufe>/<modell>@<rev>/`
des jeweiligen Kontos.

| Notebook | Stufe | Konto | Revision | Status | Laufzeit | Git | Auffälligkeiten |
|---|---|---|---|---|---|---|---|
| 00_setup_and_anchor | A | A | step98000 | ok | 3.9 s | a018239 | keine; Anker verifiziert; 1 152 Varianten + 4 Originalvarianten. Andere SHA als B, aber `stimuli.py` ist zwischen beiden gleich, und eine CPU-Nachrechnung bei 76a4a5a reproduziert `variants.csv` exakt |
| 01_baseline_and_directions | B | A | step98000 | ok | 58.9 s | 76a4a5a | `lang_dom`-Entrauschungs-Gate verfehlt (cos 0.591 < 0.7), die Klartextachse wird nicht gespeichert; lief erst nach C (Ordner 21:15 UTC) |
| 02_switching_grid | C | A | step98000 | ok | 183 min | a018239 | baute eine eigene Richtungsbank, weil B noch fehlte; Unigramm-Quelle der `freq`-Achse unbekannt; `contrasts_boot` gepoolt, Zeilenebene, n_boot 2 000; Audit nicht geprüft (`grid.parquet` 21.7 MB) |
| 03_nonequilibrium | D | A | step98000 | ok (2. Versuch) | 195.2 min | 76a4a5a | 1. Versuch CUDA-OOM, das veraltete `error.txt` liegt noch im Ordner; keine Tabelle für den Primärendpunkt; Dosen nur 0.01/0.05/0.1 (+0.2/0.5/1.0); Tiefen-`lang_dom` verfehlt das Gate (cos 0.126); Audit nicht geprüft (`damage.parquet` 25.3 MB) |
| 04_tn_linear_response | E | B | step98000 | ok | 1.26 min | 76a4a5a | nur DEV01/DEV02; Volle-Rang-Referenz übersprungen (8.46 GB), daher `truncation_error` überall NaN |
| 05_scale_and_checkpoints, Shard 4 | F (B + D, reduziert) | B | 27 Checkpoints | läuft: 20/27 fertig (Stand 09:06 UTC) | Ø 2 098.5 s je Checkpoint | 76a4a5a | lexikografische Reihenfolge, step98000/99000 kommen zuletzt; ETA etwa 13:10 UTC; keine Fehlerdateien |
| 06_geometry_trajectory | G | B | step98000 | ok | 37.0 min | a018239 | eigene Bank (kein B auf Konto B); T5-Kopplung leer, weil D auf Konto A liegt; G1-n-Matching unstratifiziert; T3-Layerwahl degeneriert |
| 07_lambda_dynamics | H | B | step98000 | ok | 6.83 min | 76a4a5a | ohne SAE (10(i) und Features leer); `sae_fvu`-Zeile fälschlich `sign_as_expected=False`; bei 10(ii) fallen 66 von 96 Starts heraus |
| 07_lambda_dynamics | H | A | step143000 | ok | 757 s | 76a4a5a | mit SAE (`EleutherAI/sae-pythia-160m-32k`, `layers.10`); n_boot 2 000; `sae_feature`-Diskriminante degeneriert; FVU-Zertifikat nicht gespeichert |
| 08_compass_heads | I | B | step98000 | ok | 8.76 min | 76a4a5a | Head-Arm hat rund 36× kleinere RMS als der Residualarm; Gegensteuer-Arm nicht dosisgleich; drei Dosen als präregistriert markiert; 1 152 Zeilen ohne Audit |
| 09_rdx_sharing | J | B | step98000 | ok, 1 Substufe gescheitert | 38.57 min (Pile 2 089.8 s) | 76a4a5a | `sharing`: kein SAE; `summary_09` zeigte trotzdem „ok“; Regel-16-Gegenpol `books` berechnet, aber nicht angewandt |
| 09_rdx_sharing | J | A | step143000 | ok | 2 335 s (Pile 2 077 s) | 76a4a5a | mit SAE, `sharing` läuft; Gegenpol weiterhin nicht angewandt |
| 10_ttt_injection | K | B | step98000 | ok | 5.41 min | 76a4a5a | alle vier Arme auf dem Platzhaltertext, weil der DAO-Korpus fehlt |
| 11_transplant | L (Stand-in `pythia-70m@step143000`) | A | step98000 | ok | 1.92 min | 76a4a5a | 3 Prompts aus einer Familie, ε = 1.0 sättigt, `v_src` passt nicht zur Quell-Layer der Abbildungen |
| 12_trace_source, Schritte 1–7 | Spuren-LM | A | – | teilweise gescheitert | 110.07 min | 76a4a5a | Serialisierung abgebrochen (558 von 2 135 Chunks fehlen), Split nie gelaufen, Vokabular aus 0 Transaktionen, Training mit Adress-OOV 0.967 |
| 12_trace_source, Schritt 8 | L (Spurenquelle) | A | step98000 | ok | 21.07 min | 76a4a5a | erbt alle Mängel der Pipeline; die Zusammenfassung meldete trotzdem `counts_for_prereg = True` |

Anmerkungen:

* **Reihenfolge und Konten.** Auf Konto A lief C (12:48–15:51 UTC) vor B (21:15 UTC). C baute
  deshalb mit dem älteren Code eine eigene Bank, während D die Bank von B nutzte. G lief auf
  Konto B, hatte keinen Zugriff auf D und baute ebenfalls eine eigene Bank. Notebook 13 liest
  beide Konten zusammen.
* **Zwei Code-Stände.** A, C und G liefen auf `a018239`, alles andere auf `76a4a5a`. Für A und G
  ist das unerheblich, weil der relevante Code gleich ist. Bei C ist offen, ob die `freq`-Achse
  aus Pile- oder aus Stimuluszählungen stammt (stiller Fallback vor `658f578`). Das betrifft die
  `freq`- und `I`-Zeilen, nicht die Endpunkte 1–3 und 6.
* **Stufe F** (Sweep, Notebook 05) wird von nichts in diesem Dokument und auch nicht von
  Notebook 13 berührt.

---

## 2. Ergebnisse nach Endpunkt

### 2.1 Primärendpunkt (PREREGISTRATION §2)

**H1:** λ(K>0, ε, `lang_dom`) − λ(K=0, ε, `lang_dom`) > 0, monoton in K und größer als derselbe
Kontrast mit `rand` (Mittel über 32 Richtungen). Statistik: Familienmittel der Differenz auf der
Testmenge, Cluster-Bootstrap mit 10 000 Ziehungen, Mindesteffekt Δλ ≥ 0.05.

**Formaler Status: nicht getestet.** Die Gründe:

* Stufe D schrieb in Lauf 1 keine Bootstrap-Tabelle.
* `damage_phase` mittelt über Varianten ohne Familienschlüssel. Die Tabelle poolt 24 Familien ×
  A–D und hat keine Audit-Spalte.
* Die Werte pro Familie stehen nur in `damage.parquet` (25.3 MB, 471 648 Zeilen). Das lag über
  der Download-Grenze der Prüfung, die deshalb auch das Soll/Ist-Audit (§4) nicht prüfen konnte.
* Von den präregistrierten Dosen liefen nur 0.01, 0.05 und 0.1 (0.001, 0.0025, 0.005 und 0.02
  fehlen), und keine Dosis wurde auf DEV eingefroren.

**Was Notebook 13 daraus macht.** `lsne.prereg_eval` liest `damage.parquet` direkt, ein
GPU-Neulauf von D ist dafür nicht nötig. Der Schritt arbeitet mit Tiefeninjektion, Zelle C1 für
K > 0 und K = 0 als Referenz, und rechnet je Familie dlam(K) für `lang_dom` und `rand` (Mittel
über die 32 Richtungen) sowie deren Differenz (DiD), alles auf der Testmenge mit 10 000
Ziehungen. Er prüft die Monotonie (`P.mono`) und das Audit (`audit.D1`; ein verfehltes Audit
macht P nicht interpretierbar). Ohne eingefrorene Dosis wird jede vorhandene präregistrierte
Dosis einzeln berichtet, mit Bonferroni-Hinweis. Die Zeilen C2/C3/seg sind Abweichungszeilen
(`P.dev.*`, nicht interpretierbar).

**Vorläufige C1-Zahlen** (aus `damage_phase`: Injektion `depth`, `kind=standard`, 96 Varianten je
Zelle = 24 Familien × A–D gepoolt, `rand` 3 072 Zeilen = 96 × 32 Richtungen; kein
Familien-Bootstrap):

| ε | λ(`lang_dom`) K0 / K2 / K4 / K8 | Δλ `lang_dom` K2 / K4 / K8 | Δλ `rand` K2 / K4 / K8 | DiD `lang_dom` − `rand` K2 / K4 / K8 |
|---|---|---|---|---|
| 0.05 | 0.0529 / 0.1382 / 0.1444 / 0.1516 | 0.0853 / 0.0915 / 0.0987 | -0.0137 / -0.0114 / -0.0096 | 0.0990 / 0.1029 / 0.1083 |
| 0.01 | – | K8: 0.0993 | – | K8: 0.1057 |
| 0.1 | – | K8: 0.0912 | – | K8: 0.1009 |
| 0.2 (explorativ) | – | K8: 0.0778 | – | K8: 0.0864 |
| 0.5 (explorativ) | – | K8: 0.0456 | – | K8: 0.0487 |
| 1.0 (explorativ) | – | K8: 0.0251 | – | K8: 0.0230 |

Vorbehalte (Gegenprüfung):

* **Kein gültiges Intervall.** Eine variantengepoolte Worst-Case-Schranke (SE = (sd_K +
  sd_0)/√24, korrigiert auf ddof = 1 und t(23)) ergibt für Δλ(`lang_dom`, K8, ε 0.05)
  [0.0457, 0.1517] und für die DiD [0.0120, 0.2047]. Jede korrigierte Untergrenze von
  Δλ(`lang_dom`) liegt unter dem Mindesteffekt 0.05. Unter der Intervallregel wäre das also nicht
  „gestützt“, selbst wenn die Schranke die präregistrierte Statistik wäre (sie ist es nicht).
* **Nur C1 trägt.** H1 nennt keine Stufe C. In C2 und C3 ist die DiD bei jeder präregistrierten
  Dosis negativ (ε 0.05, K2/K4/K8: C2 -0.0038/-0.0011/-0.0114, C3 -0.0149/-0.0196/-0.0146;
  ε 0.01: C2 -0.0163/-0.0164/-0.0267). Gewichtet man C1–C3 gleich, liegt die DiD bei 0.020–0.027
  und damit unter 0.05. Der Grund ist die Position (§4, Punkt 6): `rand`-λ springt bei ε 0.05 von
  0.0358 (K2, C1) auf 0.1713 (C2) und 0.2104 (C3).
* **Monotonie nur als Punktschätzung.** Die Schritte K2 → K4 → K8 betragen 0.006–0.009 und liegen
  weit innerhalb jedes Standardfehlers. Fast der ganze Effekt ist der Schritt K0 → K2, und genau
  dort wechselt die Sprache des Injektions-Tokens von Englisch zu Deutsch (§4, Punkt 7). Bei
  ε 0.5 ist λ nicht monoton (0.1333, 0.1336, 0.1328). (Korrektur 28.10.2026, Nachtrag 6.11: Das
  Token selbst ist bei K = 0 und K > 0 dasselbe „.“; verschieden sind die Sprache des Satzteils,
  den es abschließt, und der letzte Wechsel D → E direkt danach. Kontrolle: Notebook 14.)
* **Achse.** Die `lang_dom`-Achse an der Tiefenstelle L4 verfehlte ihr Entrauschungs-Gate
  (`cos_denoised_plain` 0.126, `within_var_removed` 0.989, d′ 0.26 → 2.66). Sie behält nur 12.6 %
  der rohen Trennung, und die rohe Achse ist nicht gespeichert.
* **λ und Schaden laufen bei hoher Dosis auseinander.** Der finale Tail-Schaden, Verhältnis K8C1
  zu K0, beträgt 1.64/1.61/1.49 bei ε 0.01/0.05/0.1, aber 0.98 bei ε 0.5 und 0.86 bei ε 1.0. Dort
  kommt das größere λ von einem kleineren frühen D, nicht von mehr Schaden.
* **Urteil der Prüfung:** nicht getestet, explorativ. Die Richtung passt in C1 zu H1, außerhalb
  von C1 ist der Effekt nicht `lang_dom`-spezifisch. Das Ergebnis darf nicht als „gestützt“
  berichtet werden. Die erste Prüfung hatte „supported“ vergeben, die Gegenprüfung hat das
  korrigiert.

**Vorschau aus dem Checkpoint-Sweep** (Stufe F, Konto B, Notebook 05, läuft noch; nach §5
explorativ). Werte aus `damage_phase` je Checkpoint, C1, Tiefeninjektion; Stand 20 von 27
Checkpoints:

* Δλ(`lang_dom`, K8 − K0) liegt bei ε 0.05 zwischen 0.0559 und 0.1106. Es ist ≥ 0.05 an 20/20
  Checkpoints bei ε 0.01, 0.05 und 0.1, an 19/20 bei 0.2, 4/20 bei 0.5 und 0/20 bei 1.0.
* Die DiD gegen `rand` ist > 0 an 20/20 Checkpoints für ε 0.01–0.5 und an 17/20 bei ε 1.0.
* Phasenmittel bei ε 0.05, früh (1k–3k) / mitte (10k–31k) / spät (125k–143k): Δλ
  0.0690 / 0.0718 / 0.0930, DiD 0.0545 / 0.0604 / 0.1095. Der Kontrast wächst vor allem, weil die
  Grundlinien schrumpfen: λ(K0, `lang_dom`) 0.0929 / 0.0599 / 0.0199, λ(K0, `rand`)
  0.0887 / 0.0516 / 0.0061 (19 Checkpoints, erste Prüfung). λ(K8, `lang_dom`) bleibt bei
  0.079–0.177.
* Spearman(log Schritt, DiD) = 0.723 (p = 3.1e-4). Die Checkpoints sind aber nicht unabhängig.
* Streng monoton in K ist der Kontrast an 9/20 Checkpoints (ε 0.05). Das Muster ist überall ein
  Sprung mit Plateau, z. B. bei 143k 0.0033 → 0.0841 → 0.0818 → 0.0791.
* Die Tiefeninjektion bewegt den Ziel-Logit kaum (|B| ≤ 0.0016 nats an jedem Checkpoint). Die
  λ-Unterschiede sind also kein Logit-Effekt.
* Vorbehalte: gepoolt über 24 Familien × A–D, kein Intervall, `rand` nur 8 Richtungen
  (reduzierter Modus), `lang_dom` wird je Checkpoint neu gefittet, und die Konfundierung K0 → K2
  besteht auch hier. step98000 und step99000 laufen zuletzt. Die D-Läufe des Sweeps laufen mit
  dem alten Code und haben deshalb kein `damage_boot`.

### 2.2 Gültigkeitsregeln (PREREGISTRATION §1, §4)

| Regel | Befund | Urteil |
|---|---|---|
| Anker (SHA256 Sequenz `bdd4bbdd…55d4`, Kontext `84c8f1b6…a750`, Ziel 22905) | in den Manifesten A und B `verified true`, Ziel decodiert zu „bey“; ebenso in H bei 98k und 143k | bestanden (Lauf gültig; kein Hypothesentest) |
| Ausschluss: Ziel-/Tail-ID, Zielwort im Kontext | 0 von 1 152 Varianten ausgeschlossen (`checks`: 1 152 Zeilen, kein Problemfeld) | bestanden |
| Ausschluss: `review.csv` (Bedeutungsabweichung) | nicht implementiert (kein Code liest `review.csv`); alle 960 Zeilen haben `meaning_ok = 1`, die Regel würde also nichts ausschließen | bestanden, Regel fehlt im Code |
| Soll/Ist des geschriebenen Vektors ≤ 1e-4 | C (`grid.parquet`) und D (`damage.parquet`) nicht geprüft, weil zu groß; I: geprüfte Zeilen max 3.49e-6, 1 152 Zeilen ohne Audit (Replace-Modus, gemittelter `rand`-Arm); Endpunkt 14 ohne Audit-Spalte | offen; Notebook 13 rechnet `audit.C` und `audit.D1` |

### 2.3 Sekundäre Endpunkte (PREREGISTRATION §3)

BH über die Gruppe ist in keinem Lauf berechnet (§2.4). n = Familien / Zeilen.

| # | Endpunkt | Statistik [95 %-KI] | n | Regel | Urteil (Prüfung) | Anmerkung |
|---|---|---|---|---|---|---|
| 1 | T(K=8, C1), Texteffekt | Test **-0.3438** [-0.5560, -0.1608], Bootstrap-p < 1e-4; Lauftabelle (gepoolt, 2 000) -0.2991 [-0.3916, -0.2087] | 12 / 24 | KI ohne 0 und BH; kein Mindesteffekt | **gestützt** (negativer Texteffekt) | 11 von 12 Familien negativ (F09 +0.0149); keine Dosis-Wirkung in K (§3.1) |
| 2 | K × C: T(8, C3) − T(8, C1) | Test **+0.0965** [0.0016, 0.2001], p = 0.045 (0.044–0.048 über Seeds) | 12 / 24 | wie 1 | **offen**, vorläufig: ob BH es trägt, hängt von der Gruppierung ab (Gegenprüfung) | DEV hat das umgekehrte Vorzeichen (-0.1263 [-0.2414, -0.0069]); gepoolt 0.0325 [-0.0316, 0.0971]; mit n_boot 2 000 und seed 2 enthält das KI 0 ([-0.0002, 0.1953]); 6 von 12 Testfamilien negativ |
| 3 | R_ws+punct+boundary − R_random, K0 → K8/C1 | Test L2 +0.0884 [-0.0851, 0.3785]; L5 -0.8365 [-2.7532, 0.2173]; L8 +0.5507 [-0.0599, 1.6684]; L11 +0.0045 [-0.1237, 0.1240] | 12 / 24 je Layer | Erwartung R_delim > R_random; BH | **offen** | jedes KI enthält 0; die Mittel werden von Paaren mit fast verschwindendem T getrieben (F12A: R_excess -22.38 bei L5) |
| 3 | dasselbe, K0 → K28/seg | Test L2 **+0.0514** [0.0132, 0.0915] (p zweiseitig 0.0066, einseitig 0.0033); L5 -0.0847 [-0.3375, 0.0557]; L8 +0.0206 [-0.0081, 0.0498]; L11 **-0.0564** [-0.1186, -0.0077] (p 0.013) | 12 / 24 je Layer | wie oben | **gemischt / empfindlich** (Gegenprüfung; erste Prüfung: offen) | L2 hält unter Gruppen-BH oder einseitig, L11 ist signifikant mit dem *falschen* Vorzeichen; der Rand liegt im Monte-Carlo-Fehler; `delim` enthält keine Leerraum-Position (§4, Punkt 10) |
| 4 | Hysteresefläche A(`lang_dom`) − A(Neustart-Null) | Test **0.2935** [0.2757, 0.3097]; gepoolt 0.2916 [0.2791, 0.3043]; p ≤ 1e-4 | 12 / 72 | KI ohne 0 und BH | **gestützt** | misst überwiegend autoregressives Kontextgedächtnis: der absteigende Ast konditioniert auf den selbst erzeugten deutschen Text; `ent` -0.0365 [-0.0458, -0.0281] |
| 5 | Nichtmonotonie D2, K8 vs K0, Auslesung `nmi_last` | C1 Test **0.0182** [0.0063, 0.0297]; Bootstrap-p 0.0022, t-Test über Familienmittel p 0.0149 | 12 / 24 | wie 1 | **offen** (Gegenprüfung; erste Prüfung: gestützt) | 9 von 12 Familien > 0; Auslesung und C-Stufe sind nicht präregistriert (eine von ≥ 8 Schnitten); Effekt +0.018 auf Grundlinie 0.130 |
| 5 | dasselbe, Auslesung `nmi_tail` | C1 Test -0.0054 [-0.0111, 0.0011] | 12 / 24 | wie 1 | **kein Nachweis** (not_supported; ohne Mindesteffekt kein Nullbefund) | entgegengesetztes Vorzeichen zu `nmi_last`; zusammen ein gemischtes Ergebnis |
| 6 | Transfer: B(`ent`) − B(`rand`), Vorzeichen +1 (State-2143-Richtung) | K=0, Test: ε 0.05 **-0.1943** [-0.2390, -0.1517]; ε 0.1 -0.4498 [-0.5508, -0.3546]; alle Varianten der Testfamilien (288 Zeilen), ε 0.05: -0.2343 [-0.2989, -0.1734]; p < 1e-4 bei jeder Dosis | 12 / 24 | einseitig in der State-2143-Richtung (so jetzt im Code) | **nicht gestützt** | das Vorzeichen kehrt sich um: bei State 2143 B(+`ent`, 0.05) = +1.345, in den Familien schadet +`ent` in 12 von 12; nur 1 von 96 K=0-Varianten gewinnt (DEV01B, logp0 -6.32); zweiseitig gelesen wäre „unterscheidet sich von Zufall“ gestützt |
| 7 | G0-Gate | 0 von 13 Snapshots, mittleres Drittel 0 von 5; nicht degeneriert (r = 47 < 191) | 24 Familien, 192 Zustände | Voraussetzung für 8–9 | **Gate verfehlt** | verfehlt wird nur d(A_E, A_D) < d(A_E, B_E): exakte Übersetzung 8.796–9.111 gegen gleichsprachige Paraphrase 3.398–4.947; die anderen Ungleichungen halten; gepoolt `mean_prefix` statt „Mean-Pool“ |
| 8 | Kontraktion G1 | wörtlich schließen alle KIs L1–L12 die 0 aus, z. B. L10 log-Vol -1.8375 [-1.9835, -1.6891], PR L11out -0.4754 [-0.5731, -0.3774] | 24 | zweiseitig; nur bei G0 | **nicht interpretierbar** (G0) | n-Matching war unstratifiziert, ein plausibler, aber nicht belegter Störfaktor (§3.2); in dieser Runde korrigiert |
| 9 | Steering-Validität T3, cos ≥ 0.3 | gespeichert 0.1766 [0.1492, 0.2058] (degenerierte Layerwahl); korrigiert (euklidisches Argmax) L11-Injektion 0.2035 [0.1756, 0.2322], L4-Injektion 0.1052 [0.0489, 0.1623] | 24 | Mindesteffekt 0.3; nur bei G0 | **nicht interpretierbar** (G0) | wörtlich bei jeder Layerwahl unter 0.3 (bestes post hoc 0.2048 / 0.2101), wäre also nicht gestützt |
| 10(i) | ρ(λ, SAE-FVU), Erwartung < 0 | kein SAE bei 98k | 0 | Vorzeichen wie erwartet | **nicht getestet** | bei 143k explorativ -0.8393 auf Test (§3.4); `sign_as_expected=False` in der Tabelle war ein Codefehler, in dieser Runde behoben |
| 10(ii) | ρ(λ, zensierte Schritte bis zur Grenze), zweiseitig | wie gelaufen 0.6424 [0.6234, 0.6645] (10 / 30 Starts); konstante Reihe als ρ = 0: Test **0.1865** [0.0510, 0.3299], gepoolt 0.2032 [0.0977, 0.3161] | Test 12 / 24 | zweiseitig, Erwartung offen | **signifikant positiv, Größe nicht interpretierbar** (Gegenprüfung; erste Prüfung: nicht interpretierbar) | 66 von 96 Starts erreichen die Grenze nie (log p0 > -1, Median dieser Starts -0.464), der Rest quert nur bei λ_rel 0 (24) oder 0 und 0.1 (6); misst „λ = 0 quert, λ > 0 nicht“ |
| 10(iii) | ρ(λ, Mahalanobis zur Zielwolke), Erwartung < 0 | Test **-0.5208** [-0.7024, -0.3318]; gepoolt -0.5539 [-0.6980, -0.3899] | 12 / 24 | Vorzeichen wie erwartet | **gestützt** | teils konstruktionsbedingt (λ zieht zur selben Wolke); λ-Sweep ab λ_rel 0.3 gesättigt; ohne λ = 0: -0.2863 [-0.5268, -0.0196]; reiner λ=0-Sprung ergäbe etwa -0.375 |
| 10(f) | Features mit \|ρ\| ≥ 0.7 auf DEV einfrieren, auf Test prüfen | kein SAE bei 98k; der Code hatte keinen Einfrierpfad | 0 | – | **nicht getestet** | Einfrierpfad in dieser Runde ergänzt (`lambda_features_frozen`); bei 143k nur gepoolt auswertbar (§3.4) |
| 11 | B(x)(λ) State 2143, λ_rel 3 vs λ 0 | B(λ0) 13.5959, B(λ3) 1.9439 nats; gleiches Vorzeichen; B(3) − B(0) = -11.652 | 1 Kontext | zweiseitig, Vorzeichenbericht | **Dokumentation** | nur 98k (99k fehlt); die Zielwolke besteht zu 768 von 771 Zuständen aus anderen Familien |
| 12 | Domänenkompass von `ent` an h₂₀₇ | E: oben `religion` z 21.124, `law` z 11.169, Rang 10/11; D: oben `family` z 25.040, `law` z 19.816, Rang 4/11 | 1 Zustand, 11 Domänen | Dokumentation | **Dokumentation**: die Atlas-Erwartung wird nicht reproduziert | nicht domänenselektiv: alle 11 E-Domänen z ≥ 10.03, Masse ×3.42–5.30; das deutsche Lexikon hat 1–10 IDs je Domäne (`law` 2) |
| 13 | B(`dom_culture_politics` nach Gegensteuern) − B(`lang_dom`), ≥ 0.05 nats | ε 0.01 -0.02978 [-0.08017, 0.00323]; ε 0.05 **-0.04088** [-0.07194, -0.00982]; ε 0.1 -0.05222 [-0.14190, 0.00218] | 4 / 16 | Familien-Bootstrap, Mindesteffekt 0.05 | **nicht gestützt** | die Obergrenze liegt bei jeder Dosis unter +0.05; nur 4 Cluster; der Gegensteuer-Arm ist nicht dosisgleich (unzentrierte Projektion auf `dom_law`) |
| 14 | Head-Arm vs Residualstrom, gleiche relative Dosis | `lang_dom` ε 0.1 +0.08835 [0.00256, 0.17606], ε 0.5 +3.18193 [2.40037, 3.94498]; `dom_culture_politics` ε 0.1 +0.10528 [-0.03772, 0.25534], ε 0.5 +1.88295 [1.08391, 2.78424] | 12 Bewertungsfamilien / 96 | zweiseitig, KI ohne 0 | **wörtlich gestützt in 3 von 4 Zellen, inhaltlich kein Beleg für Head-Spezifität** (Gegenprüfung; erste Prüfung: nicht interpretierbar) | die Residual-RMS ist im Median 35.76× größer als die des Head-Arms (B Head -0.0002 bis -0.0062), die Differenz misst also den Schaden des Residualarms; Head L11H10 (-0.04282) fast gleichauf mit L8H10 (-0.04151); Auswahl auf der Fallback-Hälfte, nicht auf DEV |
| 15 | RDX: Domänenachsen erklären ‖Δ‖² Kultur/Politik ↔ Misfolding, z ≥ 2 in ≥ der Hälfte der Layer | 4 von 4 Grid-Layern z ≥ 2 (z 9.346 / 41.113 / 19.358 / 32.473), z_mean 25.572, erklärt 0.0864 (Zufall etwa 0.003–0.004) | 16 vs 8 Zustände, 4 Sites | Mindesteffekt z ≥ 2 | **in präregistrierter Form nicht getestet** (Gegenprüfung; erste Prüfung: gestützt) | Regel 16 hat gefeuert (Gegenpol `books`), gerechnet wurde aber gegen Misfolding, das Ergebnis ist also explorativ; Stabilität über Familienpaare 0.75 / 0.50 / 0.75 / 1.00; die isotrope Zufallsbasis ist ein nachsichtiges Null |
| 16 | Pile-Schwierigkeit, Rang von `protein_misfolding` | Rang 9 von 11, NLL 2.5429 [2.0712, 2.9642] (n = 9 Sequenzen); schwerstes Drittel: books 3.1748, news 2.9901, wikipedia 2.9851, german 2.9736 | 2 000 Sequenzen | Bericht + Ersetzungsregel | **Dokumentation**; Regel gefeuert, Gegenpol `books` | P(Rang ≤ 4) ≈ 0.05; Misfolding-Bucket klein und verrauscht; bei 143k gleiches Ergebnis |
| 17 | TTT (i) Memorisation, (ii) Δ Sharing/Aktivität, (iii) Δ T und Δ B | (i) Injektion − neutral = Injektion − nofork = 0.0 [0.0, 0.0]; Injektion − permutiert -2.625 [-2.9375, -2.3125]; (iii) Injektionsarm ΔT_K8 -0.1589, ΔB_lang_dom +0.0688, Δlogp_target -6.9482 | 16 Snippets; 96 / 192 | nur DEV explorativ, dann Test | (i) und (iii) **nicht interpretierbar**, (ii) **nicht getestet** | alle Arme auf demselben Platzhaltertext, also tautologisch; (ii) ohne SAE; TTT mit LR 1e-4 und 200 Schritten senkt log p(Ziel) um etwa 7 nats |
| 18 | Stand-in-Validierung (`pythia-70m` → `pythia-160m`) | bester Layer L5: R² Hold-out 0.5930 gegen permutiert -0.4889 (Abstand 1.0819); cos 0.9726 (Null max 0.2504, z 5.55); Vorzeichengleichheit für Residual und KV | 144 Paare, 3 Prompts | Abstand ≥ 0.2, cos ≥ 0.3, gleiches Vorzeichen | **wörtlich bestanden**; zählt nach dem Code dieser Runde **nicht mehr** | ε 1.0 sättigt fD (≈ 0.99); 3 Prompts aus einer Familie; n_null 10; `v_src` passt nicht zur Quell-Layer (cos_native 0.8130 gegen 0.9726); Neulauf von Notebook 11 nötig |
| 18T | Spurenquelle (BlockGPT auf Xatu 2016) | Code-Urteil „Effekt“ für kv: B gegen Zufall +0.0275 [0.0086, 0.0482], gegen permutiert +2.7646 [1.9841, 3.5889], aber gegen das unveränderte Modell **-0.0472** [-0.0919, -0.0074] | 24 Familien, 48 Prompts | Sonde, Alignment und Effekt, nach bestandenem Stand-in | **nicht interpretierbar** | Quellmodell aus gescheiterter Pipeline (§4, Punkt 20); die Sonde schlägt die Oberflächen-Baseline nicht (0.9666 < 0.99328); der Alignment-Gewinn gegenüber dem untrainierten Zwilling ist 0.0034 (KI-Untergrenze -0.0425); nach der strengeren Regel dieser Runde „no claim“ und `counts_for_prereg = False` |

### 2.4 Was die BH-Korrektur noch ändern kann

Keine Lauftabelle berechnet BH. Die vorläufigen p-Werte:

* Endpunkte 1, 4 und 6 haben p < 1e-4 und halten jede BH-Reihenfolge.
* Endpunkt 2 (p = 0.045) hält BH nur unter einer bestimmten Gruppierung: wenn Endpunkt 3 als ein
  einseitiger, Bonferroni-kombinierter Test eingeht und 4 und 5 je p ≤ 0.045 haben. Unter den
  anderen gerechneten Gruppierungen fällt er durch (Gegenprüfung).
* Endpunkt 5 (`nmi_last`): Mit 12 Clustern ist das Perzentil-Bootstrap-p antikonservativ (0.0022
  gegenüber 0.0149 im t-Test). Bei p 0.015 bräuchte er unter 18 Endpunkten mindestens Rang 6.
* Endpunkt 3/K28seg L2 ist zweiseitig p 0.0066 gegen die Schwelle 0.00625, das liegt im
  Monte-Carlo-Fehler.

Notebook 13 legt die Gruppe fest (1–6, 8 und 9 nur bei bestandenem G0, 10, 13, 14) und gibt mit
`bh_q_endpoint` (Bonferroni innerhalb, BH über die Endpunkte) eine Sensitivitätsgruppierung aus.
Tabellen von Lauf 1 ohne `p_boot` (I, G) bleiben außerhalb der Gruppe; ihr Urteil wird dort
höchstens „offen“.

---

## 3. Explorative Befunde

Nichts in diesem Abschnitt ist präregistriert (PREREGISTRATION §5). Positive Befunde sind
Hypothesen für einen späteren Bestätigungslauf.

### 3.1 Texteffekt, Entropie und Masse auf häufigen Tokens (B, C)

* **log p(Ziel) an der letzten Kontextposition** (96 Formulierungen je Zelle):
  * K0/C1: Mittel -0.8707, Median -0.6837, Top-1-Treffer 96.9 %.
  * K > 0: Mittel zwischen -1.1062 und -1.1757, Top-1 90.6–95.8 %.
* **Keine Dosis-Wirkung in K** (Testfamilien): T(2, C1) -0.2790 [-0.5111, -0.0761],
  T(4, C1) -0.2411 [-0.4008, -0.0999], T(8, C1) -0.3438, T(16, seg) -0.2218 [-0.3743, -0.0798],
  T(28, seg) -0.3487 [-0.5287, -0.1855]. Der Effekt folgt dem Vorhandensein deutschen Texts, nicht
  der Zahl der Wechsel.
* **Längen-Konfundierung:** K0-Kontexte haben 114.3 Tokens, K2/4/8 im Mittel 151.23–152.63.
* **Formulierungen:** In A/B ist der Texteffekt größer als in den später ergänzten C/D:
  T(8, C1) auf F01–F12 -0.3438 gegen -0.1907 [-0.3186, -0.0768].
* **Masse auf häufigen Tokens (p_freq) und Entropie:**
  * Im gemeinsamen englischen Schlussabschnitt (Wortanfänge) steigt die Entropie von 3.682 (K0)
    auf 4.148 (K8C1), +0.466 je Familie (24 von 24 > 0). p_freq sinkt dabei kaum, von 0.331 auf
    0.326 (Δ -0.0053). Der Schlussabschnitt wird also unsicherer, ohne Masse auf häufigen Tokens
    zu verlieren. Die Tail-NLL steigt um +0.191 (18 von 24 Familien > 0).
  * Außerhalb des Tails sinkt p_freq an Wortanfängen von 0.395 auf 0.243. Hier mischen sich
    englische und deutsche Tokens, und die Top-100 des Pile sind englisch dominiert, das ist also
    teils ein Effekt der Sprachzusammensetzung.
  * Sprachgrenzen bei C1 sind Positionen mit hoher Entropie und wenig Masse auf häufigen Tokens:
    p_freq 0.084 / 0.053 / 0.049 (K2/4/8), Entropie 4.87 / 4.95 / 5.23. Bei C2 liegt p_freq bei
    0.208 / 0.222 / 0.217, bei C3 bei 0.232 / 0.230 / 0.224 (Entropie 3.57 / 3.80 / 4.16).
  * Die NLL nicht-terminaler englischer Tokens steigt monoton mit K: 3.956, 4.427, 4.588, 4.845,
    4.955, 5.303 für K 0/2/4/8/16/28.
  * Bei 143k steigt entlang der `target_logit`-Pfade mit λ auch p_freq am Endpunkt
    (ρ +0.918 [+0.829, +0.988]; §3.4).
* **Unembedding der Bank:**
  * `lang_dom` ist eine Achse deutscher Funktionswörter (`' dieser'` 27.53, `' wird'` 25.21, …).
  * `ent` hebt seltene Subwörter und drückt häufige Tokens (`'_'` -39.75, das Top-1 des Modells am
    2143-Kontext). Deshalb ist cos(`ent`, `freq`) = -0.4386, etwa 12 Null-SD.
  * Das Auslesen von `lang_probe` ist nicht deutsch (`'lei'`, `'chal'`, …), und
    cos(`lang_dom`, `lang_probe`) beträgt nur 0.1131.
* **Wechselwirkung I** (gepoolt, 2 000 Ziehungen): Gemischte Varianten profitieren stärker von
  −`ent` und `lang_dom`, Zufallsrichtungen zeigen nichts. I(`ent`, −1, 0.05) 0.0292
  [0.0156, 0.0446], I(`ent`, +1, 0.05) -0.0351 [-0.0540, -0.0192], I(`lang_dom`, −1, 0.05) 0.0170
  [0.0102, 0.0245], I(`rand`, −1, 0.05) 0.0005 [-0.0009, 0.0020].
* **State 2143:** log p(„bey“) bei K0 -13.5959 (Rang 6712, Top-1 `'_'`); T(K2) -0.635, T(K4)
  -0.591, T(K5) +0.054. Wechsel bewegen dieses sehr unwahrscheinliche Ziel kaum. B(`ent`, +1, 0.05)
  = 1.345 bei K0.

### 3.2 Geometrie und Trajektorie (G)

* **G0 im Detail.** In präfixgemittelten Zuständen dominiert die Sprache die Bedeutung. Eine
  exakte Übersetzung ist etwa doppelt so weit entfernt wie eine gleichsprachige Paraphrase und nur
  knapp näher als ein Paar aus einer fremden Familie. Fremdfamilien-Abstände (≈ 9.15) liegen auf
  dem Niveau zufälliger Paare (√84 = 9.17).
* **T1, Tiefe.** Das Log-Volumen steigt mit der Tiefe monoton, es gibt kein „Atmen“:
  `breathing_index` ist 0 für jede gemischte Bedingung und 0.167 für K0, das sind 2 von 24
  Familien mit 2 Vorzeichenwechseln. Die alte Spalte `layer_of_max_contraction` zeigte das
  kleinste positive Inkrement, ohne dass irgendetwas kontrahierte (in dieser Runde korrigiert).
* **T2, Position.** Das Log-Volumen schrumpft entlang des Tails zum Ziel hin: je Familie K0
  -3.1 → -9.9 (24/24), K8/C1 -39.8 → -82.8 (23/24), K2/C1 -31.0 → -77.5 (23/24). Das sind rohe
  Werte mit n = 4 gegen n = 32, vergleichbar ist nur die Form.
* **G1 nach K.** Das Log-Volumen hat bei L10 einen K-Gradienten (K2/C1 -0.95, K4/C1 -1.19, K8/C1
  -1.84), während ΔPR über K und C flach ist (-0.36 bis -0.55). Das unstratifizierte n-Matching
  (nur 11.4 % der Ziehungen enthalten jede Formulierung einmal) ist ein plausibler Störfaktor. Aus
  den gespeicherten Tabellen lässt er sich aber nicht nachweisen, und den K-Gradienten erklärt er
  allein nicht.
* **T3.** Steering an der Dossier-Stelle richtet sich klar besser als Zufall nach der diskreten
  Verschiebung aus (cos_excess 0.1884 [0.1589, 0.2181]), bleibt aber unter 0.3. Die L4-Achse
  steht an ihrem eigenen Layer fast senkrecht dazu (-0.0052 [-0.0252, 0.0142]). Nur die nicht
  präregistrierte Dosis ε 0.5 an L4 erreicht 0.3248 [0.2818, 0.3735].
* **T5, Kopplung an λ_ℓ und Lens-KL:** leer, weil D auf dem anderen Konto lag.

### 3.3 λ-Dynamik bei 98k (H)

* **Erwartung umgekehrt.** λ = 0 und λ > 0 verhalten sich entgegengesetzt zur Erwartung in
  DESIGN §4.6 für die Diskriminante (b):
  * Anteil querender Pfade (`target_logit`): 0.312 (λ_rel 0), 0.062 (0.1), 0 (≥ 0.3).
  * Endpunkt-B: +0.871, -0.274, -1.191, -1.611, -1.738, -1.793, -1.808, mit ρ(B, λ) = -1.0000 in
    jedem Start.
  * Der Grund: Die K8/C1-Zielwolke hat ein niedrigeres p(Ziel) als die K0-Starts (mittleres
    log p -1.170 gegen -0.871, Median -0.970 gegen -0.684). Mimikry und die +1-nat-Grenze ziehen
    in entgegengesetzte Richtungen.
* **Sättigung.** Ab λ_rel 0.3 liefern 5 von 7 λ-Werten denselben Endpunkt:
  * Mediane Spannweite je Start über λ_rel ≥ 0.3: Mahalanobis 0.00086, gegen |λ0 − λ0.3| = 1.61.
  * Medianer Mahalanobis-Abstand 5.256 (λ 0), 3.650 (0.1), 3.648 (30).
  * Mögliche Ursache: Die Einheit von `lambda_scale` ist für diese KDE (Bandbreite 59.20,
    n_eff 50) zu groß.
* **Unkonvergierte Pfade.** 12–13 der 96 `target_logit`-Pfade enden bei jedem λ_rel > 0 auf
  `max_steps` (100 Schritte), sind also nicht konvergiert.
* **Andere Diskriminanten.** Die Querungshäufigkeit sinkt auch hier mit λ: `probe`
  ρ(crossed) -0.8144 [-0.8466, -0.7683], `entropy` -0.7716 [-0.8113, -0.7275].

### 3.4 SAE bei step143000 (H und J, Konto A)

Explorativ in doppeltem Sinn: Der Checkpoint ist nicht der präregistrierte (98k/99k, Endpunkt 11
und alle Bestätigungsläufe sind bei 98k), und SAE-Feature-Auswahl ist nach §5 explorativ. Die
143k-Zahlen füllen Endpunkt 10(i) bei 98k also **nicht**.

**Lauf H.**

* Status ok, 0 gescheiterte Substufen, Git `76a4a5a` (gleich dem aktuellen HEAD), 757 s, FP32.
* SAE `hub:EleutherAI/sae-pythia-160m-32k/layers.10` (= L11.block_input, TopK k = 32,
  32 768 Latents); `sae_feature` = 451 (`features_D[0]` der Bank).
* n_boot 2 000. 96 Starts: DEV01–04 und F01–F20 × A–D, DEV und Test gepoolt.
* n_target 768 (K8/C1), KDE-Bandbreite 56.76 (98k: 59.20). Anker verifiziert.

**Präregistrierte Zeilen** (Diskriminante `target_logit`; gespeichert = 2 000 Ziehungen, gepoolt;
die übrigen Spalten sind Nachrechnungen mit 10 000 Ziehungen auf Familienmittel-Ebene, seed 0):

| Größe | gespeichert | Test F01–F12 × {A, B} | DEV | ohne λ = 0 |
|---|---|---|---|---|
| `sae_fvu` (Erwartung < 0) | -0.6752 [-0.7894, -0.5521], 24 / 96 (10 000: [-0.7879, -0.5569]) | **-0.8393** [-0.9449, -0.7262], 12 / 24 | -0.5469 [-0.8460, -0.3393] | λ_rel ≥ 0.1: -0.5435 [-0.7006, -0.3750] (Test -0.7452 [-0.9143, -0.5619]); nur λ_rel ≥ 0.3: -0.4469 [-0.6417, -0.2365] |
| `steps_to_cross_censored` (zweiseitig) | +0.6554 [+0.6296, +0.6812], 11 / 35 (61 konstante Reihen fallen heraus) | +0.7308 [+0.6597, +0.8018], 4 / 8 | +0.6755 (3 Familien) | konstante Reihe als ρ = 0: gepoolt +0.2430 [+0.1227, +0.3696] (24 / 96), Test +0.2436 [+0.0589, +0.4440] (12 / 24) |
| `mahal_to_ref_mean` (Erwartung < 0) | -0.5651 [-0.6901, -0.4286], 24 / 96 (10 000: [-0.6931, -0.4267]) | -0.5521 [-0.7262, -0.3661] | -0.6228 [-0.8594, -0.3884] | λ_rel ≥ 0.1: -0.3042 [-0.5089, -0.0827]; Test -0.2833 [-0.5619, +0.0143] |

Lesart:

* **`sae_fvu`.** Der Effekt besteht vor allem darin, dass der λ=0-Pfad den rekonstruierbaren
  Bereich verlässt. Die mediane FVU am Endpunkt nach λ_rel 0 / 0.1 / 0.3 / 1 / 3 / 10 / 30 ist
  0.3118 / 0.2247 / 0.2199 / 0.2195 / 0.2186 / 0.2184 / 0.2183; die ungestörten K0-Starts liegen im
  Median bei 0.2231. Dazu kommt eine winzige, aber konsistente monotone Drift (etwa 0.001) über
  die gesättigten λ-Werte. Sie trägt das negative Vorzeichen auch nach Ausschluss von λ < 0.3
  (-0.4469).
* **Übereinstimmung mit 98k für (ii) und (iii):** Mahalanobis -0.5651 gegen -0.5539; Schritte
  0.6554 (11 Familien / 35 Starts) gegen 0.6424 (10 / 30).
* **Grenze unerreichbar.** 61 von 96 Starts haben log p0 > -1 (Median -0.683; bei 98k 66/96).
  Querende Anteile nach λ_rel: 0.365, 0.104, dann 0.
* **BH** rechnet keine Tabelle. Die KIs von (i) und (iii) liegen weit von 0.

**FVU-Zertifikat: nicht gespeichert.**

* `common.direction_bank` berechnet `bank.meta["sae"] = {fvu, certified = fvu ≤ 0.4}` auf den
  Aktivierungen aller Tokens. H und J speichern die Bank aber nicht: Es gibt kein
  `directions_meta.json` in `runs/H` oder `runs/J @step143000`, und das Manifest hat kein
  FVU-Feld. Die einzige Stufe-B-Bank bei 143k auf Drive (Sweep, Konto B) hat keine SAE-Metadaten.
* Ersatzgröße: die `sae_fvu` der ungestörten K0-Starts (λ_rel-0-Zeilen der
  `sae_feature`-Diskriminante, Abstand 0). Median 0.2231, Mittel 0.2304, Spannweite 0.114–0.355;
  Endpunkte auf der Wolke 0.218.
* Ihr Nenner ist die Varianz je Stichprobe der K8/C1-Wolke, nicht die FVU-Definition der Bank.
  Beide Werte liegen unter 0.4, sind aber **nicht** die Zertifizierungszahl.
* step143000 ist der Checkpoint, auf dem das SAE trainiert wurde, und damit der, an dem es am
  besten passen sollte.

**Diskriminante (c) `sae_feature`: degeneriert.**

* Alle 672 Pfade haben `initial_g = final_g = 1.0`, alle 96 λ=0-Pfade enden „flat“ (Abstand 0),
  und die Diskriminante quert bei keinem λ.
* Ursache: Feature 451 hat an den Starts und am Median der Zielwolke eine Präaktivierung ≤ 0.
  `build_discriminants` setzt dann b = a0 + 1 = 1, und g = −relu(pre) + 1 ist konstant ohne
  Gradient. Die λ>0-Pfade sind reiner Dichteabstieg.
* Ihre `lambda_boot`-Zeilen (z. B. `sae_fvu` +0.005 [-0.167, +0.175]; crossed, final_g und steps
  NaN mit 0 Clustern) sagen nichts über (c).
* Code: `stage_h_lambda.build_discriminants` (b = a0 + 1, wenn aT ≤ a0) und `select_sae_feature`
  (prüft nicht, ob das Feature an der Dossier-Position aktiv ist). In dieser Runde **nicht**
  behoben.

**Pfaddiagnostik.**

* Querende Anteile nach λ_rel 0 / 0.1 / 0.3 / 1 / 3 / 10 / 30:
  * `target_logit` 0.365 / 0.104 / 0 / 0 / 0 / 0 / 0
  * `probe` 1 / 1 / 1 / 0.969 / 0.833 / 0.677 / 0.667
  * `entropy` 1 / 0.990 / 0.969 / 0.917 / 0.823 / 0.719 / 0.719
  * `sae_feature` überall 0
* Anteil der Pfade, die auf `max_steps` enden: `target_logit` 0.10–0.135 (λ_rel > 0), `probe`
  0.885 bei λ_rel 0.1, `entropy` 0.09–0.26.
* Sättigung wie bei 98k (`target_logit`, mediane Spannweite je Start über λ_rel ≥ 0.3 gegen
  |λ0 − λ0.3|): Mahalanobis 0.0014 gegen 1.52, `sae_fvu` 0.0021 gegen 0.098, `dist_from_start`
  0.012 gegen 2.49, `final_density` 7.7e-6 gegen 0.068. Der mediane Mahalanobis-Abstand ist 5.17
  bei λ 0 und 3.652–3.654 für jedes λ_rel ≥ 0.1.
* Nur der reine Dichte-Sweep (`sae_feature`) ist nicht gesättigt: 5.06, 4.31, 4.00, 3.81, 3.71,
  3.68, 3.67.

**Weitere `lambda_boot`-Zeilen bei 143k** (Auswahl, gepoolt):

* `target_logit`: B -0.996 [-1.000, -0.988], `final_density` +0.458 [+0.324, +0.591], `p_freq`
  +0.918 [+0.829, +0.988], `entropy` +0.897 [+0.779, +0.993], `n_shared_features_with_ref`
  -0.337 [-0.430, -0.247].
* `probe`: `sae_fvu` -0.939 [-0.961, -0.916] (Test -0.951 [-0.984, -0.917]).
* `entropy`: `sae_fvu` -0.289 [-0.420, -0.155].
* `n_active_features` ist überall NaN, weil TopK immer genau k = 32 Latents aktiviert.

**Feature-Kandidaten** (gepoolt; die präregistrierte Regel „auf DEV einfrieren, auf Test prüfen“
ist nicht auswertbar, weil `lambda_features` (57 MB CSV / 8.8 MB parquet) über der
Download-Grenze lag; alle Kandidaten sind explorativ):

* **`candidate_all` ist kein Stabilitätskriterium.** Es ist wahr, sobald jeder Start, in dem das
  Feature überhaupt aktiv ist, ein Kandidat ist. Für `target_logit` kommen 180 der
  293 `candidate_all`-Features in einem einzigen Start vor, keines in mehr als 17 von 96.
* **Strengere Regel** (n_starts ≥ 24 und Kandidatenanteil ≥ 0.9): `target_logit` hat 7 solche
  Features, 4 davon vorzeichenstabil (alle schalten bei λ > 0 an), keines mit n_starts ≥ 48.
  Beispiele:
  * 32253: ρ̄ 0.9606, Kandidat in 44/47 Starts. Unter `entropy` ist ρ̄ -0.908, das Vorzeichen
    kippt also zwischen Diskriminanten.
  * 30379: 0.9787, Kandidat in 41/42 Starts.
  * 11142: 0.9613, Kandidat in 28/29 Starts, reines Anschalten bei λ > 0.
  * 5404: aktiv in 84 von 96 Starts, aber ρ zwischen -0.612 und 1.0.
  * Negativ: 26072 (-0.793, 87 Starts, schaltet in 26 aus) und 80 (-0.7528, 51 Starts).
* **`probe`:** 43 Features in ≥ 48 Starts mit Anteil ≥ 0.9 und stabilem Vorzeichen (z. B. 25495
  0.989, Kandidat in 93/94; 1579 -0.9335 in 92/96).
* **`entropy`:** 24206 (0.9885, `candidate_all` über 71 Starts; unter `target_logit` -0.680).
* **`sae_feature`:** nicht als (c) deutbar. 2241 hat ρ zwischen -1 und +1 über die Starts.
* **Vorbehalt Sättigung.** Ein reines An/Aus-Muster bei λ = 0 ergibt nur |ρ| = 0.612 (= √(3/8)).
  |ρ| ≥ 0.7 verlangt eine Rangfolge über fast identische Endpunkte und rankt deshalb teils
  numerische Drift.

**Lauf J bei 143k.**

* Status ok, 0 gescheiterte Substufen (bei 98k war `sharing` ohne SAE gescheitert), 2 335 s.
* Pile-Prüfung wie bei 98k: Misfolding auf Rang 9 von 11, Gegenpol `books`. Der Ersatz wird für
  die RDX-Kontraste weiterhin nicht verwendet.
* **Sharing.** Jaccard der Mengen aktiver SAE-Latents an der letzten Position, L11.block_input,
  deskriptiv und ohne Bootstrap:

  | Paar | Jaccard |
  |---|---|
  | everyday ↔ culture_politics | 0.0700 |
  | everyday ↔ law | 0.0223 |
  | everyday ↔ protein_misfolding | 0.0377 |
  | culture_politics ↔ law | 0.0615 |
  | culture_politics ↔ protein_misfolding | 0.0402 |
  | law ↔ protein_misfolding | 0.0155 |
  | K0 ↔ K8C1 | 0.7017 (888 / 875 Latents, 727 gemeinsam) |
  | global (Schnitt aller Gruppen) | 0.0029 (3 von 1 036) |

  Die Werte sind nicht vergleichbar, weil der Jaccard der Vereinigungsmengen mit der
  Gruppengröße wächst (64 / 16 / 8 / 8 / 96 / 96 Zeilen). K0 ist die Vereinigung der
  Domänengruppen, Jaccard(Gruppe, K0) ist also per Konstruktion n_Gruppe / n_K0.
* **Netzwerkaktivität.**
  * `frac_active` = 32/32 768 per TopK-Konstruktion.
  * Der Wechsel K0 → K8C1 lässt die mittlere Masse fast unverändert (72.14 → 71.46) und senkt das
    Participation Ratio von 115.64 auf 102.16.
  * Head-Profile: alle Kosinusse ≥ 0.856; K0 gegen K8C1 bei L2/L5/L8/L11 0.99918, 0.99932,
    0.99414, 0.99966.
* **RDX mit SAE-Basis.**
  * Die Basis gibt es nur an L11 (n_sites = 1), mit k = 10 Decoder-Zeilen (`features_D` +
    `features_E`).
  * Kultur/Politik ↔ Misfolding: Auslesung last z 1.996 (Mindesteffekt nicht erreicht), mean
    z 8.604.
  * law ↔ Kultur/Politik: last 2.657, mean 10.674.
  * E ↔ D: last 4.808, mean 10.907.
  * K0 ↔ K8: last 5.170, mean 11.995.
  * Alle Zeilen sind nicht präregistriert. Pro Einheit erreichen nur 33–50 % z ≥ 2, die gepoolten
    Effekte replizieren also in der Mehrheit der Familien nicht.
* **Präregistrierte Zeile 15 bei 143k** (Domänenachsen, last): z 8.37 / 33.90 / 16.97 / 25.93,
  z_mean 21.29, erklärt 0.077 (98k: z_mean 25.57). Hauptkomponente `dom_protein_misfolding`
  (Anteil 0.62–0.71).

### 3.5 Hysterese, Relaxation, Suszeptibilität, Lens (D)

* **Hysterese nach K** (gepoolt): K0 0.4501 [0.4305, 0.4698], K8 0.2026 [0.1829, 0.2223], K28
  0.2221 [0.2030, 0.2403].
  * Die Fläche ist bei K0 am größten und halbiert sich mit Wechseln.
  * `ent` gibt eine negative Fläche (K8 -0.0617 [-0.0798, -0.0453]).
  * State 2143, `lang_dom`: K0 0.3012 (z 6.72), K5 0.1749 (z 2.64).
  * Hinweis: `area_z` = Fläche / SD der Null, nicht (Fläche − Null)/SD.
* **Relaxation.** Zeitkonstanten sind fast nur für `lang_dom` identifizierbar, und höhere Dosis
  heißt schnellere Antwort:
  * K0, ε 0.5: 95/96 identifizierbar, Median τ 7.01 Tokens (IQR 5.18–9.62).
  * K8, ε 0.2: 78/96, 11.77 (7.45–22.34); K8, ε 0.5: 37/96, 2.00.
  * K28, ε 0.2: 83/96, 12.85.
  * `ent` ist in 10–20 % der K>0-Zeilen identifizierbar, bei K0 nie.
  * 41 identifizierbare Fits sind randzensiert (35 bei τ = 1000, 6 bei 0.5; Gegenprüfung) und
    müssen vor jeder Zusammenfassung maskiert werden.
* **Suszeptibilität,** mittleres fD bei ε 0 / 0.2 / 0.5 / 1.0:
  * `lang_dom` K0: 0.001 / 0.049 / 0.900 / 0.990
  * `lang_dom` K8: 0.311 / 0.792 / 0.983 / 0.994
  * `ent` K8: 0.311 / 0.219 / 0.031 / 0.006

  `lang_dom` sättigt den deutschen Output bis ε 0.5, `ent` unterdrückt Deutsch in gemischten
  Kontexten.
* **Lens.** Das State-2143-Ziel liegt am letzten Layer weit hinten (Rang 6712 / 9792 / 9598 / 8125
  für K0/K2/K4/K5), die Standardfamilien haben das Ziel fast immer auf Rang 0 (1 076 von 1 152).
* **Dossier-Stelle D1 ist strukturell leer:** λ ist NaN in allen 288 Dossier-Zeilen, Front 0,
  Ausbreitung ≤ 1 Position. Eine Injektion am letzten Kontext-Token bei L11 hat nur einen
  Fit-Layer und keine späteren Positionen.

### 3.6 FDT (D)

* Effektive Temperatur (Projektion, `T_eff_proj_0`, Familienmittel), `lang_dom`: K0 0.0837
  [0.0051, 0.1556], K8 1.1638 [0.9806, 1.3438], K28 1.3008 [0.7848, 2.1092].
* Für `ent` ist das Verhältnis schlecht definiert: K8 1.3418 [-2.624, 6.164], χ < 0 in 54 % der
  Zeilen.
* `T_eff_fD_0` ist NaN in 28 K0-Zeilen, wo χ_fD exakt 0 ist; sonst liegt es bei K0 zwischen -0.077
  und 0.224 (Gegenprüfung), das Minimum über alle Zeilen ist -467.6.
* 8 Stichproben je Bedingung. Das Verhältnis explodiert, wenn χ gegen 0 geht.

### 3.7 Zellularautomat (D)

* K8/C1 − K0, gepaarter Familien-Bootstrap: `lz76` +0.3345 [0.322, 0.3478], `lz76_tail` +0.0353
  [0.0194, 0.0525], `block_entropy_2x2_tail` +0.1397 [0.0871, 0.1947], `activity_tail` +0.0018
  [-0.0037, 0.0076].
* Die Komplexität steigt fast nur im vollen Feld (deutsche Kontext-Tokens), im gemeinsamen Tail
  kaum.
* Das Symbolfeld ist von Stellen unter der Schwelle (N) dominiert: `frac_N` 0.69 (K0) und 0.58
  (K8) bei einer Schwelle von 0.00239. Das Markerlexikon hat nur 6 E- und 6 D-Token-IDs. Das
  Ergebnis hängt an diesem Lexikon und an der automatischen Schwelle.

### 3.8 Stufe E: lineare Antwort und Tensornetz (Konto B, DEV01/DEV02)

* **Nichtlinearität.** Normgewichtet ab der Injektion liegt sie bei 0.0133–0.0642 für ε 0.01 und
  0.144–0.604 für ε 0.1. Das ist überwiegend der gerade Anteil zweiter Ordnung. Der mediane
  Kosinus gemessen/linear ist 1.000 bei ε 0.01 und 0.969–0.988 bei ε 0.1.
* **K8 gegen K0.** Bei K8 reagiert `lang_dom` nichtlinearer als bei K0 (0.216–0.604 gegen
  0.144–0.201 bei ε 0.1), mit großem ungeradem Anteil (0.102–0.264 gegen 0.022–0.044).
* **Faktorisiertes Tensornetz (r = 64) gegen den exakten JVP.** Das Netz trifft die lineare
  Antwort nicht:
  * Fehler 0.689–0.856 (Median 0.805), an der letzten Position 0.848–1.073; Kosinus an der
    letzten Position -0.492 bis 0.615.
  * Die Amplitude geht verloren: Das faktorisierte Netz hat nach der Injektion nur 0.26 der
    linearen Norm (Median), und schon an der Injektionsposition ist der Fehler 0.545.
  * Ob das an der Trunkierung oder an der eingefrorenen Attention liegt, lässt sich nicht
    trennen: Die Volle-Rang-Referenz (8.46 GB) wurde übersprungen, und eine
    Frozen-Attention-Referenz wird nicht berechnet.
* **TNCO** findet denselben Pfad wie `opt_einsum` (10^7.6438 FLOP) und ist 30× langsamer bei der
  Pfadsuche (Millisekunden). Die eigene TNCO-Kostenzahl ist um log10 2 kleiner, weil sie
  Multiply-Adds zählt.
* **float64** war nötig: Bei ε 0.01 hätten 54.3 % der Positionen in float32 ein SNR < 10.
* **Vorbehalt:** E injiziert die an L11 gefitteten Vektoren an L4. Laut den D-Manifesten haben
  diese mit den L4-Achsen nur cos 0.27–0.34 (`lang_dom`) bzw. 0.09–0.23 (`ent`).

### 3.9 Checkpoint-Sweep: Replikate der Endpunkte 4 und 5 (Gegenprüfung)

* **Nichtmonotonie (Test, gepaart, 10 000 Ziehungen).**
  * `nmi_last`: Das KI schließt 0 nur an 3 von 20 Checkpoints positiv aus (z. B. step3000
    +0.0139 [0.0046, 0.0247]).
  * `nmi_tail`: Das KI liegt an 8 von 20 Checkpoints unter 0 (z. B. step143000 -0.0152
    [-0.0260, -0.0055]).
  * Die K0-Grundlinie steigt über das Training (`nmi_last` 0.030 → 0.181).
* **Hysterese** (16 Familien, DEV und Test gemischt):
  * step1000: K0/C1 0.558 [0.538, 0.575], K8/C1 0.224 [0.193, 0.256], K28/seg 0.271
    [0.241, 0.302].
  * step143000: 0.451 [0.429, 0.473], 0.223 [0.188, 0.254], 0.251 [0.218, 0.281].
  * Die Hysterese ist schon bei step1000 groß.
* Der Sweep rechnet nur B und D. Für die Endpunkte 1–3 und 6 gibt es kein Checkpoint-Replikat.

### 3.10 Kompass, Domänenachsen, RDX bei 98k (I, J)

* **Domänenachsen.** Sie stehen fast senkrecht auf der Bank, sind untereinander aber stark
  korreliert: cos(`dom_law`, `dom_culture_politics`) 0.755, law ↔ misfolding 0.577, culture ↔
  misfolding 0.637. Gegensteuern gegen `dom_law` entfernt deshalb viel von
  `dom_culture_politics`.
* **Jede Steuerrichtung senkt log p(Ziel)** in jeder Familiengruppe. `lang_dom` schadet den
  Misfolding-Familien am meisten (K8/C1, ε 0.1: -0.2704 [-0.3018, -0.2390], 2 Cluster).
* **RDX-Domänenachsen.** Bei der Auslesung „mean“ erklären sie die Sprachkontraste über Zufall
  hinaus (K0 ↔ K8: 4/4 Sites, z_mean 4.15), bei „last“ nicht (0/4). Die `heads`-Basis ist falsch
  gebaut (§4, Punkt 16), ihre Ergebnisse (z. B. E ↔ D bei L8, z_mean 19.32) sind nicht belastbar.
* **Pile-NLL** ist auf Sequenzen, die step98000 noch nicht gesehen hat, leicht höher (2.6943
  gegen 2.6387).

### 3.11 Spurenmodell (Notebook 12), nur Technik

Mit dem kaputten Vokabular (§4, Punkt 20) sind die Modellzahlen mit nichts Geplantem
vergleichbar:

* Validierungsverlust 0.3135 nats/Token, gegen Bigramm 1.0614, Unigramm 3.5788 und den
  untrainierten Zwilling 10.1309. Adressen, Selektoren und Topics fielen auf OOV-Tokens, und
  damit wurde die schwerste Klasse trivial.
* Durchsatz auf der A100: 244 679 Tokens/s, MFU 17.96 %. Ein korrekter Lauf mit V ≈ 72 000 ist pro
  Token etwa 1.7× langsamer.
* Der Alignment-Gewinn gegenüber dem untrainierten Zwilling hängt stark vom Ridge-α ab (α = 1:
  0.0650; α = 100: 0.2828). Die α-freie Procrustes-Abbildung der Transplantation zeigt keinen
  Gewinn.

---

## 4. Abweichungen von der Präregistrierung und bekannte Mängel

„Behoben“ = mit Commit `97d4104` eingespielt. Die
vorhandenen Lauftabellen ändern sich dadurch nicht. Was neue Werte braucht, steht in §5.

### 4.1 Auswertung und Statistik

| # | Abweichung / Mangel | Behoben in dieser Runde | Offen (Entscheidung oder Daten) |
|---|---|---|---|
| 1 | 2 000 statt 10 000 Bootstrap-Ziehungen (§2), in jeder Stufe | `Limits.n_boot = 10000`; Notebook 13 rechnet jede konfirmatorische Zeile mit 10 000. Der laufende Sweep bleibt absichtlich bei 2 000 (`LIMITED_N_BOOT`, Resume-Signatur festgehalten), damit alle Checkpoints gleich gerechnet sind | – |
| 2 | DEV- und Testfamilien gepoolt, Zeilen- statt Familienmittel, keine Testzeile; Bank (`lang_dom`, `lang_probe`) auf allen Familien gefittet | Spalten `split` (all / test / dev), `confirmatory`, `stat_level`, `p_boot` in C, D, G, H (in I `p_boot`); `prereg_eval` rechnet auf F01–F12 × {A, B} | Die Stufe-G-Wolken einer Testfamilie enthalten weiter die Formulierungen C/D (vermerkt). K und die Spurenquelle haben keinen Split. Die Bank wird weiter auf allen Familien gefittet (Zirkularität für T3; eine DEV-only-Option fehlt) |
| 3 | BH über die Sekundärgruppe nirgends berechnet | `prereg_eval`: BH über 1–6, 8 und 9 (nur bei G0), 10, 13, 14; `bh_q_endpoint` als Sensitivität | Die Gruppierung als Entscheidung festhalten: die Urteile zu 2 und 3/K28seg hängen davon ab |
| 4 | Mindesteffekt nach Punktschätzung (`min_effect_met` in I, `steering_boot`) | eine Intervallregel (`common.min_effect_verdict`: erfüllt nur bei `ci_low ≥ δ`; offen / null / negativ) in D, I und `prereg_eval` | Das ist eine Klarstellung nach Sichtung der Daten und sollte als datierte Ergänzung der Präregistrierung geführt werden |
| 5 | **Präregistrierung nicht getaggt.** `prereg-v1` existiert nicht. Die letzte Änderung an `PREREGISTRATION.md` ist Commit `58ec270` (26.10.2026 00:41 UTC), also vor dem ersten Lauf (Stufe C ab 12:48 UTC, auf Commit `a018239` von 12:41 UTC) | – | Entscheidung: den Tag nachträglich auf `58ec270` setzen und alle später festgelegten Lesarten (Punkte 4, 9, 12, 14, 21 und die Regel ρ = 0 bei 10(ii)) als datierte Ergänzungen notieren, mit Vermerk, dass sie nach Sichtung von Lauf 1 kamen |
| 6 | Soll/Ist-Audit (§4) für C und D nicht prüfbar, Endpunkt 14 ohne Audit | Audit-Zusammenfassung im Manifest von C und D (max, Quantile, n > 1e-4, `valid`); `prereg_eval` rechnet `audit.C` und `audit.D1` aus den Tabellen von Lauf 1; I schreibt `audit_rel_err` für Head- und Residualarm | Replace-Modus (Gegensteuer-Arm 13) und gemittelter `rand`-Arm weiter ohne Audit (1 152 Zeilen NaN) |
| 7 | Ausschlussregel `review.csv` (§4) nicht implementiert | – | Würde in Lauf 1 nichts ausschließen (960/960 `meaning_ok`). Implementieren oder begründet streichen |

### 4.2 Primärendpunkt

| # | Abweichung / Mangel | Behoben in dieser Runde | Offen (Entscheidung oder Daten) |
|---|---|---|---|
| 8 | Keine Tabelle für die Statistik (kein `damage_boot`, `damage_phase` ohne Familienschlüssel) | `damage_contrasts` (je Formulierung Δλ `lang_dom`, Δλ `rand` über alle Richtungen, DiD) und `damage_boot` (Familienmittel, Splits, Intervallregel, `monotone_in_K`); `prereg_eval` rechnet P direkt aus `damage.parquet`, ohne GPU-Neulauf | – |
| 9 | **Positions-Konfundierung außerhalb C1.** Für K > 0 bei C2/C3/seg injiziert der Code am letzten deutschen Token, nicht am letzten Token von Einheit 9 (§2); `rand`-λ steigt dort 4–5× | Spalte `position_deviation`, `confirmatory` nur C1; `prereg_eval`: `P.dev.*` nicht interpretierbar | Entscheidung: C1 als konfirmatorische Zelle festschreiben (H1 nennt kein C) und als Abweichung dokumentieren, oder eine positionsgleiche K=0-Referenz rechnen (zusätzlicher D1-Durchlauf über die K=0-Varianten) |
| 10 | K0 → K>0 wechselt auch die Sprache des Injektions-Tokens (Englisch → Deutsch); Sprung bei K = 2, danach Plateau, in Lauf 1 und an jedem Sweep-Checkpoint (Korrektur 28.10.2026: das Token ist dasselbe „.“, verschieden sind die Sprache des Satzteils davor und der letzte Wechsel danach; Kontrolle festgelegt in Nachtrag 6.11, Notebook 14) | – | Kontrolle: positionsgleiches englisches Token in K>0 (oder umgekehrt), ein reduzierter D1-Durchlauf. Entscheidung, wie die Monotonie-Klausel gelesen wird |
| 11 | Dosen 0.001, 0.0025, 0.005, 0.02 fehlen; keine Dosis auf DEV eingefroren | `prereg_eval` berichtet jede vorhandene präregistrierte Dosis einzeln, mit Bonferroni-Hinweis; `FROZEN_DOSE` / `--frozen-dose` | Entscheidung: Dosis jetzt auf DEV einfrieren (nicht mehr blind, weil die gepoolten Zahlen bekannt sind) oder Bonferroni über drei Dosen. Die fehlenden Dosen gibt es nur mit einem Neulauf von D; `Limits.damage_doses` ist unverändert (0.01, 0.05, 0.1) |
| 12 | Tiefen-`lang_dom` (L4) verfehlt das Entrauschungs-Gate (cos 0.126; L11: 0.591); die rohe Achse wird nicht gespeichert, obwohl die Warnung „beide Achsen berichten“ verlangt | – | Entscheidung, welche Achse der Primärendpunkt meint; `save_bank` verwirft `vec_plain` weiterhin |
| 13 | Dossier-Stelle D1 strukturell leer (ein Fit-Layer, keine späteren Positionen) | – | Entscheidung: λ dort streichen oder die Positionsausbreitung von einer früheren Position an L11 messen |

### 4.3 Sekundärendpunkte

| # | Abweichung / Mangel | Behoben in dieser Runde | Offen (Entscheidung oder Daten) |
|---|---|---|---|
| 14 | Endpunkte 2 und 6 ohne eigene Zeile; `STAGES.md` verwies 6 auf `quantity="I"` | gepaarte Zeilen `T_C3_minus_C1` und `B_ent_minus_rand` (je K, C, Vorzeichen, ε und gepoolt); `STAGES.md` korrigiert; 6 einseitig im State-2143-Vorzeichen +1, −1 als Kontrolle | Lesart von 6 bestätigen (einseitig +1 oder zweiseitig); eine Kontrolle für das Zielwahrscheinlichkeits-Regime fehlt (State 2143 log p0 -13.6, Familien Median p 0.5) |
| 15 | **Endpunkt 3: Leerraumklasse nicht messbar.** Im gemeinsamen Suffix gibt es keine Leerraum-Position (ws hat überall 0 Positionen), `delim` = punct + boundary auf 2–6 Positionen; dazu ist das Recovery-Verhältnis schlecht konditioniert (Nenner-Untergrenze 1e-6 nats) | `prereg_eval`: S3 einseitig (DESIGN 4.1); explorativer Überschuss in nats (`S3x`) | Die Leerraum-Hälfte der Hypothese bleibt ungetestet. Entscheidung: R_punct+boundary als Abweichung akzeptieren oder Region und Klassen ändern; Mindest-\|T\| oder nats-Überschuss als Ergänzung |
| 16 | G: T3-Layer per Gleitkommarauschen; G1-n-Matching unstratifiziert; `layer_of_max_contraction` ohne Kontraktion; G0 poolt `mean_prefix` statt „Mean-Pool“; T5 leer; RDX `heads`-Basis falsch gebaut (J: Heads von Block l an `L{l}.block_input`, die erst danach schreiben) | T3 = Argmax des euklidischen `norm_K` (Mahalanobis-Wahl bleibt als Spalte); stratifiziertes n-Matching je Formulierung; NaN ohne negatives Inkrement | Neulauf G (die Werte ändern sich). T5 braucht D im selben Laufverzeichnis. Die G0-Poolung ist dokumentiert, nicht geändert. Die `heads`-Basis in J ist nicht korrigiert |
| 17 | H: 10(ii) verliert konstante Reihen (66/96); `sign_as_expected=False` bei NaN; kein DEV-Einfrierpfad für Features | konstante zensierte Reihe = ρ 0 (die Zeile „wie gelaufen“ bleibt als `as_run`); `sign_as_expected` leer bei NaN; Splits; `lambda_features_frozen` | Die +1-nat-Grenze ist für 66/96 (143k: 61/96) Starts unerreichbar, und die Zielwolke liegt auf der falschen Seite. Das λ-Raster sättigt ab λ_rel 0.3. Die `sae_feature`-Diskriminante ist degeneriert (§3.4). H und J speichern das FVU-Zertifikat nicht. 99k fehlt für 11 |
| 18 | I: drei Dosen als präregistriert markiert; Head-Arm 36× kleiner; Gegensteuer-Arm nicht dosisgleich | eine Dosis präregistriert (auf DEV eingefroren, sonst 0.05; Manifest `contrast_13_eps`); Arm `residual_abs` mit derselben absoluten Norm wie der Head-Arm, dazu `delta_norm` | Gegensteuern bleibt eine unzentrierte Projektion. Der Head ist nicht auf DEV eingefroren (Fallback-Hälfte mit überwiegend Testfamilien; `--worst-head L:H`). Deutsches Lexikon mit 1–10 IDs je Domäne; der Kompass ist nicht domänenselektiv |
| 19 | **J: Regel-16-Gegenpol `books` nicht angewandt** | `rdx_summary`: `evaluated_pole`, `counterpole_rule16`, `preregistered_form = False`; Manifest `rule16`; `prereg_eval` kennzeichnet 15 | Es gibt keine `books`-Stimulusfamilien und keine `books`-Achse. Entscheidung: Stimuli schreiben (dann Neulauf J) oder 15 ausdrücklich als explorativ führen. Das isotrope Zufallsnull ist nachsichtig; die `pca`-Basis ist in-sample |

### 4.4 Daten, Eingaben und Technik

| # | Abweichung / Mangel | Behoben in dieser Runde | Offen (Entscheidung oder Daten) |
|---|---|---|---|
| 20 | **Spurenpipeline (Notebook 12).** Die Serialisierung brach ab (`BrokenProcessPool`): 558 von 2 135 Chunks fehlen, die Abdeckung liegt bei 73.9 %, Sept.–Dez. 2016 fehlen. `serialize_params.json` wurde nie geschrieben, der Split lief nie. Das Vokabular wurde aus 0 Transaktionen gebaut (4 Adressen, 0 Selektoren, 0 Topics) und endete mit Code 0. Training mit Adress-OOV 0.967, trotzdem `counts_for_prereg = True` | Fix-Runde 2 (`docs/TRACE_SOURCE.md` §13): `serialize_params.json` vor dem ersten Chunk, Wiederholung mit weniger Workern, Code 2 mit Liste fehlender Chunks; `vocab` Code 1 bei 0 Transaktionen; `train --max-oov-address` (0.5); Pipeline-Zellen brechen laut ab (`pipeline()`, `require()`); `counts_for_prereg` verlangt jeden Schritt ok | Neulauf ab Schritt 2 (§5.3) |
| 21 | Spuren-Urteil: der kv-„Effekt“ besteht nur gegenüber zerstörerischen Kontrollen; die Sonde schlägt die Oberflächen-Baseline nicht | `trace_verdict`: Effekt auch gegen das unveränderte Modell, in vorab festgelegter Richtung, `align_beats_untrained`, `beats_surface` (datierte Ergänzung in `frozen_feature.json`), `lm_corpus_ok`; Tabelle `trace_damage`; `rejudge_trace` für alte Läufe | – |
| 22 | Stand-in (18): 3 Prompts aus einer Familie, ε 1.0 sättigt, `v_src` layer-inkonsistent | 12 Prompts reihum über die Familien, ε 0.3, `v_src` je Quell-Layer; ein alter Stand-in-Lauf zählt nicht mehr | Neulauf Notebook 11; nicht geändert: `n_null` 10 und der Kosinus, der auf denselben E/D-Texten gefittet wird, die beide Achsen definieren (in-sample) |
| 23 | **TTT-Korpus fehlt** (`data/ttt/dao_fork`, `neutral`, `nofork` enthalten nur READMEs); Memorisation ohne Dynamikumfang (2–3 Schritte bei LR 1e-4, Schwelle 0.5); `exceeds_all_controls` vergleicht Beträge ohne KI | – | Daten liefern; LR und Schwelle neu kalibrieren; Kriterium mit Intervall und DEV/Test-Split |
| 24 | `ent` nicht ARI-orthogonalisiert (`data/s4/axes.npz` fehlt) | – | optional nach DESIGN §3; Datei liefern, wenn die Dossier-Konstruktion gemeint ist |
| 25 | C lief vor B mit eigener Bank (`a018239`), Unigramm-Quelle der `freq`-Achse unbekannt | Code seit `658f578` strikt (kein stiller Fallback) | p_freq bei ε 0 zwischen C (`grid_summary`) und B (`baseline`) vergleichen; bei Abweichung Notebook 02 neu |
| 26 | Zwei Drive-Konten: G sah D nicht | Notebook 13 liest beide Wurzeln (Verknüpfung `LSNE_B`) | Neuläufe von G im Konto, in dem D liegt |
| 27 | Veraltetes `error.txt` neben einem ok-Manifest (D); Substufenfehler in `summary_09` versteckt | Ein ok-Lauf entfernt veraltete `error.txt` / `substage_errors.txt`; Status `ok_with_substage_errors` | Das alte `error.txt` in `runs/D` (Konto A) liegt noch da (Notebook 13 ignoriert es) |
| 28 | Stufe E: L11-Vektoren an L4 injiziert; keine Frozen-Attention-Referenz; zwei Kostenkonventionen in `cost_log10` | – | Code-Ergänzung offen |
| 29 | Stufe F: lexikografische Checkpoint-Reihenfolge; die Sub-Manifeste nennen `f_stages` [B, C, D], obwohl nur B und D liefen | – (bewusst nicht angefasst, der Sweep läuft) | kosmetisch |
| 30 | CLI-Hilfe `--n-unigram` nennt den Standard 200 (tatsächlich 2 000) | – | kosmetisch |

---

## 5. Nächste Schritte

### 5.0 Voraussetzung: Korrekturen auf den Branch bringen

Die Notebooks klonen den Branch `claude/language-switching-non-equilibrium-ejpz0q` von GitHub. Die
Korrekturen dieser Runde, `lsne/prereg_eval.py` und `notebooks/13_prereg_evaluation.ipynb` sind mit
Commit `97d4104` gepusht (erledigt). Notebooks, die vorher gestartet wurden, liefen mit `76a4a5a`;
dort existiert Notebook 13 nicht.

Der laufende Sweep hat seinen Code beim Start geklont und bemerkt den Push nicht. Wird Notebook 05
nach einem Abbruch neu gestartet, nutzt es den neuen Code. Laut Code dieser Runde bleiben die
Checkpoint-Läufe vergleichbar: n_boot im Sweep fest bei 2 000, Resume-Signatur unverändert. Neu
kommen je Checkpoint `damage_contrasts`, `damage_boot` und das Audit im Manifest hinzu.

### 5.1 Jetzt, auf CPU: Notebook 13 auf den Daten von Lauf 1

Das stört den Sweep nicht: Stufe F wird nie geöffnet.

1. Im Konto B den Ordner `LSNE` freigeben und in Konto A als Verknüpfung `LSNE_B` unter „Meine
   Ablage“ anlegen (`docs/RUNBOOK_COLAB.md` §11).
2. Notebook 13 in Konto A mit CPU-Laufzeit ausführen: `REVISION = "step98000"`,
   `N_BOOT = 10000`, `FROZEN_DOSE = None` (es gibt keine auf DEV eingefrorene Dosis).
3. Die Ausgabe `MyDrive/LSNE/prereg_eval/pythia-160m@step98000/` danach z. B. in `….run1`
   umbenennen oder kopieren, damit eine spätere Auswertung nach den Neuläufen sie nicht
   überschreibt.

Was Notebook 13 aus Lauf 1 liefern kann:

* das formale Urteil zum Primärendpunkt je Dosis, mit Audit D1;
* das Audit von C;
* S1–S6 auf der Testmenge mit BH;
* G0;
* S10(ii)/(iii) mit der Regel ρ = 0.

Was es aus Lauf 1 nicht liefern kann:

* S13/S14 im BH (die I-Tabelle von Lauf 1 hat kein `p_boot`, das Urteil wird höchstens „offen“);
* S8/S9 (nicht interpretierbar wegen G0);
* S10(i) (kein SAE bei 98k);
* S15 in präregistrierter Form;
* S17, S18 und S18T (siehe §2.3).

Notebook 13 lässt sich auch mit `REVISION = "step143000"` explorativ für H/J bei 143k ausführen.

### 5.2 Checkpoint-Sweep (Notebook 05, Konto B)

* **Erledigt** am 27.10.2026 um 13:21 UTC: 27 Checkpoints, 81 Teilschritte `ok`, 0 Fehler
  (`summary_05_scale_and_checkpoints_shard4.json`).
* Ursprünglicher Hinweis: Nicht anfassen, bis er fertig ist (ETA etwa 13:10 UTC). Danach die D-Manifeste von step98000 und
  step99000, `plan`/`runs` und `summary_05_scale_and_checkpoints_shard4.json` prüfen. Ist die
  Sitzung abgebrochen, Notebook 05 mit `REUSE = True` neu starten: fertige Paare werden
  übersprungen, D setzt aus `_resume` fort.
* Explorativ danach: Familien-Bootstrap je Checkpoint auf der Testmenge aus `damage.csv` und
  Vergleich von F@step98000 (reduziert, 8 `rand`-Richtungen) mit dem Primärwert aus Notebook 13.
  Dafür gibt es noch keinen fertigen Auswerteschritt.

### 5.3 Neuläufe nach den Korrekturen (GPU)

Allgemein gilt:

* `REUSE = False` setzen, sonst lädt das Notebook den alten ok-Lauf.
* Ein Neulauf schreibt in denselben Ordner `runs/<stufe>/pythia-160m@step98000/`. Wer die Tabellen
  von Lauf 1 behalten will, benennt den alten Ordner vorher um (z. B. `…@step98000.run1`;
  `prereg_eval` sucht nur den exakten Ordnernamen).
* Neuläufe auf Konto A, solange der Sweep auf Konto B läuft. Das hat einen weiteren Vorteil: G, H
  und I finden dort die Stufe-B-Bank und Stufe D (T5).

| Reihenfolge | Notebook | Warum | Aufwand |
|---|---|---|---|
| 1 | 11 (Stand-in) | neues Design (12 Prompts reihum, ε 0.3, `v_src` je Quell-Layer); der alte Lauf zählt nicht mehr als Voraussetzung für 18T | etwa 2 min (Lauf 1: 1.92 min) |
| 2 | 12 ab Schritt 2 | `REUSE = True`, `FORCE_STEPS` leer: die Serialisierung setzt bei den fehlenden Chunks fort, der Split läuft erstmals, Vokabular bis Stufe L werden neu gerechnet, das Modell mit dem leeren Vokabular wird nach `….stale-<Zeit>` verschoben | etwa 3–6 h (CPU mit viel RAM für serialize/split/vocab, A100 für den Rest), etwa 40–70 CU |
| 3 | 06 (G) | stratifiziertes n-Matching, euklidische T3-Layerwahl, `split`/`p_boot`; T5 nur mit D im selben Konto | Lauf 1: 37 min |
| 4 | 07 (H), step98000 | Splits, Regel ρ = 0, `lambda_features_frozen`, `sign_as_expected`. **Nur einmal rechnen: auf Konto B mit SAE (§5.4)**, nicht zusätzlich auf Konto A. Sonst gibt es zwei H-Läufe bei 98k, und die Auswertung nimmt den, der zuletzt fertig wurde | etwa 7 min |
| 5 | 08 (I) | eine präregistrierte Dosis, Intervallregel, `p_boot` (damit 13/14 in die BH-Gruppe kommen), Arm `residual_abs`, Head-Audit | Lauf 1: 8.76 min |
| 6 | 07 (H), step99000 | Endpunkt 11 verlangt 98k/99k, nur 98k liegt vor | etwa 7 min |
| 7 | 13 (CPU) | Auswertung nach den Neuläufen | Minuten |

Optional:

* Notebook 02 (C), falls p_freq bei ε 0 zwischen C und B abweicht (1–3 h).
* Notebook 03 (D), nur wenn die fehlenden präregistrierten Dosen nachgeholt werden sollen. Dafür
  muss erst `Limits.damage_doses` geändert werden, das ist in dieser Runde nicht geschehen (etwa
  3.3 h).
* Notebook 10 (K) erst, wenn die TTT-Korpora vorliegen (§4, Punkt 23); laut Runbook dann mit SAE
  bei step143000 und `TTT_CHECKPOINTS`.

### 5.4 SAE bei 98k gegen 143k (Notebooks 01, 07, 09)

Ziel: Endpunkt 10(i) und `sharing` am präregistrierten Checkpoint, verglichen mit 143k.

1. **Auf Konto B (der Sweep ist fertig):** Notebook 01 mit `USE_SAE = True` bei `step98000`.
   Stufe B schreibt die SAE-Güte in `runs/B/pythia-160m@step98000/directions_meta.json`
   (`bank_meta.sae`: `fvu`, `certified` = fvu ≤ 0.4). H und J speichern sie nicht. Konto B hat
   noch keinen `runs/B@step98000`, die Bank, auf die sich D in Konto A beruft, bleibt also
   unberührt.
2. Danach 07 und 09 mit `USE_SAE = True`, `REUSE = False`, `step98000`, ebenfalls auf Konto B. Sie
   überschreiben die alten H- und J-Läufe von Lauf 1 dort. Dieser H-Lauf ersetzt Zeile 4 aus §5.3: Mit
   SAE rechnet Stufe H dieselben Tabellen und zusätzlich die SAE-Spalten (die Zufallsrichtungen der
   Bank hängen nicht am SAE).
3. **Ergebnis (27.10.2026, 16:37 UTC, Konto B):** FVU = **0.198** bei `step98000` (SAE
   `EleutherAI/sae-pythia-160m-32k`, `layers.10`, k = 32, 170 286 Aktivierungen); `certified = true`.
   Die übrigen Richtungen der Bank sind gleich wie ohne SAE (z. B. `lang_dom` `diff_norm` 23.7434 in
   beiden Läufen). 10(i) ist bei 98k damit auswertbar, sobald H mit SAE gelaufen ist.
4. Lesart:
   * Ist das SAE bei 98k zertifiziert (FVU ≤ 0.4), ist 10(i) bei 98k auswertbar. Die Notebooks
     vermerken „nur bei step143000 zertifizierbar“; gemessen ist das bei 98k bisher nicht.
   * Ist es nicht zertifiziert, ist 10(i) bei 98k nicht interpretierbar, und 143k bleibt die
     beste, explorative Angabe.
5. **Vorher entscheiden:** Die `sae_feature`-Auswahl prüft die Aktivität des Features nicht. Ohne
   Codeänderung ist Diskriminante (c) wahrscheinlich wieder degeneriert. Für 10(i) selbst spielt
   das keine Rolle, weil 10(i) die Diskriminante `target_logit` nutzt.

### 5.5 Entscheidungen, die bei dir liegen

1. **Präregistrierung datieren** (§4, Punkt 5): `prereg-v1` nachträglich auf `58ec270` setzen und
   die Klarstellungen dieser Runde als datierte Ergänzungen nach Lauf 1 führen.
2. **Primärendpunkt** (Punkte 9–12):
   * C1 als konfirmatorische Zelle festschreiben, oder eine positionsgleiche K=0-Referenz rechnen;
   * Dosis nachträglich auf DEV einfrieren, oder Bonferroni über 0.01/0.05/0.1;
   * fehlende Dosen nachholen (Neulauf D) oder als Abweichung führen;
   * welche `lang_dom`-Achse bei verfehltem Gate gilt;
   * ob eine Kontrolle für die Sprache des Injektions-Tokens vor einer Bestätigung nötig ist.
3. **BH-Gruppierung** (Punkt 3), weil Endpunkte 2 und 3/K28seg daran hängen.
4. **Endpunkt 3** (Punkt 15): R_punct+boundary als Abweichung akzeptieren oder das Design ändern;
   Mindest-|T| bzw. nats-Überschuss.
5. **Endpunkt 6:** einseitig im State-2143-Vorzeichen (so im Code) oder zweiseitig.
6. **Endpunkt 10:** Grenze und λ-Raster neu kalibrieren (unerreichbare Grenze, Sättigung ab
   λ_rel 0.3); Regel zur `sae_feature`-Auswahl; ob 10(i) bei 98k nur mit Zertifikat zählt.
7. **Endpunkt 13:** Gegensteuer-Arm zentrieren und dosisgleich machen; welche Dosis eingefroren
   wird.
8. **Endpunkt 14:** Head auf einem DEV-Lauf einfrieren (`--worst-head`); welcher Dosisabgleich
   (relativ wie präregistriert oder absolut) die Hauptzeile ist.
9. **Endpunkt 15:** `books`-Gegenpol bauen (Stimuli + Achse, Neulauf J) oder 15 als explorativ
   erklären.
10. **Endpunkt 17:** TTT-Korpora liefern; Memorisation neu kalibrieren.
11. **Kleinere Punkte:** `review.csv`-Regel implementieren oder streichen (Punkt 7);
    `data/s4/axes.npz` für ein ARI-orthogonales `ent` (Punkt 24); Bank nur auf DEV fitten
    (Punkt 2).
