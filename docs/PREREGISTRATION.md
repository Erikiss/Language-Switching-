# Präregistrierung LS-NEQ (Entwurf zum Einfrieren vor Stufe C/D)

Stand 25.10.2026. Dieses Dokument wird vor dem ersten Bestätigungslauf eingefroren
(Git-Tag `prereg-v1`). Änderungen danach sind explorativ und werden als solche markiert.

## 1. Fixierte Identitäten

- Anker: Sequenz 5 025 183 aus `EleutherAI/pile-standard-pythia-preshuffled`;
  SHA256 der 2049 Tokens `bdd4bbdd…55d4`; SHA256 der 208 Kontext-Tokens `84c8f1b6…a750`;
  Ziel 22905. Abweichung ⇒ Lauf ungültig.
- Modelle/Revisionen: siehe `docs/DESIGN.md` §7. Standarddatentyp FP32.
- Eingriffsstelle: Block-Eingang Layer 11, letzte Kontextposition; δ = ε·RMS(h)·g/RMS(g).
- Dosen: ε ∈ {0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1}. Für Steering-Richtungen
  (`lang_dom`, `sae_f`) zusätzlich ε ∈ {0.2, 0.5, 1.0} (explorativ).
- Split: Entwicklungsfamilien `DEV01–DEV04` (Dosis-, Layer- und Schwellenwahl);
  Testfamilien `F01–F12` × Formulierung {A, B} (Bestätigung). Original-2143 separat.

## 2. Primärer Endpunkt

λ(K, ε, dir): Wachstumsexponent der Störungsausbreitung (D1) über die Tiefe,
Fit von ln D(ℓ, t) auf ℓ ∈ [ℓ₀+1, 12], gemittelt über die Positionen des gemeinsamen
englischen Schlussabschnitts; Injektion an (ℓ₀ = 11 → ersetzt durch ℓ₀ = 4 für den
Tiefenfit, Dossier-Stelle ℓ₀ = 11 zusätzlich für die Positionsausbreitung),
Position = letztes Token der letzten gewechselten Einheit (Einheit 9).

**H1:** λ(K>0, ε, `lang_dom`) − λ(K=0, ε, `lang_dom`) > 0 und monoton in K, und
größer als derselbe Kontrast mit `rand` (Mittel über 32 Richtungen).
**H0:** kein Unterschied.

Teststatistik: Familienmittel der Differenz; 95 %-Konfidenzintervall per
Cluster-Bootstrap (10 000 Ziehungen, Cluster = Familie, beide Formulierungen und
alle Masken innerhalb der Familie gemeinsam gezogen). Mindesteffekt (praktisch
relevant): Δλ ≥ 0,05 pro Layer. Ein Intervall, das Null einschließt und die
Mindesteffektgrenze nicht ausschließt, gilt als „offen“, nicht als Nullbefund.

## 3. Sekundäre Endpunkte (vorab festgelegt, BH-korrigiert über die Gruppe)

1. T(K) auf Testfamilien (Texteffekt), Kontrast K=8 vs K=0 bei C1.
2. Wechselwirkung K × C: T(8, C3) − T(8, C1).
3. Counterfactual-Recovery R_ws+punct+boundary − R_random auf dem Layer-Raster.
4. Hysteresefläche A_hyst(`lang_dom`) − A_hyst(Neustart-Null).
5. Nichtmonotonie-Index (D2) bei K=8 vs K=0.
6. Transfer: B(x) mit `ent` (aus State 2143) vs. 32 Zufallsrichtungen auf Testfamilien.
7. Geometrie-Gate G0 (kein Hypothesentest, Voraussetzung): unter trunkierter
   Mahalanobis-Distanz, Mean-Pool, Deflation k=5, mittlere Layer: d(A_E, A_D) <
   d(A_E, B_E) < d(Fremdfamilie) in ≥ 50 % der Layer; sonst werden 8–9 nicht interpretiert.
   Stichprobenhinweis: Bei N gepoolten Zuständen wird der Mahalanobis-Rang auf
   min(128, (N−1)//4) begrenzt; bei einer entarteten Metrik (r ≥ N−1) gibt das Gate kein
   Urteil ab.
8. Kontraktion unter Sprachwechsel G1: Δ log-Vol(K=8, C1) − Δ log-Vol(K=0) (Mean-Pool,
   k=5) und dasselbe für PR (Last-Token); zweiseitig, Vorzeichen offen.
9. Steering-Validität T3: Kosinus zwischen der Steuerverschiebung (`lang_dom`, ε = 0,05)
   und mean(K=8) − mean(K=0) pro Layer, gegen 32 Zufallsrichtungen; Mindesteffekt
   cos ≥ 0,3 im Layer mit maximaler diskreter Verschiebung.

10. λ-Dynamik (Stufe H), Diskriminante (b) −log p(Ziel) an der Dossier-Stelle,
    Zielklassen-Dichte aus K=8/C1-Zuständen: Spearman-ρ zwischen λ und (i) dem
    SAE-Rekonstruktionsfehler am Endpunkt (Erwartung ρ < 0: Mimikry-Endpunkte sind
    besser rekonstruierbar), (ii) der Zahl der Schritte bis zur Grenzüberquerung (nie querende Pfade
    werden bei n_steps+1 zensiert, nicht verworfen),
    (iii) dem Mahalanobis-Abstand zur Zielwolke (Erwartung ρ < 0). Features mit
    |ρ| ≥ 0,7 über den Sweep auf Entwicklungsfamilien werden als Kandidaten eingefroren
    und auf Testfamilien geprüft.
11. B(x)(λ) für State 2143 bei 98k/99k: Vorzeichen des Nutzens bei λ_rel = 3 gegenüber
    λ = 0 (zweiseitig; Erwartung offen).

12. Domänen-Kompass (Stufe I): Domänenprofil der State-2143-Richtung `ent` am Zustand
    h₂₀₇ (Logit-Lens-Proxy, z-Werte gegen 32 Zufallsrichtungen); Bericht, ob `law` das
    höchste z erhält (Atlas-Erwartung). Kein Test, Dokumentation.
13. Richtung × Domäne: T(K=8, C1) und B(x) für `dom_culture_politics` (nach
    Gegensteuern gegen `dom_law`) vs. `dom_law` vs. `lang_dom`, getrennt nach
    Familiendomäne (Kultur/Politik F13–F16, Jura F17–F18, Alltag F01–F12);
    vorab festgelegter Kontrast: B(dom_culture_politics) − B(lang_dom) auf
    Kultur/Politik-Familien, Familien-Bootstrap, Mindesteffekt 0,05 nats.
14. Head-Arm: B bei Steering am schlechtesten Head (Auswahl auf Entwicklungsfamilien
    eingefroren) vs. am Residualstrom bei gleicher relativer Dosis; zweiseitig.
    Ohne eingefrorene Vorgabe wird der Head auf einer Hälfte der Familien jeder
    Domänengruppe ausgewählt und auf der disjunkten anderen Hälfte getestet.

15. RDX-Anteil (Stufe J): Bruchteil von ‖Δ‖² der Differenz Kultur/Politik ↔
    Proteinfehlfaltung, der durch die Domänenachsen erklärt wird, gegen 32 Zufallsbasen
    gleicher Größe; Mindesteffekt z ≥ 2 in mindestens der Hälfte der Layer.
16. Empirische Schwierigkeitsrangfolge: mittlere Token-NLL pro Pile-Teilgebiet aus
    N ≥ 2000 gezogenen Sequenzen; Bericht des Rangs von `protein_misfolding`. Liegt es
    nicht im schwersten Drittel, wird der Gegenpol durch das tatsächlich schwerste
    Teilgebiet ersetzt und das in der Auswertung vermerkt.
17. Test-Time-Training (Stufe K): (i) Memorisationsgeschwindigkeit der injizierten
    Textstücke gegen angepasste Kontrolltexte (Schritte bis zur NLL-Schwelle,
    zweiseitig); (ii) Δ Sharing-Index und Δ Netzwerkaktivität nach TTT gegenüber den
    drei Kontrollarmen; (iii) Δ T(K=8) und Δ B(x) nach TTT. Alle drei nur auf
    Entwicklungsfamilien explorativ, Bestätigung auf Testfamilien mit eingefrorener
    Schrittzahl und Lernrate.

18. Transplantation (Stufe L), Stand-in-Validierung: Alignment-R² auf Hold-out-Paaren
    gegen permutierte Paare (Mindestabstand 0,2); Kosinus zwischen transferierter und
    nativer Sprachachse in Pythia ≥ 0,3 im besten Layer; Vorzeichengleichheit der
    Verschiebung des Sprachanteils in generiertem Text für Residual-Substitution und
    KV-Pflanzung gegenüber der nativen Achse. Die Validierung verwendet ein vom Ziel verschiedenes Quellmodell
    (Standard Pythia-70M); das Zielmodell selbst oder eine Isometrie davon zählt nur als
    Funktionsprobe. Erst nach bestandener Validierung wird ein
    Spurenmodell als Quelle gewertet; dessen „Teilen“-Merkmal wird vorab im Quellmodell
    definiert und eingefroren.

## 4. Ausschlussregeln (vor Sichtung der Modellwerte)

- Variante ausgeschlossen, wenn Ziel-ID oder Tail-Token-IDs vom K=0-Fall abweichen
  oder das Zielwort im Kontext vorkommt.
- Variante ausgeschlossen, wenn die zweisprachige Prüfung (verblindet, Datei
  `data/stimuli/review.csv`) Bedeutungsabweichung markiert.
- Läufe mit Soll/Ist-Abweichung des geschriebenen Vektors > 1e-4 (relativ, FP32)
  werden als ungültig markiert.

## 5. Mehrfachtests und Exploration

Alle Phasendiagramme, Checkpoint-Sweeps, Skalierungsarme, SAE-Feature-Auswahl und
FDT-Verhältnisse sind explorativ. Positive explorative Befunde werden als neue
Hypothesen für eine spätere Bestätigung formuliert, nicht als Ergebnis dieses Tests.

## 6. Nachträge nach Lauf 1 (27.10.2026, nach Sichtung der Daten)

Der Tag `prereg-v1` wurde am 27.10.2026 **nachträglich** gesetzt, auf Commit `58ec270`
(26.10.2026, 00:41 UTC). Das ist die letzte Änderung an diesem Dokument vor dem ersten Lauf
(erster Lauf ab 26.10.2026, 12:45 UTC, Code-Stand `a018239`). Abschnitte 1–5 sind seitdem
unverändert.

Die folgenden Punkte wurden **nach** Sichtung von Lauf 1 festgelegt. Es sind Klarstellungen
von Lücken des Textes oben, keine Änderungen der Hypothesen. Ergebnisse, die von ihnen
abhängen, werden in der Auswertung als „nach Sichtung festgelegt“ gekennzeichnet. Umgesetzt
sind sie in `lsne/prereg_eval.py` (Notebook 13, Code-Stand `97d4104`); Begründungen und
Zahlen in `docs/RESULTS_RUN1.md` §4.

1. **Mindesteffekte** werden am 95-%-Intervall beurteilt: erfüllt, wenn die untere Grenze
   ≥ Mindesteffekt; ausgeschlossen, wenn die obere Grenze < Mindesteffekt; sonst „offen“.
2. **Primärendpunkt, Zelle:** konfirmatorisch ist C1. Nur dort liegt die Injektion genau an
   der präregistrierten Position (letztes Token von Einheit 9). Für C2/C3/seg hat der Code am
   letzten deutschen Token injiziert; diese Zeilen sind Abweichungen und werden nicht
   interpretiert.
3. **Primärendpunkt, Dosis:** In Lauf 1 wurde keine Dosis auf DEV eingefroren, und gerechnet
   wurden nur 0,01 / 0,05 / 0,1. Berichtet wird jede dieser Dosen einzeln, mit
   Bonferroni-Korrektur über die Dosen. Die Dosen 0,001 / 0,0025 / 0,005 / 0,02 fehlen
   (Abweichung).
4. **BH-Gruppe:** Endpunkte 1–6, 8 und 9 (nur bei bestandenem G0), 10, 13 und 14, eine Zeile
   je Teiltest. Die Gruppierung „Bonferroni je Endpunkt, dann BH über Endpunkte“ wird als
   Sensitivität mitberichtet.
5. **Endpunkt 3** wird einseitig in der Richtung aus DESIGN §4.1 getestet. Im gemeinsamen
   Suffix kommt keine Leerraum-Position vor, daher misst R_ws+punct+boundary in Lauf 1
   faktisch R_punct+boundary (Abweichung).
6. **Endpunkt 6** wird einseitig im State-2143-Vorzeichen (+1) getestet; das Vorzeichen −1
   ist Kontrolle.
7. **Endpunkt 10(ii):** Zensierte Reihen, die über alle λ konstant sind, gehen mit ρ = 0 ein
   statt verworfen zu werden; die Zeile „wie gelaufen“ bleibt als Sensitivität.
8. **Endpunkt 13:** präregistriert ist eine Dosis (auf DEV eingefroren, sonst 0,05).
9. **Endpunkt 15:** Der nach Regel 16 fällige Gegenpol `books` wurde in Lauf 1 nicht
   angewandt; Endpunkt 15 ist für Lauf 1 explorativ.

Noch offene Entscheidungen (Liste in `docs/RESULTS_RUN1.md` §5.5) werden, sobald getroffen,
hier mit Datum ergänzt.
