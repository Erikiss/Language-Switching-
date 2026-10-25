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
8. Kontraktion unter Sprachwechsel G1: Δ log-Vol(K=8, C1) − Δ log-Vol(K=0) (Mean-Pool,
   k=5) und dasselbe für PR (Last-Token); zweiseitig, Vorzeichen offen.
9. Steering-Validität T3: Kosinus zwischen der Steuerverschiebung (`lang_dom`, ε = 0,05)
   und mean(K=8) − mean(K=0) pro Layer, gegen 32 Zufallsrichtungen; Mindesteffekt
   cos ≥ 0,3 im Layer mit maximaler diskreter Verschiebung.

10. λ-Dynamik (Stufe H), Diskriminante (b) −log p(Ziel) an der Dossier-Stelle,
    Zielklassen-Dichte aus K=8/C1-Zuständen: Spearman-ρ zwischen λ und (i) dem
    SAE-Rekonstruktionsfehler am Endpunkt (Erwartung ρ < 0: Mimikry-Endpunkte sind
    besser rekonstruierbar), (ii) der Zahl der Schritte bis zur Grenzüberquerung,
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
