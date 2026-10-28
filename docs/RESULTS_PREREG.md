# Präregistrierte Auswertung nach den Neuläufen (Notebook 13, 27./28.10.2026)

Diese Datei fasst die präregistrierte Auswertung (`lsne.prereg_eval`, Notebook 13) zusammen. Sie stützt
sich auf die Läufe von Lauf 1 (`docs/RESULTS_RUN1.md`) und die Neuläufe nach den Korrekturen vom
27.10.2026. Maßgeblich ist die Präregistrierung in `docs/PREREGISTRATION.md` (Stand `58ec270`) mit den
datierten Nachträgen in Abschnitt 6, zuletzt 6.11 vom 28.10.2026 (Kontrolle der Sprache an der
Injektionsstelle, festgelegt nach dem Bericht von 07:07 UTC; der Kontrolllauf steht aus, §2.2).

Quellen auf Drive (Konto A, `MyDrive/LSNE/prereg_eval/`):

* `pythia-160m@step98000/`: maßgebliche Auswertung vom 28.10.2026, 07:07 UTC (Code `e0e7481`). Vorher:
  04:55 UTC nach dem Ende von Notebook 12 (Notebook 13 hatte 360 Minuten auf Notebook 12 gewartet) und
  06:42 UTC, versehentlich mit dem alten Code (Setup-Fehler, §9); beide mit den Regeln vor dem 28.10.
* `pythia-160m@step99000/`: Auswertung vom 27.10.2026, 19:22 UTC. Bei 99k gibt es nur Stufe H.

**Stand der Regeln.** Der Bericht von 07:07 UTC wendet die beiden Festlegungen vom 28.10.2026 an:
Nachtrag 6.10 (Monotonie in K) und die Einordnung der Dosis 0,5 bei Endpunkt 14 als explorativ
(PREREGISTRATION §1). Gegenüber 04:55 UTC sind Schätzer und Intervalle unverändert; geändert haben sich die
Urteile des Primärendpunkts, die BH-q-Werte (m = 26 statt 28) und die Rolle der Zeilen zu 14 bei ε 0,5;
neu sind die Zeilen `S14abs`. Alle Zahlen dieser Datei stimmen mit dem Bericht überein (q-Werte bis auf
Rundung identisch).

Begriffe wie in `docs/RESULTS_RUN1.md`: **gestützt**, **nicht gestützt**, **offen**, **nicht getestet**,
**nicht interpretierbar**, **Gate verfehlt**. Soweit nicht anders vermerkt, sind Intervalle
95 %-Perzentilintervalle eines Familien-Cluster-Bootstraps (10 000 Ziehungen) auf F01–F12 × {A, B}.
Ausnahmen: 13 (4 Kultur/Politik-Familien), 14 (12 Familien der Bewertungshälfte, disjunkt zur
Auswahlhälfte), 8/9 (Formulierungen A–D gepoolt), 10 (ii) „wie gelaufen“ (5 Familien), Spurenmodell §5.1
(Sonde und Alignment: 1 000 Ziehungen über Spurpaare; Transplantation: 24 Cluster).

## Kurzfassung

* **Primärendpunkt: offen** bei allen drei Dosen (Nachtrag 6.10).
  * In Zelle C1 steigt λ(`lang_dom`, K = 8) gegenüber K = 0 um 0,094–0,099. Die untere Intervallgrenze liegt
    bei 0,069–0,074, über dem Mindesteffekt 0,05. Gegenüber Zufallsrichtungen bleibt ein Unterschied von
    0,102–0,109 (Intervalle über 0).
  * Fast der ganze Effekt liegt im Schritt K = 0 → 2. Injiziert wird bei K = 0 und K > 0 in dasselbe Token,
    den Punkt „.“ am Ende von Einheit 9; bei K > 0 schließt er aber einen deutschen Satzteil ab, und der
    letzte Wechsel D → E folgt direkt danach. Ob der Sprachwechsel im Text wirkt oder die Sprache an der
    Injektionsstelle, prüft die Kontrolle aus Nachtrag 6.11 (Notebook 14, Lauf ausstehend; §2.2, §8).
  * Ein weiterer Anstieg mit K ist nicht nachweisbar, aber auch nicht ausgeschlossen: Die Zuwächse pro Schritt
    haben Intervalle von etwa −0,03 bis +0,03. Damit ist „monoton in K“ weder gezeigt noch widerlegt, und
    H1 bleibt **offen**.
* **Sekundär gestützt** (BH über 26 Tests): 1 (Wechsel senkt log p des Ziels); 4 (misst großteils
  Kontextgedächtnis); 5 nur mit `nmi_last` (Auslesung nicht präregistriert, `nmi_tail` offen);
  10 (i)–(iii) (Pfade sättigen ab λ_rel 0,3; 10 (ii) mit 17 von 24 konstanten Reihen; 10 (iii) teils
  konstruktionsbedingt). 10 (i) ist neu: Das SAE ist bei 98k zertifiziert.
* **Gemischt:** 3. Ein Teiltest ist gestützt, einer liegt signifikant gegen die erwartete Richtung, sechs
  sind offen.
* **Nicht gestützt:** 6 (Vorzeichen umgekehrt), 13. **Offen:** 2 und 14 bei der präregistrierten Dosis
  0,1. Bei gleicher absoluter Norm unterscheiden sich Head und Residuum gar nicht.
* **Gate G0 verfehlt** (0 von 13 Layern): 8 und 9 sind nicht interpretierbar.
* **Übernommene Endpunkte:** 10f gestützt (Vorbehalt Sättigung). 11 offen (Dokumentation). 12 und 16 sind
  Dokumentation; die Erwartung ist nicht erfüllt. 15 nicht getestet. 17 (i), (iii) nicht interpretierbar
  (Platzhalter-Korpus), 17 (ii) nicht getestet. 18 (Stand-in) gestützt.
* **18T nicht gestützt:** Der Spurenmodell-Lauf ist gültig und zählt. Ein Sharing-Effekt wird nicht
  beansprucht, weil die Sonde am eingefrorenen Layer und das Alignment-Kriterium verfehlt sind. Sharing
  selbst wurde ohne SAE nicht gemessen; im Modus „both“ steigt T gegen alle Kontrollen.
* **Injektions-Audits gültig:** D max 3,9e-06, C max 4,0e-05 (Grenze 1e-4). Andere Stufen prüft die
  Auswertung nicht.

## 1. Datengrundlage

| Stufe | Konto | Lauf | Stand | Notebook |
|---|---|---|---|---|
| C | A | Lauf 1, 26.10. 15:51 UTC | `a018239` | 02 |
| D | A | Lauf 1, 27.10. 00:45 UTC | `76a4a5a` | 03 |
| G | A | Neulauf, 27.10. 16:30 UTC | `97d4104` (Code) | 06 |
| H | B | Neulauf mit SAE, 27.10. 16:59 UTC | `97d4104` (Code) | 07 |
| I | A | Neulauf, 27.10. 16:26 UTC | `97d4104` (Code) | 08 |
| J | B | Neulauf mit SAE, 27.10. 17:08 UTC | `97d4104` (Code) | 09 |
| K | B | Lauf 1, 26.10. 21:27 UTC | `76a4a5a` | 10 |
| L (Stand-in) | A | Neulauf, 27.10. 16:49 UTC | `97d4104` (Code) | 11 |
| L (Spurenmodell) | A | Neulauf, 28.10. 04:50 UTC | `97d4104` (Code) | 12 |
| H bei 99k | A | Neulauf, 27.10. 16:42 UTC | `97d4104` (Code) | 07 |

Die Manifeste der Neuläufe tragen die Commits `7c61580` bzw. `9981ab1`. Diese ändern gegenüber `97d4104`
nur Dokumente. Die alten G- und I-Läufe auf Konto B hat die Auswertung als älter verworfen (es gilt der
neueste Lauf). Die H- und J-Läufe von Lauf 1 wurden vor den Neuläufen auf Konto B in `…@step98000.run1`
umbenannt. H und J liefen mit `--use-sae` (laut `summary_07`/`summary_09`).

Das SAE (`EleutherAI/sae-pythia-160m-32k`, `layers.10`) rekonstruiert die Aktivierungen bei `step98000` mit
FVU = 0,198 (Grenze 0,4) und ist damit dort zertifiziert (Notebook 01 auf Konto B, 27.10. 16:37 UTC).

## 2. Primärendpunkt (PREREGISTRATION §2)

Δλ = λ(K, `lang_dom`) − λ(K = 0, `lang_dom`), Zelle C1, Tiefen-Injektion, K = 8; DiD = Δλ(`lang_dom`) −
Δλ(`rand`, Mittel über 32 Zufallsrichtungen).

| Dosis ε | Δλ `lang_dom` (K = 8) | Δλ `rand` | DiD | Δλ bei K = 2 / 4 / 8 | P(monoton) im Bootstrap | Urteil nach 6.10 | nur Punktschätzer (a) |
|---|---|---|---|---|---|---|---|
| 0,01 | 0,0949 [0,0705; 0,1183] | −0,0070 [−0,0260; 0,0106] | 0,1019 [0,0875; 0,1174] | 0,0915 / 0,1010 / 0,0949 | 0,147 | **offen** | nicht gestützt |
| 0,05 | 0,0993 [0,0736; 0,1248] | −0,0101 [−0,0293; 0,0079] | 0,1094 [0,0910; 0,1289] | 0,0915 / 0,0993 / 0,0993 | 0,258 | **offen** | gestützt |
| 0,1 | 0,0941 [0,0690; 0,1184] | −0,0105 [−0,0296; 0,0077] | 0,1046 [0,0856; 0,1246] | 0,0882 / 0,0918 / 0,0941 | 0,236 | **offen** | gestützt |

Die letzte Spalte zeigt die Regel vor Nachtrag 6.10, wie sie die Berichte von 04:55 und 06:42 UTC
auswiesen.

Zuwächse zwischen den K-Stufen (Δλ `lang_dom`):

| Dosis ε | K 0 → 2 | K 2 → 4 | K 4 → 8 |
|---|---|---|---|
| 0,01 | 0,0915 [0,0646; 0,1170] | 0,0095 [−0,0129; 0,0277] | −0,0061 [−0,0274; 0,0204] |
| 0,05 | 0,0915 [0,0646; 0,1169] | 0,0078 [−0,0164; 0,0281] | 0,0001 [−0,0228; 0,0287] |
| 0,1 | 0,0882 [0,0613; 0,1130] | 0,0036 [−0,0189; 0,0236] | 0,0023 [−0,0187; 0,0267] |

Der Schritt 0 → 2 ist bei jeder Dosis belegt, die späteren Schritte weder belegt noch widerlegt. Einseitige
p-Werte für Δλ und DiD: ≤ 0,0003 nach Bonferroni über die drei Dosen. Das ist die Auflösungsgrenze von
10 000 Ziehungen, kein exakter Wert.

### 2.1 Was die Zahlen zeigen

* **Effekt in C1:** Bei jeder Dosis liegt Δλ sicher über dem Mindesteffekt. Zufallsrichtungen zeigen in C1
  keinen Effekt (Δλ `rand` ≈ −0,01, Intervall um 0). Der Effekt ist also größer als bei Zufallsrichtungen.
  Ob ihn der Sprachwechsel im Text verursacht oder die Sprache des Satzteils an der Injektionsstelle (das
  Token selbst ist dasselbe „.“), trennen die Daten nicht; das prüft der Kontrolllauf nach Nachtrag 6.11
  (§2.2).
* **Dosis-Wirkung in K:** nicht gezeigt. Der Sprung passiert zwischen K = 0 und K = 2. Danach bleibt Δλ
  innerhalb der Unsicherheit gleich; ein weiterer Anstieg von bis zu etwa 0,03 pro Schritt ist mit den Daten
  noch vereinbar.
* **Zellen außerhalb von C1** (C2, C3, seg): Nach Nachtrag 6.2 sind das Abweichungszeilen; sie werden nicht
  interpretiert. Deskriptiv: Dort steigt Δλ `rand` ebenso wie Δλ `lang_dom` auf 0,13–0,16, die DiD liegt bei
  −0,014 bis +0,001. Der Überschuss gegenüber Zufall aus C1 zeigt sich bei Injektion am letzten deutschen
  Token also nicht. Ob das an der Position oder an der Bedingung liegt, trennt erst eine positionsgleiche
  K = 0-Referenz.

### 2.2 Vorbehalte

* In C1 ist das Token an der Injektionsstelle bei K = 0 und K > 0 dasselbe: der Punkt „.“ am Ende von
  Einheit 9. Die Marke D bei K > 0 kommt nur daher, dass Satzzeichen dem vorangehenden Segment zugeordnet
  werden. Verschieden sind die Sprache des Satzteils, den der Punkt abschließt (bei K > 0 deutsch), und der
  letzte Wechsel D → E: Er folgt bei K > 0 direkt auf die Stelle und liegt damit zwischen ihr und dem
  englischen Schlussabschnitt. Die Zufallsrichtungen schließen aus, dass dieser Unterschied λ für beliebige
  Richtungen hebt. Sie schließen nicht aus, dass die Sprachachse nach einem deutschen Satzteil anders wirkt
  als nach einem englischen. Die Kontrolle dazu legt Nachtrag 6.11 fest (Notebook 14, Zeilen `P.ctl`; Lauf
  ausstehend).
* Die `lang_dom`-Achse an der Tiefen-Injektionsstelle (Layer 4) verfehlt ihren Entrauschungs-Test
  (cos_denoised_plain 0,126 < 0,70). Sie bewahrt dort wenig vom einfachen D-E-Mittelwertunterschied.
* `lang_dom` ist auf allen Standardvarianten gefittet, die Testfamilien eingeschlossen.
* Die Urteile beruhen auf unkorrigierten 95 %-Intervallen je Dosis. Die Mehrfachprüfung über drei Dosen ist
  darin nicht kontrolliert; korrigiert sind nur die p-Werte.
* Gerechnet wurden nur die Dosen 0,01 / 0,05 / 0,1; 0,001 / 0,0025 / 0,005 / 0,02 fehlen (Nachtrag 6.3).

### 2.3 Festlegung: „monoton in K“ (28.10.2026, Nachtrag 6.10)

PREREGISTRATION §2 verlangt „Δλ > 0 und monoton in K, und größer als derselbe Kontrast mit `rand`“, legt
aber nicht fest, wie Monotonie geprüft wird. Erwogen wurden drei Lesarten:

| Lesart | Ergebnis mit diesen Daten |
|---|---|
| (a) Punktschätzer streng steigend (Code bis 28.10.) | 2 von 3 Dosen gestützt; hängt am Rauschen |
| **(b) Anstieg mit Intervall belegt (festgelegt)** | bei keiner Dosis belegt, bei keiner widerlegt → **offen** |
| (c) kein belegter Abfall | bei allen 3 Dosen erfüllt |

Festgelegt ist (b), die strengste Lesart, weil die Festlegung erst nach Sichtung der Daten erfolgt. Für den
unentschiedenen Fall gilt die Regel aus §2: Ein Intervall, das weder belegt noch ausschließt, ergibt
„offen“, keinen Nullbefund. „Nicht gestützt“ gäbe es nur, wenn ein Schritt in K ein Intervall unter 0
hätte. Ergebnis: **Effekt in C1 über dem Mindesteffekt und über Zufall; Anstieg in K unentschieden;
Primärendpunkt offen.**

## 3. Sekundäre Endpunkte (BH-Gruppe, m = 26)

| Endpunkt | Schätzer [95 %-KI] | q (BH) | Urteil | Hinweis |
|---|---|---|---|---|
| 1: T(K = 8, C1) | −0,3438 [−0,5560; −0,1608] | 0,0013 | **gestützt** | Wechsel senkt log p des Ziels |
| 2: T(8, C3) − T(8, C1) | 0,0965 [0,0016; 0,2001] | 0,148 | **offen** | |
| 3: R_excess, K0 vs K28seg, Layer 2 | 0,0514 [0,0132; 0,0915] | 0,013 | **gestützt** | |
| 3: K0 vs K28seg, Layer 11 | −0,0564 [−0,1186; −0,0077] | 1,0 | **nicht gestützt** | gegen die erwartete Richtung |
| 3: übrige 6 Teiltests | | ≥ 0,20 | **offen** | Leerraum-Klasse leer: misst R_punct+boundary |
| 4: Hysteresefläche | 0,2935 [0,2757; 0,3097] | 0,0013 | **gestützt** | misst großteils Kontextgedächtnis |
| 5: nmi_last | 0,0182 [0,0063; 0,0297] | 0,010 | **gestützt** | Auslesung nicht präregistriert |
| 5: nmi_tail | −0,0054 [−0,0111; 0,0011] | 0,227 | **offen** | |
| 6: Transfer `ent` (+1), alle 7 Dosen | z. B. ε 0,05: −0,2343 [−0,2989; −0,1734] | 1,0 | **nicht gestützt** | Vorzeichen umgekehrt; Kontrolle −1 positiv |
| 10 (i): ρ(λ, SAE-FVU) | −0,8006 [−0,8914; −0,7039] | 0,0013 | **gestützt** | neu: SAE bei 98k zertifiziert |
| 10 (ii): ρ(λ, Schritte bis Grenze) | 0,1865 [0,0510; 0,3299] | 0,010 | **gestützt** | 17 von 24 Reihen konstant (ρ = 0) |
| 10 (iii): ρ(λ, Mahalanobis) | −0,5208 [−0,7024; −0,3318] | 0,0013 | **gestützt** | teils konstruktionsbedingt |
| 13: Gegensteuern `dom_law`, ε 0,05 | −0,0409 [−0,0719; −0,0098] | 1,0 | **nicht gestützt** | 4 Familien |
| 14: Head L11H10 − Residuum, ε 0,1, cult./pol. | 0,1053 [−0,0413; 0,2592] | 0,322 | **offen** | |
| 14: dasselbe, `lang_dom` | 0,0884 [−0,0020; 0,1760] | 0,161 | **offen** | |

Hinweise:

* **4:** Der absteigende Ast hängt am deutschen Text, den der aufsteigende erzeugt hat; die Neustarts tun das
  nicht. Die Fläche misst daher großteils autoregressives Kontextgedächtnis. Sie ist ohne Wechsel am größten
  (K = 0: 0,4380 [0,4078; 0,4701]; K = 8: 0,1996; K = 28: 0,2427).
* **5:** Welche Auslesung gilt, legt PREREGISTRATION 5 nicht fest; beide stehen in der BH-Gruppe. Lauf 1 hatte
  5 als „offen“ geführt.
* **10:** Ab λ_rel 0,3 sättigen die Pfade; ρ vergleicht daher vor allem λ = 0 mit λ > 0. Bei 10 (ii) ist die
  Grenze von +1 nat für 17 von 24 Teststarts unerreichbar. Ohne die konstanten Reihen („wie gelaufen“)
  ergibt sich 0,6313 [0,6124; 0,6692] auf 5 Familien.
* **14:** Präregistriert ist nur ε 0,1. Die Dosis 0,5 ist nach PREREGISTRATION §1 explorativ und steht nicht
  in der BH-Gruppe. Dort ist die Differenz bei gleicher relativer Dosis groß (cult./pol. 1,8829
  [1,0761; 2,8026], `lang_dom` 3,1819 [2,3780; 3,9455]). Bei gleicher relativer Dosis schreibt das Residuum
  aber im Median 36-mal so viel (RMS) wie der Head, die Differenz misst also vor allem den Schaden des
  Residuum-Arms. Bei gleicher absoluter Norm (Zeilen `S14abs`) verschwindet sie bei beiden Dosen, z. B.
  `lang_dom`, ε 0,1: 0,0008 [−0,0013; 0,0030]; ε 0,5: 0,0042 [−0,0062; 0,0152].
* **Sensitivität (Bonferroni je Endpunkt, BH über 9 Endpunkte):** 1, 4, 5 und 10 bleiben unter q 0,05, 3 knapp
  (0,049); 2 liegt bei 0,068, 14 bei 0,143, 6 und 13 bei 1,0.

## 4. Gate G0 und Endpunkte 8–9

G0 verfehlt: In keinem der 5 mittleren Layer (0 von 13 insgesamt) gilt die Ordnung
d(A_E, A_D) < d(A_E, B_E) < d(andere Familie). Nach PREREGISTRATION 7 werden 8 (G1-Kontraktion) und 9
(T3-Steuerung) nicht interpretiert. Ohne Gate wäre 8 in 17 von 26 Zeilen gestützt (ohne BH) und 9 nicht
gestützt (cos 0,20 bzw. 0,08 statt ≥ 0,3); beides wird nicht interpretiert.

## 5. Übernommene Endpunkte 10f–18

| Endpunkt | Ergebnis | Urteil |
|---|---|---|
| 10f: SAE-Features, auf DEV eingefroren | 1 606 Kandidaten, 443 auf Test bestätigt, 435 nach BH | **gestützt** (Vorbehalt: Sättigung wie bei 10) |
| 11: B bei λ_rel 3 gegen λ 0 (State 2143) | 98k: 13,60 → 1,94; 99k: 14,43 → 2,29; Vorzeichen gleich | **offen** (Dokumentation, keine Inferenz) |
| 12: Domänenkompass | Lexikon E: Spitze Religion, Recht Rang 10 von 11; Lexikon D: Spitze Familie, Recht Rang 4 | Dokumentation; Erwartung (Recht an der Spitze) nicht erfüllt |
| 15: RDX-Anteil gegen `books` | Gegenpol nicht berechnet | **nicht getestet** (Nachtrag 6.9) |
| 15 mit `misfolding`-Gegenpol | z ≥ 2 an 4 von 4 Stellen, erklärter Anteil 0,086 | als Abweichungszeile „gestützt“; nur explorativ (Nachtrag 6.9) |
| 16: Pile-Schwierigkeit `protein_misfolding` | Rang 9 von 11, am schwersten `books` | Dokumentation; nicht im schwersten Drittel, daher Gegenpol `books` |
| 17 (i), (iii): Test-Time-Training | Platzhalter-Korpus | **nicht interpretierbar** |
| 17 (ii): Sharing / Aktivität | braucht ein SAE | **nicht getestet** |
| 18: Stand-in-Validierung (pythia-70m) | bester Layer 5: cos 0,973, R²-Abstand 1,082, 4 von 4 Layern | **gestützt** |
| 18T: Spurenmodell als Quelle | gültiger Lauf, kein Anspruch (Sonde, Alignment verfehlt; §5.1) | **nicht gestützt** |

### 5.1 Spurenmodell (Notebook 12, Endpunkt 18T)

Der Lauf ist gültig (`counts_for_prereg = true`). Alle Schritte endeten ok; der Download wurde aus dem ersten
Lauf wiederverwendet. Die Stand-in-Validierung ist bestanden, das eingefrorene Merkmal blieb unverändert,
und das Korpus ist gesund.

* **Daten:** 13,53 Mio. Trainings-Transaktionen (Ethereum 2016), Vokabular 77 364 Token. Adress-OOV 0,155 im
  Training (Lauf 1: 0,967) und 0,192 im Hold-out.
* **Modell:** `blockgpt`, 104 Mio. Parameter, 20 000 Schritte, 328 Mio. Tokens. Validierungsverlust 0,410
  nats pro Token gegen 1,386 (Bigramm) und 11,34 (untrainiert).
* **Sonde** am eingefrorenen letzten Layer (Fan-out k = 5, längengematcht): AUC trainiert 0,986 gegen
  untrainiert 0,973; Differenz 0,013 [−0,003; 0,083] → Kriterium verfehlt.
* **Alignment mit Pythia:** besser als permutiert (R² 0,32 gegen −0,42), aber nicht besser als der
  untrainierte Zwilling (residualisierter Gewinn −0,005, untere Grenze −0,035). Ein reines Längenmodell
  erreicht R² 0,94.
* **Explorativ** (nicht eingefroren, viele Vergleiche): Bei k = 5 erfüllen die Layer 3–7 das
  Sondenkriterium. Das residualisierte Alignment schlägt den untrainierten Zwilling in 45 von 117
  Layer-Zellen (α = 1). Das begründet keinen Anspruch.
* **Transplantation** in Pythia (Dosis 1,0):
  * Residuum-Modus: B ändert sich gegen das unveränderte Modell um −11,05 [−11,77; −10,33], statt zu
    steigen. Zufalls- und permutierte Kontrollen schaden ähnlich (−9,53 bzw. −13,84).
  * kv-Modus: kein Effekt in der vorgegebenen Richtung (B −0,05 [−0,09; −0,02]).
  * Modus „both“: T steigt gegen alle Kontrollen (+0,30 [0,14; 0,45] gegen das unveränderte Modell); B
    ändert sich um −11,40.
* **Urteil:** Kein Anspruch, weil Sonde und Alignment-Kriterium verfehlt sind. Sharing-Index und
  Netzwerkaktivität wurden ohne SAE nicht berechnet.

Laufzeiten gegen Plan (Stunden):

| Schritt | Ist | Plan |
|---|---|---|
| Serialisierung | 7,1 | 0,25–0,7 |
| Split | 0,47 | 0,08–0,4 |
| Vokabular | 0,92 | 0,17–0,5 |
| Training | 1,54 | 0,8–1,5 |
| Evaluation | 0,92 | 0,5–0,75 |
| Alignment | 0,09 | 0,10–0,35 |

Die Serialisierung dauerte so lange wegen der Chunks der DoS-Angriffe vom Herbst 2016; laut Ausgabe von
Notebook 12 haben sie bis zu 58 Mio. Roh-Tokens pro Chunk. Hold-out-OOV, Sonden, Paare und Stufe L lagen
im Plan.

## 6. Ergebnisse bei 99k (nur Stufe H)

| Endpunkt | 99k | 98k |
|---|---|---|
| 10 (ii) | 0,2454 [0,0844; 0,4240], **gestützt** | 0,1865 [0,0510; 0,3299], **gestützt** |
| 10 (iii) | −0,6726 [−0,8214; −0,5045], **gestützt** | −0,5208 [−0,7024; −0,3318], **gestützt** |
| 11 | B 14,43 → 2,29 | B 13,60 → 1,94 |

Die q-Werte bei 99k stammen aus einer BH-Gruppe mit nur 2 Zeilen (10 (ii), 10 (iii)) und sind mit denen bei
98k (m = 26) nicht vergleichbar. 10 (i) ist bei 99k nicht getestet, weil 07 bei 99k ohne SAE lief. Alle
anderen Endpunkte erscheinen bei 99k als „nicht getestet“, weil es dort nur Stufe H gibt.

## 7. Injektions-Audits (PREREGISTRATION §4)

| Stufe | Zeilen | größter relativer Fehler | Zeilen über 1e-4 | Urteil |
|---|---|---|---|---|
| D (Schadensausbreitung) | 471 648 | 3,9e-06 | 0 | gültig |
| C (Raster) | 647 360 | 4,0e-05 | 0 | gültig |

## 8. Offene Entscheidungen

1. ~~Primärendpunkt, Monotonie~~: entschieden am 28.10.2026 (Nachtrag 6.10, §2.3).
2. **Primärendpunkt, Achse:** ob die `lang_dom`-Achse trotz verfehltem Entrauschungs-Test an Layer 4 gilt.
3. ~~Primärendpunkt, Tokensprache~~: festgelegt am 28.10.2026 durch Nachtrag 6.11 (Commit `2ce9c71`), vor dem
   Kontrolllauf: Kontrolle der Sprache an der Injektionsstelle mit einem eigenen D1-Lauf (Notebook 14); ohne
   gültigen Kontrolllauf ist der Primärendpunkt höchstens „offen“. Der Lauf steht aus.
4. **Endpunkt 14:** welcher der beiden vorhandenen Kontraste (gleiche relative Dosis oder gleiche absolute
   Norm) konfirmatorisch gilt.
5. **Endpunkt 15:** `books`-Gegenpol bauen (Neulauf J) oder 15 explorativ lassen.
6. **Endpunkt 17:** TTT-Korpora liefern.
7. Die übrigen Punkte aus `docs/RESULTS_RUN1.md` §5.5 (Endpunkt 10 neu kalibrieren, Gegensteuer-Arm von 13,
   fehlende Dosen in D).

## 9. Laufprotokoll 27./28.10.2026

* Neuläufe nach dem Korrektur-Commit `97d4104`: 06, 08, 11 und 07 (99k) auf Konto A; 01, 07 (98k) und 09
  mit SAE auf Konto B; 12 auf Konto A; 13 bei 98k vor und nach 12 sowie bei 99k.
* `REUSE = True` lud zweimal einen alten Lauf, statt neu zu rechnen (01 ohne SAE, 11 im alten Design). Beide
  wurden mit `REUSE = False` wiederholt.
* Die erste Verknüpfung `LSNE_B` war eine Windows-Linkdatei und wurde durch eine Drive-Verknüpfung ersetzt.
* Notebook 13 wartet seit `dfb473b` auf ein laufendes Notebook 12 (`WAIT_FOR_12 = "auto"`).
* Der Lauf von Notebook 13 um 06:42 UTC nutzte die wiederverwendete Colab-Laufzeit der Nacht. Die
  Setup-Zelle brachte den flachen Klon nicht auf den neuen Stand (`pull --ff-only` scheitert bei
  Tiefe 1, der Fehler wurde ignoriert), und schon geladene Module überdeckten den Code. Behoben in
  `1e346d8`: Die Zelle setzt den Zweig auf `FETCH_HEAD`, bricht bei Fehlern ab und lädt `lsne` neu.
* Die Vorbehaltstexte zu 18 und 18T beschrieben bis zum Bericht von 06:42 UTC das Design vor den
  Korrekturen; seit dem Bericht von 07:07 UTC sind sie aktuell.
* Nachtrag 6.11 (Kontrolle der Sprache an der Injektionsstelle) ist seit dem 28.10.2026, 08:05 UTC committet
  (`2ce9c71`), also vor dem Kontrolllauf mit Notebook 14, der noch aussteht.
