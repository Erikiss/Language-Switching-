# Präregistrierte Auswertung nach den Neuläufen (Notebook 13, 27./28.10.2026)

Diese Datei fasst die präregistrierte Auswertung (`lsne.prereg_eval`, Notebook 13) zusammen. Sie stützt
sich auf die Läufe von Lauf 1 (`docs/RESULTS_RUN1.md`) und die Neuläufe nach den Korrekturen vom
27.10.2026. Maßgeblich ist die Präregistrierung in `docs/PREREGISTRATION.md` (Stand `58ec270`) mit den
datierten Nachträgen in Abschnitt 6.

Quellen auf Drive (Konto A, `MyDrive/LSNE/prereg_eval/`):

* `pythia-160m@step98000/`: Auswertung vom 28.10.2026, 04:55 UTC, nach dem Ende von Notebook 12. Notebook 13
  hat 360 Minuten auf Notebook 12 gewartet (`summary_13_…_step98000.json`, `wait_for_12`).
* `pythia-160m@step99000/`: Auswertung vom 27.10.2026, 19:21 UTC. Bei 99k gibt es nur Stufe H.

Begriffe wie in `docs/RESULTS_RUN1.md`: **gestützt**, **nicht gestützt**, **offen**, **nicht getestet**,
**nicht interpretierbar**, **Gate verfehlt**. Alle Intervalle sind 95 %-Perzentilintervalle eines
Familien-Cluster-Bootstraps (10 000 Ziehungen) auf den Testfamilien F01–F12 × {A, B}.

## Kurzfassung

* **Primärendpunkt:** Sprachwechsel erhöht den Wachstumsexponenten λ bei allen drei Dosen um 0,094–0,099.
  Die untere Grenze des Intervalls liegt bei 0,069–0,074, also über dem Mindesteffekt 0,05. Gegenüber
  Zufallsrichtungen bleibt ein Unterschied von 0,102–0,109 (Intervalle klar über 0).
* **Aber:** H1 verlangt auch „monoton in K“. Der Effekt ist bei K = 2 schon voll da und wächst danach nicht
  weiter: Kein Zuwachs von K = 2 auf 4 oder von 4 auf 8 ist von 0 unterscheidbar. Nach der am 28.10.2026
  festgelegten Lesart (PREREGISTRATION 6.10: Anstieg mit Intervall belegt) ist der **Primärendpunkt bei allen
  drei Dosen nicht gestützt**. Ergebnis: **Wechseleffekt bestätigt, Dosis-Wirkung über K nicht bestätigt,
  H1 als Ganzes nicht gestützt** (§2.3).
* **Sekundär gestützt** (BH über 28 Tests): 1, 4, 5 (ein Messwert), 10 (i) mit SAE bei 98k (neu), 10 (ii),
  10 (iii), 14 bei der höheren Dosis (mit starkem Vorbehalt, §3). **Nicht gestützt:** 6 (Vorzeichen
  umgekehrt), 13. **Offen:** 2, 3 (überwiegend), 14 bei der niedrigeren Dosis.
* **Gate G0 verfehlt** (0 von 13 Layern): 8 und 9 sind nicht interpretierbar.
* **Übernommene Endpunkte:** 10f gestützt (mit Vorbehalt), 12 und 16 nicht gestützt, 15 nicht getestet,
  17 nicht interpretierbar (Platzhalter-Korpus), 18 (Stand-in) gestützt, **18T nicht gestützt**: Das
  Spurenmodell ist gültig trainiert und zählt für die Präregistrierung, zeigt aber keinen Sharing-Effekt.
* **Läufe gültig:** Die Injektions-Audits von C und D liegen weit unter der Grenze 1e-4.

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
nur Dokumente. Die alten G- und I-Läufe auf Konto B hat die Auswertung als älter verworfen (gilt der neueste
Lauf). Die alten H- und J-Läufe von Lauf 1 liegen auf Konto B als `…@step98000.run1`.

Das SAE (`EleutherAI/sae-pythia-160m-32k`, `layers.10`) rekonstruiert die Aktivierungen bei `step98000` mit
FVU = 0,198 (Grenze 0,4) und ist damit dort zertifiziert (Notebook 01 auf Konto B, 27.10. 16:37 UTC).

## 2. Primärendpunkt (PREREGISTRATION §2)

Δλ = λ(K, `lang_dom`) − λ(K = 0, `lang_dom`), Zelle C1, Tiefen-Injektion, K = 8; DiD = Δλ(`lang_dom`) −
Δλ(`rand`, Mittel über 32 Zufallsrichtungen).

| Dosis ε | Δλ `lang_dom` (K = 8) | Δλ `rand` | DiD | Δλ bei K = 2 / 4 / 8 | P(monoton) im Bootstrap | Urteil nach 6.10 (b) | Urteil nach Punktschätzern (a) |
|---|---|---|---|---|---|---|---|
| 0,01 | 0,0949 [0,0705; 0,1183] | −0,0070 [−0,0260; 0,0106] | 0,1019 [0,0875; 0,1174] | 0,0915 / 0,1010 / 0,0949 | 0,147 | **nicht gestützt** | nicht gestützt |
| 0,05 | 0,0993 [0,0736; 0,1248] | −0,0101 [−0,0293; 0,0079] | 0,1094 [0,0910; 0,1289] | 0,0915 / 0,0993 / 0,0993 | 0,258 | **nicht gestützt** | gestützt |
| 0,1 | 0,0941 [0,0690; 0,1184] | −0,0105 [−0,0296; 0,0077] | 0,1046 [0,0856; 0,1246] | 0,0882 / 0,0918 / 0,0941 | 0,236 | **nicht gestützt** | gestützt |

Die Spalte (a) ist das, was der Bericht auf Drive vom 28.10., 04:55 UTC, noch ausweist; er wurde vor der
Festlegung erzeugt. Seit diesem Stand rechnet `lsne/prereg_eval.py` nach (b).

Einseitige p-Werte nach Bonferroni über die drei Dosen: 0,0003 für Δλ und für DiD bei jeder Dosis.

Zuwächse zwischen den K-Stufen (Δλ `lang_dom`):

| Dosis ε | K 2 → 4 | K 4 → 8 |
|---|---|---|
| 0,01 | 0,0095 [−0,0129; 0,0277] | −0,0061 [−0,0274; 0,0204] |
| 0,05 | 0,0078 [−0,0164; 0,0281] | 0,0001 [−0,0228; 0,0287] |
| 0,1 | 0,0036 [−0,0189; 0,0236] | 0,0023 [−0,0187; 0,0267] |

### 2.1 Was die Zahlen zeigen

* **Wechseleffekt:** robust. Bei jeder Dosis liegt Δλ sicher über dem Mindesteffekt, und Zufallsrichtungen
  zeigen in C1 keinen Effekt (Δλ `rand` ≈ −0,01, Intervall um 0). Der Effekt ist also spezifisch für die
  Sprachachse.
* **Dosis-Wirkung in K:** nicht gezeigt. Der Sprung passiert zwischen K = 0 und K = 2. Danach bleibt Δλ
  innerhalb der Unsicherheit gleich. Das passt zum Bild aus Lauf 1 (Sprung, dann Plateau).
* **Zellen außerhalb von C1** (Abweichungszeilen C2, C3, seg; Injektion am letzten deutschen Token):
  Δλ `lang_dom` 0,13–0,15, aber Δλ `rand` ebenfalls 0,13–0,16, DiD ≈ 0. Dort erzeugt jede Richtung denselben
  λ-Anstieg; das ist ein Effekt der Injektionsstelle, nicht der Sprachachse. Nur C1 (positionsgleich zu K = 0)
  trennt beides.

### 2.2 Vorbehalte

* Die `lang_dom`-Achse an der Tiefen-Injektionsstelle (Layer 4) verfehlt ihren Entrauschungs-Test
  (cos_denoised_plain 0,126 < 0,70). Sie bewahrt dort wenig vom einfachen D-E-Mittelwertunterschied.
* Bei K > 0 ist das Token an der Injektionsstelle (letztes Token von Einheit 9) deutsch, bei K = 0 englisch.
  Die Zufallsrichtungen schließen aus, dass allein die Tokensprache λ hebt, aber nicht, dass die Sprachachse
  auf einem deutschen Token anders wirkt als auf einem englischen.
* `lang_dom` ist auf allen Standardvarianten gefittet, die Testfamilien eingeschlossen.
* Gerechnet wurden nur die Dosen 0,01 / 0,05 / 0,1; 0,001 / 0,0025 / 0,005 / 0,02 fehlen (Nachtrag 6.3).

### 2.3 Entscheidung: Lesart von „monoton in K“ (28.10.2026)

PREREGISTRATION §2 verlangt „Δλ > 0 und monoton in K, und größer als derselbe Kontrast mit `rand`“, legt
aber nicht fest, wie Monotonie geprüft wird. Drei Lesarten wurden erwogen:

| Lesart | Ergebnis |
|---|---|
| (a) Punktschätzer streng steigend (so im Code) | 2 von 3 Dosen gestützt; hängt am Rauschen |
| (b) Anstieg mit Intervall belegt (Zuwächse > 0) | bei keiner Dosis erfüllt → H1 nicht gestützt |
| (c) kein belegter Abfall (kein Zuwachs signifikant < 0) | bei allen 3 Dosen erfüllt → H1 gestützt |

Festgelegt ist (b), als datierter Nachtrag 6.10 in `docs/PREREGISTRATION.md`. Gewählt wurde die strengste
Lesart, weil die Festlegung erst nach Sichtung der Daten erfolgt und das Ergebnis nicht begünstigen soll.
Ergebnis: **Wechseleffekt bestätigt, Dosis-Wirkung über K nicht bestätigt, H1 als Ganzes nicht gestützt.**

## 3. Sekundäre Endpunkte (BH-Gruppe, m = 28)

| Endpunkt | Schätzer [95 %-KI] | q (BH) | Urteil | Hinweis |
|---|---|---|---|---|
| 1: T(K = 8, C1) | −0,3438 [−0,5560; −0,1608] | 9,3e-04 | **gestützt** | |
| 2: T(8, C3) − T(8, C1) | 0,0965 [0,0016; 0,2001] | 0,127 | **offen** | |
| 3: R_excess, K0 vs K28seg, Layer 2 | 0,0514 [0,0132; 0,0915] | 0,011 | **gestützt** | |
| 3: K0 vs K28seg, Layer 11 | −0,0564 [−0,1186; −0,0077] | 1,0 | **nicht gestützt** | gegen die erwartete Richtung |
| 3: übrige 6 Teiltests | | ≥ 0,18 | **offen** | Leerraum-Klasse leer: misst R_punct+boundary |
| 4: Hysteresefläche | 0,2935 [0,2757; 0,3097] | 9,3e-04 | **gestützt** | misst großteils Kontextgedächtnis |
| 5: nmi_last | 0,0182 [0,0063; 0,0297] | 0,008 | **gestützt** | |
| 5: nmi_tail | −0,0054 [−0,0111; 0,0011] | 0,207 | **offen** | |
| 6: Transfer `ent` (+1), alle 7 Dosen | z. B. ε 0,05: −0,2343 [−0,2989; −0,1734] | 1,0 | **nicht gestützt** | Vorzeichen umgekehrt; Kontrolle −1 positiv |
| 10 (i): ρ(λ, SAE-FVU) | −0,8006 [−0,8914; −0,7039] | 9,3e-04 | **gestützt** | neu: SAE bei 98k zertifiziert |
| 10 (ii): ρ(λ, Schritte bis Grenze) | 0,1865 [0,0510; 0,3299] | 0,008 | **gestützt** | 17 von 24 Reihen konstant (ρ = 0) |
| 10 (iii): ρ(λ, Mahalanobis) | −0,5208 [−0,7024; −0,3318] | 9,3e-04 | **gestützt** | teils konstruktionsbedingt |
| 13: Gegensteuern `dom_law`, ε 0,05 | −0,0409 [−0,0719; −0,0098] | 1,0 | **nicht gestützt** | 4 Familien |
| 14: Head L11H10 − Residuum, ε 0,1 (cult./pol. / lang) | 0,1053 / 0,0884 | 0,30 / 0,14 | **offen** | |
| 14: dasselbe, ε 0,5 (cult./pol. / lang) | 1,8829 [1,0761; 2,8026] / 3,1819 [2,3780; 3,9455] | 9,3e-04 | **gestützt** | siehe unten |

Hinweise:

* **4:** Der absteigende Ast hängt am deutschen Text, den der aufsteigende erzeugt hat; die Neustarts tun das
  nicht. Die Fläche misst daher großteils autoregressives Kontextgedächtnis.
* **10:** Ab λ_rel 0,3 sättigen die Pfade; ρ vergleicht daher vor allem λ = 0 mit λ > 0. Bei 10 (ii) ist die
  Grenze von +1 nat für 17 von 24 Teststarts unerreichbar. Ohne die konstanten Reihen („wie gelaufen“)
  ergibt sich 0,6313 [0,6124; 0,6692] auf 5 Familien.
* **14:** Bei gleicher relativer Dosis schreibt das Residuum im Median 36-mal so viel (RMS) wie der Head. Die
  positive Differenz bedeutet vor allem, dass der Residuum-Arm stärker schadet. Das ist kein Beleg dafür,
  dass der Head besonders ist.
* **Sensitivität (Bonferroni je Endpunkt, BH über 9 Endpunkte):** 1, 3, 4, 5, 10 und 14 bleiben unter
  q 0,05; 2 liegt bei 0,058; 6 und 13 bei 1,0.

## 4. Gate G0 und Endpunkte 8–9

G0 verfehlt: In keinem der 5 mittleren Layer (0 von 13 insgesamt) gilt die Ordnung
d(A_E, A_D) < d(A_E, B_E) < d(andere Familie). Nach PREREGISTRATION 7 werden 8 (G1-Kontraktion) und 9
(T3-Steuerung) nicht interpretiert. Ohne Gate wäre 9 nicht gestützt (cos 0,20 bzw. 0,08 statt ≥ 0,3).

## 5. Übernommene Endpunkte 10f–18

| Endpunkt | Ergebnis | Urteil |
|---|---|---|
| 10f: SAE-Features, auf DEV eingefroren | 1 606 Kandidaten, 443 auf Test bestätigt, 435 nach BH | **gestützt** (Vorbehalt: Sättigung wie bei 10) |
| 11: B bei λ_rel 3 gegen λ 0 (State 2143) | 98k: 13,60 → 1,94; 99k: 14,43 → 2,29; Vorzeichen gleich | Dokumentation |
| 12: Domänenkompass | Lexikon E: Spitze Religion, Recht Rang 10 von 11; Lexikon D: Spitze Familie, Recht Rang 4 | **nicht gestützt** |
| 15: RDX-Anteil gegen `books` | Gegenpol nicht berechnet | **nicht getestet** (Nachtrag 6.9) |
| 15 mit `misfolding`-Gegenpol (Abweichung) | z ≥ 2 an 4 von 4 Stellen, erklärter Anteil 0,086 | explorativ |
| 16: Pile-Schwierigkeit `protein_misfolding` | Rang 9 von 11, am schwersten `books` | **nicht gestützt** |
| 17: Test-Time-Training | Platzhalter-Korpus | **nicht interpretierbar** |
| 18: Stand-in-Validierung (pythia-70m) | bester Layer 5: cos 0,973, R²-Abstand 1,082, 4 von 4 Layern | **gestützt** |
| 18T: Spurenmodell als Quelle | gültiger Lauf, kein Sharing-Effekt (§5.1) | **nicht gestützt** |

### 5.1 Spurenmodell (Notebook 12, Endpunkt 18T)

Der Lauf ist gültig (`counts_for_prereg = true`): Alle Schritte endeten ok, die Stand-in-Validierung ist
bestanden, das eingefrorene Merkmal blieb unverändert, und das Korpus ist gesund.

* **Daten:** 13,53 Mio. Trainings-Transaktionen (Ethereum 2016), Vokabular 77 364 Token, Adress-OOV 0,155
  im Training und 0,192 im Hold-out (Lauf 1: 0,967).
* **Modell:** `blockgpt`, 104 Mio. Parameter, 20 000 Schritte, 328 Mio. Tokens; Validierungsverlust 0,410 nats
  pro Token gegen 1,386 (Bigramm) und 11,34 (untrainiert).
* **Sonde** (eingefroren: Fan-out k = 5, längengematcht): AUC trainiert 0,986 gegen untrainiert 0,973;
  Differenz 0,013 [−0,003; 0,083] → Kriterium verfehlt.
* **Alignment mit Pythia:** besser als permutiert (R² 0,32 gegen −0,42), aber nicht besser als der
  untrainierte Zwilling (residualisierter Gewinn −0,005, untere Grenze −0,035).
* **Transplantation** in Pythia: Im Residuum-Modus sinkt B gegen das unveränderte Modell um 11,05
  [−11,77; −10,33] statt zu steigen; Zufalls- und permutierte Kontrollen schaden ähnlich. Im kv-Modus kein
  Effekt in der vorgegebenen Richtung (B −0,05 [−0,09; −0,02]). Im Modus „both“ steigt T gegen alle Kontrollen (+0,30 [0,14; 0,45] gegen das unveränderte Modell),
  B fällt aber um 11,40.
* **Urteil:** Kein Anspruch, weil Sonde und Alignment-Kriterium verfehlt sind. Das ist der im Notebook als
  erwartet beschriebene Ausgang.

Laufzeiten gegen Plan (Stunden): Serialisierung 7,1 (Plan 0,25–0,7; die Chunks der DoS-Angriffe vom Herbst
2016 haben 40–60 Mio. Roh-Tokens), Split 0,47 (0,08–0,4), Vokabular 0,92 (0,17–0,5), Training 1,54 (0,8–1,5),
Evaluation 0,92 (0,5–0,75); Hold-out-OOV, Sonden, Paare, Alignment und Stufe L lagen im Plan.

## 6. Ergebnisse bei 99k (nur Stufe H)

| Endpunkt | 99k | 98k |
|---|---|---|
| 10 (ii) | 0,2454 [0,0844; 0,4240], q 0,003, **gestützt** | 0,1865 [0,0510; 0,3299], **gestützt** |
| 10 (iii) | −0,6726 [−0,8214; −0,5045], q 2e-04, **gestützt** | −0,5208 [−0,7024; −0,3318], **gestützt** |
| 11 | B 14,43 → 2,29 | B 13,60 → 1,94 |

10 (i) ist bei 99k nicht getestet, weil 07 bei 99k ohne SAE lief. Alle anderen Endpunkte erscheinen bei 99k
als „nicht getestet“, weil es dort nur Stufe H gibt.

## 7. Injektions-Audits (PREREGISTRATION §4)

| Stufe | Zeilen | größter relativer Fehler | Urteil |
|---|---|---|---|
| D (Schadensausbreitung) | 471 648 | 3,9e-06 | gültig |
| C (Raster) | 647 360 | 4,0e-05 | gültig |

## 8. Offene Entscheidungen

1. ~~Primärendpunkt, Monotonie~~: entschieden am 28.10.2026, Lesart (b) (§2.3, PREREGISTRATION 6.10).
2. **Primärendpunkt, Achse:** ob die `lang_dom`-Achse trotz verfehltem Entrauschungs-Test an Layer 4 gilt.
3. **Primärendpunkt, Kontrolle der Tokensprache** vor einer Bestätigung (z. B. K = 0 mit deutschem Token an
   der Injektionsstelle).
4. **Endpunkt 14:** Dosisabgleich nach geschriebener RMS statt relativer Dosis, sonst bleibt der Vergleich vom
   Residuum-Schaden bestimmt.
5. **Endpunkt 15:** `books`-Gegenpol bauen (Neulauf J) oder 15 explorativ lassen.
6. **Endpunkt 17:** TTT-Korpora liefern.
7. Die übrigen Punkte aus `docs/RESULTS_RUN1.md` §5.5 (Endpunkt 10 neu kalibrieren, Endpunkt 13 Gegensteuer-Arm,
   fehlende Dosen in D).

## 9. Laufprotokoll 27./28.10.2026

* Neuläufe nach dem Korrektur-Commit `97d4104`: 06, 08, 11 und 07 (99k) auf Konto A; 01, 07 (98k) und 09
  mit SAE auf Konto B; 12 auf Konto A; 13 dreimal (98k vor und nach 12, 99k).
* Stolpersteine: `REUSE = True` lud zweimal einen alten Lauf statt neu zu rechnen (01 ohne SAE, 11 im alten
  Design); beide wurden mit `REUSE = False` wiederholt. Die erste Verknüpfung `LSNE_B` war eine
  Windows-Linkdatei und wurde durch eine Drive-Verknüpfung ersetzt.
* Notebook 13 wartet seit `dfb473b` auf ein laufendes Notebook 12 (`WAIT_FOR_12 = "auto"`).
* Die Vorbehaltstexte zu 18 und 18T im Bericht auf Drive beschreiben noch das Design vor den Korrekturen.
  Im Code sind sie seit diesem Stand aktualisiert; die Urteile ändert das nicht.
