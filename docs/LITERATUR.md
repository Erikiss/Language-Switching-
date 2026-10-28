# Literatur zu den Befunden von LS-NEQ

Stand: 28.10.2026. Bezug sind die Befunde aus `docs/RESULTS_PREREG.md`, `docs/RESULTS_RUN1.md` und dem Bericht zu
Endpunkt 6.

**So ist die Liste entstanden.** Pro Befund hat ein Agent gezielt gesucht. Ein zweiter Agent hat jede Quelle unabhängig
nachgeprüft: Titel, Autoren, Jahr und Erscheinungsort über arXiv, die ACL Anthology, die Seiten der Konferenzen oder, beim
Beitrag von Kissane u. a., das Alignment Forum. Danach haben drei weitere Agenten jeden Eintrag inhaltlich gegen die
Abstracts und, wo nötig, die Volltexte geprüft; ihre Korrekturen sind eingearbeitet. „Inhalt“ gibt wieder, was die Arbeit
selbst berichtet. „Bezug“ ist unsere eigene Einschätzung, wie die Arbeit zu LS-NEQ passt; das sind Hypothesen, keine
Aussagen der Autoren. Preprints sind nicht begutachtet. Abschnitt 6 sammelt Arbeiten der letzten zwei Wochen.

**Die Befunde, auf die sich die Liste bezieht:**

1. **Sprachwechsel und Störungsausbreitung (Primärendpunkt).** In der konfirmatorischen Zelle C1 (Wechsel an den Grenzen
   der Einheiten, Injektion am Block-Eingang von Layer 4 in den Punkt am Ende von Einheit 9) steigt λ entlang der
   Sprachachse `lang_dom` mit Sprachwechsel um etwa 0,1 pro Layer; die Zufallsrichtungen ändern sich dort nicht
   nachweisbar. In den Zellen C2, C3 und seg, die an einer anderen Stelle injizieren und nicht interpretiert werden,
   steigen die Zufallsrichtungen dagegen ebenso stark. Fast der ganze Effekt liegt im Schritt von K = 0 auf K = 2, also
   mit der ersten deutschen Passage. Ein weiterer Anstieg mit K ist weder gezeigt noch ausgeschlossen; deshalb ist der
   Primärendpunkt „offen“ (Nachtrag 6.10). Ob der Wechsel wirkt oder die Sprache unmittelbar an der Injektionsstelle,
   soll die Kontrolle aus Nachtrag 6.11 klären; der Lauf steht noch aus und kann auch „offen“ enden.
   Wichtig für den Literaturvergleich: `lang_dom` ist eine entrauschte Mittelwertdifferenz (D − E, ohne die oberste
   gemeinsame Hauptkomponente). An Layer 4 hat sie mit der einfachen Differenz nur den Kosinus 0,126.
2. **Umkehr der Entropie-Richtung (Endpunkt 6).** `ent` erhöht die Entropie überall, schadet dem richtigen Zielwort in
   den Testfamilien aber, während es am Ausgangszustand State 2143 hilft. Nach einer nachträglich gefundenen, noch zu
   prüfenden Erklärung hilft eine flachere Verteilung nur Zielwörtern, die im Kontext überraschender sind als die Entropie
   der Vorhersage.
3. **Checkpoint-Trend (explorativ, vorläufig).** Der Abstand zwischen `lang_dom` und den Zufallsrichtungen (DiD, Zelle C1,
   ε = 0,05) wächst über das Training von etwa 0,055 (step1000–3000) auf etwa 0,11 (step125000–143000). Das liegt vor
   allem daran, dass die λ-Grundlinien ohne Wechsel (K = 0) schrumpfen, entlang `lang_dom` wie entlang der
   Zufallsrichtungen; Δλ(`lang_dom`) selbst steigt nur von etwa 0,069 auf 0,093. Die Werte sind eine Vorschau aus 20 von 27
   Checkpoints, gepoolt und ohne Intervalle (`docs/RESULTS_RUN1.md` §2.1).
4. **SAE-Transfer.** Das SAE von step143000 rekonstruiert die Aktivierungen bei step98000 mit FVU 0,198 (Grenze 0,4). Die
   FVU desselben SAE bei step143000 nach derselben Definition liegt nicht vor.

---

## 1. Störungsausbreitung, Stabilität und Dynamik über die Tiefe

*Zu Befund 1 und 3.*

**Ferreira da Silva, Heimersheim (2026): Evidence for feature-specific error correction in LLMs.**
arXiv-Preprint. [arXiv:2606.24964](https://arxiv.org/abs/2606.24964)

- *Inhalt:* Stört den Residualstrom in einem frühen Layer und misst die Änderung in einem späteren. Kandidaten für
  Merkmalsrichtungen (kontrastive Richtungen, gepoolt aus Begriffen, zehn Sprachen und Programmiersprachen; MELBO; SAE)
  reagieren empfindlicher als Mischungen aus je zwei solchen Richtungen (Exponent p > 2); Zufalls-, PCA- und
  Zufallsdifferenz-Richtungen tun das nicht (p ≈ 2). Gezeigt an sechs Modellen mit 1,7 bis 9 Mrd. Parametern; ein eigenes
  Ergebnis nur für Sprachrichtungen gibt es nicht.
- *Bezug:* Methodisch das unserem Aufbau am nächsten verwandte Design (Störung früh, Messung später, Vergleich mit
  Zufallsrichtungen). Ihre Zufallsdifferenz-Richtungen, die die Kovarianz der Daten erben, wären eine sinnvolle Ergänzung
  zu unseren isotropen Zufallsrichtungen.

**Finlayson, Pernice, Todd u. a. (2026): Language Model Activations Inhabit Privileged Error-Correcting Basins.**
arXiv-Preprint, eingereicht am 3.10.2026. [arXiv:2610.04183](https://arxiv.org/abs/2610.04183)

- *Inhalt:* Aktivierungen liegen in anziehenden „Becken“; entlang semantischer Steuerrichtungen gibt es
  merkmalsspezifische Becken. Wird die Steuerstärke so gewählt, dass sie Beckengrenzen überquert, gelingt die Steuerung
  zwischen Sprachen besser.
- *Bezug:* Eine mögliche Lesart des Sprungs bei K = 0 → 2: Die erste deutsche Passage bringt den Zustand nahe an eine
  Beckengrenze entlang der Sprachachse, wo Störungen schneller wachsen. Das erklärte auch, warum die Zufallsrichtungen in
  C1 nicht reagieren.

**Janiak, Karwowski, Mangat u. a. (2024): Characterizing stable regions in the residual stream of LLMs.**
NeurIPS 2024 Workshop SciForDL. [arXiv:2409.17113](https://arxiv.org/abs/2409.17113)

- *Inhalt:* Der Residualstrom zerfällt in stabile Regionen. Innerhalb reagiert die Ausgabe kaum, an den Grenzen sehr
  empfindlich. Die Regionen fehlen bei zufälliger Initialisierung, entstehen im Training und werden mit Training und
  Modellgröße schärfer.
- *Bezug:* Die nächste Analogie zum Checkpoint-Trend: Ohne Wechsel (K = 0) sinkt λ über das Training, entlang `lang_dom`
  ebenso wie entlang der Zufallsrichtungen; nach dem Wechsel sinkt λ entlang `lang_dom` weniger stark, deshalb wächst der
  Kontrast. Stabile Regionen sagen außerdem stufenartige statt gleitende Reaktionen voraus.

**Fernando, Guitchounts (2026): Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network
Topology.** arXiv-Preprint. [arXiv:2605.14258](https://arxiv.org/abs/2605.14258)

- *Inhalt:* Zerlegt die Jacobi-Matrizen der Layer vollständig in Eigenwerte. Ihr Produkt über die Layer bildet einen
  niedrigrangigen Engpass, der Störungen auf wenige Dimensionen bündelt; dieser Engpass ist gelernt (OLMo, Beginn gegen
  Ende des Trainings).
- *Bezug:* Ein konkreter Folgetest: Wie stark überlappt `lang_dom` mit den führenden Singulärvektoren der Jacobi-Matrix
  von Layer 4 bis 12, über Checkpoints und über K?

**Cowsik, Nebabu, Qi, Ganguli (2024): Geometric Dynamics of Signal Propagation Predict Trainability of Transformers.**
arXiv-Preprint. [arXiv:2403.02579](https://arxiv.org/abs/2403.02579)

- *Inhalt:* Theorie zu Ordnung und Chaos über die Tiefe von Transformern bei zufälliger Initialisierung. Zwei
  Lyapunov-Exponenten sagen voraus, wie gut sich das Netz trainieren lässt.
- *Bezug:* Die begriffliche Grundlage, λ als Lyapunov-Exponenten über die Tiefe zu lesen. Sie gilt nur bei
  Initialisierung, also allenfalls für die frühesten Checkpoints.

**Jaca, Benedek, Török (2026): Chaos in reason: How chain-of-thought LLMs can look for an answer.**
arXiv-Preprint. [arXiv:2607.27805](https://arxiv.org/abs/2607.27805)

- *Inhalt:* Jacobi-Analyse von Teilblöcken trainierter Sprachmodelle. MLP und Attention verstärken Störungen, die
  Normalisierung faltet sie zurück. Endliche Störungen wachsen in frühen Layern, flachen in der Mitte ab und schrumpfen
  spät; die Größe des gemeinsamen Kontexts verändert die Stabilität.
- *Bezug:* Ein Hinweis, dass deutscher Kontext über das Attention-Routing die Verstärkung entlang der Sprachachse erhöhen
  könnte. Achtung: Ihre Lyapunov-Exponenten laufen über erzeugte Tokens, nicht über die Tiefe.

**Islam, Villarreal, Nishino u. a. (2026): Numerical Instability and Chaos: Quantifying the Unpredictability of Large
Language Models.** arXiv-Preprint. [arXiv:2604.13206](https://arxiv.org/abs/2604.13206)

- *Inhalt:* Misst, wie stark kleine Störungen über die Layer verstärkt werden. Es gibt einen frühen „Lawinen“-Effekt und
  eingabeabhängige Schwellen zwischen stabilem, chaotischem und signaldominiertem Verhalten; die Schwellen verschieben
  sich mit der Gleitkommagenauigkeit.
- *Bezug:* Eine methodische Mahnung: Der λ-Kontrast sollte über Störungsgrößen hinweg Bestand haben, bevor man ihn
  mechanistisch deutet. Unsere drei Dosen sind ein erster Schritt.

---

## 2. Mehrsprachigkeit und Sprachwechsel im Modell

*Zu Befund 1.*

**Quirke, Heindrich, Gurnee, Nanda (2023): Training Dynamics of Contextual N-Grams in Language Models.**
ATTRIB-Workshop, NeurIPS 2023. [arXiv:2311.00863](https://arxiv.org/abs/2311.00863)

- *Inhalt:* In Pythia-70M, ebenfalls auf dem Pile trainiert, entsteht früh im Training ein Neuron, das deutschen Kontext
  erkennt. Es steuert spätere Neuronen, die deutsche N-Gramme fortsetzen, und das Modell hängt über die späteren
  Checkpoints immer stärker davon ab.
- *Bezug:* Gleiche Modellfamilie, gleiche Sprache, gleiche Checkpoint-Reihe. Ein Deutsch-Detektor, der einmal anspringt
  und dann sättigt, würde erklären, warum der Schritt zur ersten deutschen Passage fast den ganzen Effekt trägt. Direkter
  nächster Schritt: prüfen, ob `lang_dom` in Pythia-160m mit solchen Neuronen zusammenfällt.

**Nie, Schmid, Schütze (2025): Mechanistic Understanding and Mitigation of Language Confusion in English-Centric Large
Language Models.** Findings of EMNLP 2025. [arXiv:2505.16538](https://arxiv.org/abs/2505.16538)

- *Inhalt:* In englischzentrierten Modellen ist der „Verwechslungspunkt“, das erste Token in der falschen Sprache in der
  erzeugten Ausgabe, die kritische Stelle. Der Übergang in die Zielsprache scheitert in den letzten Layern, wo auch die
  entscheidenden Neuronen sitzen.
- *Bezug:* Ein mechanistisches Vorbild dafür, einen Sprachwechsel an einer bestimmten Stelle Layer für Layer zu
  verfolgen. Bei uns bündelt sich der Effekt allerdings nicht an einer Wechselstelle, sondern im Schritt von K = 0 auf
  K = 2; injiziert wird in den Punkt am Ende von Einheit 9, direkt vor dem letzten Wechsel zurück ins Englische.

**Chang, Tu, Bergen (2022): The Geometry of Multilingual Language Model Representations.** EMNLP 2022.
[arXiv:2205.10964](https://arxiv.org/abs/2205.10964)

- *Inhalt:* In XLM-R unterscheiden sich die Sprachen vor allem in ihren Mittelwerten entlang sprachempfindlicher Achsen.
  Verschiebt man Repräsentationen um die Mittelwertdifferenz, sagt das Modell Tokens der anderen Sprache vorher;
  sprachneutrale Information liegt auf orthogonalen Achsen.
- *Bezug:* Der direkte Vorläufer von Sprachachsen aus Mittelwertdifferenzen. `lang_dom` geht von der Differenz D − E aus,
  entfernt aber die oberste gemeinsame Hauptkomponente; an der Tiefen-Injektionsstelle (Layer 4) bleibt von der einfachen
  Differenz wenig übrig (Kosinus 0,126). Ein Vergleich mit der einfachen Mittelwertdifferenz liegt daher nahe.

**Gurgurov, Al Ghussin, Baeumel u. a. (2026): CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark.**
Findings of ACL 2026. [arXiv:2601.08331](https://arxiv.org/abs/2601.08331)

- *Inhalt:* Vergleich von Methoden zur Sprachsteuerung über 32 Sprachen. Die einfache, normierte Differenz der
  Sprachmittelwerte im Residualstrom schneidet in Llama-3.1-8B-Instruct am besten ab; in Aya-Expanse-8B liegen laut Anhang
  die Prompting-Baselines und LAPE im Mittel vorn. Sprachspezifische Struktur zeigt sich vor allem in späteren Layern.
- *Bezug:* Ein Gegenstück auf der Verhaltensebene zu einer Wirkung entlang einer Sprachachse, die über die Tiefe wächst.
  `lang_dom` ist dagegen eine entrauschte Mittelwertdifferenz; an Layer 4 hat sie mit der einfachen Differenz nur den
  Kosinus 0,126.

**Park, Yoon, Jun, Lee (2026): Code-Switching Reveals Language Anchoring in Multilingual LLMs.** Laut arXiv Findings of
EMNLP 2026. [arXiv:2606.19668](https://arxiv.org/abs/2606.19668)

- *Inhalt:* Setzt man einzelne Wörter einer anderen Sprache in einen englischen Satz, bleiben die Repräsentationen am
  Englischen „verankert“ (im Mittel in 76 % der Layer). Liefert die andere Sprache den Satzrahmen, verschieben sich die
  Repräsentationen zu dieser Sprache, am stärksten in den letzten rund 30 % der Layer. Das Muster folgt der Sprache des
  Satzrahmens, nicht Englisch als solchem (repliziert mit Spanisch und weiteren Ausgangssprachen).
- *Bezug:* Ihr Design unterscheidet sich von unserem: Sie setzen einzelne Wörter in einen Fragesatz ein, bei uns wechseln
  ganze Einheiten (C1) oder Satzteile (C2, C3) mit ihrer eigenen Grammatik die Sprache, und ein gemeinsamer englischer
  Schlussabschnitt folgt. Ihre Metrik als Funktion von K würde prüfen, ob mehr Wechsel als bei K = 2 wenig hinzufügen.

**Wendler, Veselovsky, Monea, West (2024): Do Llamas Work in English? On the Latent Language of Multilingual
Transformers.** ACL 2024. [arXiv:2402.10588](https://arxiv.org/abs/2402.10588)

- *Inhalt:* Eine Logit-Lens-Analyse von Llama-2 findet drei Phasen: Eingaberaum, ein englischnaher Begriffsraum in den
  mittleren Layern, ein sprachspezifischer Ausgaberaum in den letzten Layern.
- *Bezug:* Für ein englischdominantes Modell wie Pythia eine prüfbare Vorhersage für das Tiefenprofil der Ausbreitung
  entlang `lang_dom`: in der Mitte gedämpft, spät wieder verstärkt.

**Goncharov, Kondusov, Zaytsev (2025): Language steering in latent space to mitigate unintended code-switching.**
AINL 2026 (laut arXiv-Kommentar). [arXiv:2510.13849](https://arxiv.org/abs/2510.13849)

- *Inhalt:* In zwei kleinen Modellen (Qwen2.5-1.5B, Llama-3.2-1B) trennt im letzten Layer die erste Hauptkomponente der
  Hidden States paralleler Übersetzungen die Sprachen fast perfekt (96–100 %); in frühen Layern sind die Sprachgruppen
  nur locker getrennt. Steuern entlang dieser Komponente in den letzten Layern verringert ungewolltes Code-Switching
  stark.
- *Bezug:* Stützt für die letzten Layer, die Sprachidentität als im Wesentlichen eindimensionale Achse zu behandeln.
  Unsere Tiefen-Injektion liegt aber in Layer 4. Ihre Hauptkomponente wäre eine Vergleichsrichtung zu `lang_dom`.

**Marchisio, Ko, Bérard, Dehaze, Ruder (2024): Understanding and Mitigating Language Confusion in LLMs.**
EMNLP 2024. [arXiv:2406.20052](https://arxiv.org/abs/2406.20052)

- *Inhalt:* Sprachverwechslung tritt vor allem dort auf, wo die Verteilung des nächsten Tokens flach ist; eine höhere
  Sampling-Temperatur verstärkt sie.
- *Bezug:* Eine prüfbare Vorhersage für die Erzeugungsexperimente von LS-NEQ (deutscher Anteil f_D): Wechsel in der
  erzeugten Sprache sollten dort häufiger sein, wo die Entropie hoch ist.

---

## 3. Entropie und Zuversicht

*Zu Befund 2, der Umkehr der Entropie-Richtung.* Die Detailanalyse steht im eigenen Bericht zu Endpunkt 6. Kurz: B misst,
wie stark der Eingriff die Wahrscheinlichkeit des richtigen nächsten Wortes ändert. `ent` erhöht überall die Entropie.
Nach einer nachträglich gefundenen, noch zu prüfenden Erklärung hilft das nur Zielwörtern, die im Kontext überraschender
sind als die Entropie der Vorhersage, wie „bey“ am State 2143 (Rang ≈ 6 700). In den Testfamilien steht das Zielwort an der
Spitze der Verteilung und verliert.

**Stolfo, Wu, Gurnee u. a. (2024): Confidence Regulation Neurons in Language Models.** NeurIPS 2024.
[arXiv:2406.16254](https://arxiv.org/abs/2406.16254)

- *Inhalt:* Sprachmodelle regeln ihre Zuversicht über „Entropie-Neuronen“. Diese schreiben in einen fast wirkungslosen
  Unterraum der Unembedding-Matrix und wirken über die Skalierung der letzten LayerNorm wie eine Temperatur.
  „Token-Frequenz-Neuronen“ schieben die Ausgabe zur Häufigkeitsverteilung hin oder von ihr weg. Untersucht unter anderem
  an Pythia 410M und 1B; dort erklärt die LayerNorm-Skalierung nur etwa 30–40 % der Wirkung der Entropie-Neuronen (bei
  GPT-2, LLaMA2 7B und Phi-2 etwa 80 %).
- *Bezug:* Ein plausibler Kanal für `ent`. Eine Temperaturerhöhung hilft einem Token nur, wenn es vorher überraschender
  war als die Verteilung im Mittel, wie das Zielwort bei State 2143. Steht es vorne, wie in allen Testfamilien, schadet
  sie. Weil in Pythia nur ein Teil der Wirkung über die Temperatur läuft, ist die Temperatur-Lesart von `ent` eine
  Hypothese.

**Tan, Chanin, Lynch u. a. (2024): Analysing the Generalisation and Reliability of Steering Vectors.**
NeurIPS 2024. [arXiv:2407.12404](https://arxiv.org/abs/2407.12404)

- *Inhalt:* Wie gut sich ein Modell steuern lässt, schwankt stark zwischen Eingaben. Bei manchen Datensätzen bewirkt
  derselbe Vektor bei einem großen Teil der Eingaben das Gegenteil, schon innerhalb der Verteilung, aus der er stammt; die
  Autoren führen das oft auf eine Bevorzugung bestimmter Antwortpositionen oder -tokens zurück. Steuervektoren übertragen
  sich besser, wenn das ungesteuerte Verhalten in Ausgangs- und Zielsituation ähnlich ist.
- *Bezug:* Ein dokumentierter Fall, in dem derselbe Steuervektor bei einem Teil der Eingaben das Gegenteil bewirkt. Bei
  uns hängt das Vorzeichen nach einer nachträglichen, noch zu prüfenden Erklärung vom Ausgangszustand ab (Überraschung
  des Zielworts gegen Entropie).

**Li, Li, Huang (2026): Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models.**
EMNLP 2026. [arXiv:2602.01654](https://arxiv.org/abs/2602.01654)

- *Inhalt:* Ein fester Steuervektor setzt voraus, dass eine Richtung überall gleich wirkt; ändert sich die lokal wirksame
  Richtung mit dem Zustand, kann die Wirkung schwach werden oder sich umkehren. Die Arbeit lernt deshalb eine
  differenzierbare Bewertungsfunktion für das Konzept (ein kleines MLP, trainiert auf Präferenzdaten) und nimmt deren
  lokalen Gradienten an jeder Aktivierung als Steuerrichtung. Wird die Richtung seltener neu berechnet, steuert sie
  schwächer.
- *Bezug:* `ent` ist ein eingefrorener lokaler Gradient, allerdings der Entropie und nicht eines gelernten Konzepts. Nach
  unseren Daten behält er seine Entropie-Wirkung überall; eine je Familie neu berechnete Richtung (in `docs/DESIGN.md`
  §4.3 als eigener Arm geplant, aber nicht umgesetzt) bleibt trotzdem der saubere Vergleich.

**Cancedda (2024): Spectral Filters, Dark Signals, and Attention Sinks.** ACL 2024.
[arXiv:2402.09221](https://arxiv.org/abs/2402.09221)

- *Inhalt:* „Dunkle“ Signale am Ende des Spektrums der Unembedding-Matrix erzeugen Attention-Sinks.
- *Bezug:* Richtungen im fast wirkungslosen Unterraum sind nicht wirkungslos. Einfache Diagnose: messen, wie viel von
  `ent`, `lang_dom` und den Zufallsrichtungen in den unteren Singulärrichtungen der Unembedding-Matrix liegt.

**Park, Choe, Veitch (2024): The Linear Representation Hypothesis and the Geometry of Large Language Models.**
ICML 2024. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)

- *Inhalt:* Unterscheidet zwei lineare Darstellungen eines Begriffs, eine auf der Ausgabeseite (zum Auslesen) und eine
  auf der Eingabeseite (zum Steuern), und verbindet sie über ein „kausales Skalarprodukt“.
- *Bezug:* Hintergrund für Gradienten-Richtungen: Ein Gradient gehört zur Ausleseseite. Ihn direkt als Steuervektor zu
  verwenden, setzt stillschweigend die euklidische Geometrie voraus.

---

## 4. SAE-Transfer und Merkmalsdynamik

*Zu Befund 4.*

**Ge, Shu, Wu u. a. (2026): Evolution of Concepts in Language Model Pre-Training.** ICLR 2026.
[arXiv:2509.17196](https://arxiv.org/abs/2509.17196)

- *Inhalt:* Crosscoder über 32 Checkpoints von Pythia-160M. Die meisten Merkmale bilden sich um step1000 und ändern dabei
  ihre Richtung stark; danach drehen sie sich nur noch langsam weiter.
- *Bezug:* Der nächste Kontext zu unserem Transfer von step143000 auf step98000: Späte Merkmale sind weitgehend geteilt
  und driften langsam. Nächster Schritt wäre ein Crosscoder für 98k und 143k.

**Kissane, Krzyzanowski, Conmy, Nanda (2024): SAEs (usually) Transfer Between Base and Chat Models.** Alignment Forum,
Zwischenbericht. [Beitrag](https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models)

- *Inhalt:* Eingesetzt ins Chat-Modell erreichen SAEs des Basismodells fast denselben Kreuzentropie-Verlust wie eigene
  Chat-SAEs (Mistral-7B, Qwen 1.5 0.5B). Die erklärte Varianz liegt aber spürbar niedriger (Mistral-7B: 60,9 % gegen
  69,2 %), und beides gilt erst nach Ausschluss der Aktivierungen mit sehr großer Norm (ohne diesen Filter 32,2 % erklärte
  Varianz).
- *Bezug:* Unsere FVU ist ein Varianzmaß und kann deshalb schlechter aussehen als die Treue im Modell. Prüfen, ob die FVU
  von 0,198 von Positionen mit großer Norm getrieben ist, und zusätzlich einen Einsetztest machen.

**Lieberum u. a. (2024): Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.**
BlackboxNLP-Workshop 2024. [arXiv:2408.05147](https://arxiv.org/abs/2408.05147)

- *Inhalt:* Offene SAEs für Gemma 2. SAEs des vortrainierten Modells funktionieren auch im instruktionsgetunten Modell,
  gemessen am Verlust fast so gut wie eigene, gemessen an der FVU etwas schlechter.
- *Bezug:* FVU-Grenze (0,4) und Treue im Modell können auseinanderlaufen. Ergänzend einen Einsetztest machen (Änderung
  von Verlust oder KL, wenn die Rekonstruktion ins Modell eingesetzt wird).

**Xu, Wang, Huang, Wang (2024): Tracking the Feature Dynamics in LLM Training: A Mechanistic Study.**
arXiv-Preprint. [arXiv:2412.17626](https://arxiv.org/abs/2412.17626)

- *Inhalt:* Eine Folge von SAEs über Pythia-Checkpoints, jedes vom vorigen aus initialisiert (Methode SAE-Track).
  Merkmalsrichtungen driften deutlich weiter, auch wenn ihre Bedeutung schon feststeht.
- *Bezug:* Ob ein Teil unserer FVU von 0,198 auf eine solche Drift zwischen step98000 und step143000 zurückgeht, ist
  offen: Die FVU desselben SAE bei step143000 liegt nicht vor, und ohne diese Basis lässt sich ein Transferverlust nicht vom
  eigenen Rekonstruktionsfehler des SAE trennen. Die Arbeit selbst prüft kein festes SAE über Checkpoints.

**Suresh, Stanley, Joseph u. a. (2026): At the Edge of Understanding: Sparse Autoencoders Trace The Limits of
Transformer Generalization.** arXiv-Preprint. [arXiv:2606.26396](https://arxiv.org/abs/2606.26396)

- *Inhalt:* Ungewohnte Eingaben (Tippfehler, Jailbreaks) lassen in einem SAE mehr seltene, unpassende Merkmale anspringen
  und erhöhen oft den Rekonstruktionsfehler; beides fasst ein „Energie“-Wert zusammen, der solche Zustände während der
  Inferenz anzeigt (GPT-2, Llama 3.1 8B; nur SAEs mit L1-Strafe, keine TopK-SAEs).
- *Bezug:* Passt zur Lesart aus Stufe H, dass ungesteuerte Pfade den Bereich verlassen, den das SAE gut rekonstruiert. Die
  Arbeit untersucht aber ungewohnte Eingaben, keine Pfade im Aktivierungsraum; bei unserem TopK-SAE ist nur der
  Rekonstruktionsfehler-Teil ihres Signals übertragbar.

**Lee, Heimersheim (2024): Investigating Sensitive Directions in GPT-2: An Improved Baseline and Comparative Analysis
of SAEs.** NeurIPS 2024 Workshops ATTRIB und SciForDL. [arXiv:2410.12555](https://arxiv.org/abs/2410.12555)

- *Inhalt:* Ergebnisse von Störungsexperimenten hängen stark von der Vergleichsrichtung ab. Isotrope Zufallsrichtungen
  sind problematisch, weil echte Aktivierungen nicht isotrop sind; Richtungen vom Ausgangszustand zu einer anderen
  Aktivierung verzerren den Vergleich, weil sie den negativen Ausgangszustand enthalten. Empfohlen werden
  „Mischungs“-Richtungen.
- *Bezug:* Spricht dafür, neben unseren isotropen Zufallsrichtungen auch Mischungs-Richtungen als Vergleich zu nehmen:
  die Differenz zweier kovarianzangepasster Zufallsaktivierungen oder zweier anderer echter Aktivierungen.

**Mayne, Yang, Mahdi (2024): Can sparse autoencoders be used to decompose and interpret steering vectors?**
arXiv-Preprint. [arXiv:2411.08790](https://arxiv.org/abs/2411.08790)

- *Inhalt:* SAEs zerlegen Steuervektoren irreführend, aus zwei Gründen: Steuervektoren liegen außerhalb der
  Eingabeverteilung, für die SAEs gebaut sind, und sie können bedeutsame negative Projektionen auf Merkmalsrichtungen
  haben, die ein SAE nicht abbilden kann.
- *Bezug:* Aussagen eines SAE über Zustände abseits der Daten brauchen diesen Vorbehalt.

---

## 5. Trainingsdynamik bei Pythia

*Zu Befund 3, dem Checkpoint-Trend.* Dazu gehören auch Quirke u. a. (Abschnitt 2: ein Deutsch-Detektor entsteht früh im
Training, das Modell stützt sich danach immer stärker darauf), Janiak u. a. (Abschnitt 1) und Ge u. a. (Abschnitt 4: die
meisten Merkmale bilden sich um step1000).

**Tao, Cheng, Xie (2026): Persistent Depth Ordering amid Shifting Block-Bypass Responses in Language Model
Pretraining.** arXiv-Preprint, eingereicht am 1.10.2026. [arXiv:2610.01165](https://arxiv.org/abs/2610.01165)

- *Inhalt:* In sechs Pythia-160M-Läufen mit verschiedenen Seeds wächst zwischen step14000 und step143000 das Update, das
  beim Überbrücken eines Blocks fehlt (relative RMS 0,39 → 0,59). Skaliert man diese Störung in ihrer eigenen Richtung auf
  gleiche relative Größe, sinkt die Reaktion an der Ausgabe (JS-Divergenz) im Mittel über vier Stärken. Bei der kleinsten
  Stärke steigt sie dagegen; nur bei den beiden größten sinkt sie. Die Arbeit verfolgt außerdem, wie stark relative
  Störungen über die Tiefe abklingen, für diese Richtung gegen Zufallsrichtungen.
- *Bezug:* Dasselbe Modell, mehrere Seeds. Unsere Dosen (0,01–0,1 der RMS des Zustands) liegen bei oder unter ihrer
  kleinsten Stufe, wo die Reaktion spät im Training eher größer ist. Ob der Befund zu unseren sinkenden λ-Grundlinien ohne
  Wechsel (K = 0, für `lang_dom` wie für die Zufallsrichtungen) passt, ist deshalb offen.

**Chen, Beyer, Zhu, Chen (2026): Why Does Post-Training Quantization Work?** arXiv-Preprint.
[arXiv:2609.11716](https://arxiv.org/abs/2609.11716)

- *Inhalt:* Fehler in den Hidden States wachsen über die Tiefe langsam, weil der neue Fehler jedes Blocks dem geerbten
  Fehler entgegengerichtet ist. Diese Gegenwirkung entsteht im Vortraining; in Checkpoints von Pythia-1.4B und -2.8B ist
  sie bei der Initialisierung nahe null und am Ende deutlich ausgeprägt.
- *Bezug:* Ein Kandidat für den Mechanismus hinter den sinkenden λ-Grundlinien. Diagnose: pro Layer den Kosinus zwischen
  der geerbten Störung (gestörter minus ungestörter Zustand am Blockeingang) und der Änderung des Block-Updates (gestörtes
  minus ungestörtes Update) messen und prüfen, ob `lang_dom` nach dem Wechsel dieser Dämpfung teilweise entgeht.

**Belrose, Pope, Quirke, Mallen, Fern (2024): Neural Networks Learn Statistics of Increasing Complexity.**
ICML 2024. [arXiv:2402.04362](https://arxiv.org/abs/2402.04362)

- *Inhalt:* Netze lernen zuerst einfache Statistiken wie Unigramme und Bigramme, dann komplexere; gezeigt unter anderem an
  Pythia-Checkpoints mit mehreren Seeds.
- *Bezug:* Bei step1000 kennt Pythia schon Token-Statistiken. Der frühe Kontrast könnte daher teils lexikalisch sein. Das
  spricht dafür, die Kontrolle aus Nachtrag 6.11 auch an frühen Checkpoints zu rechnen.

**Li, Agrawal, Ghosh u. a. (2025): Tracing the Representation Geometry of Language Models from Pretraining to
Post-training.** NeurIPS 2025. [arXiv:2509.23024](https://arxiv.org/abs/2509.23024)

- *Inhalt:* Spektrale Maße auf OLMo- und Pythia-Checkpoints zeigen drei Phasen: einen Kollaps während des Warm-ups, eine
  Ausdehnungsphase und eine späte Kompressionsphase, die die Varianz auf wenige dominante Richtungen bündelt.
- *Bezug:* Eine späte Kompression könnte erklären, warum λ ohne Wechsel (K = 0) über das Training sinkt, entlang der
  Zufallsrichtungen wie entlang `lang_dom`. Warum λ nach dem Wechsel entlang `lang_dom` weniger stark sinkt, erklärt sie
  allein nicht. Achtung: Hauptanalyse am letzten Layer und letzten Token; die Phasen zeigen sich auch in mittleren Layern,
  gezeigt aber nur für Modelle ab 1 Mrd. Parametern.

---

## 6. Neueste Arbeiten (14. bis 27.10.2026)

Diese Sammlung zielt auf Arbeiten der letzten zwei Wochen. Sie sind noch nicht begutachtet und zum Teil nur an
Spielzeugmodellen gezeigt, bringen aber neue Blickwinkel. Grundlage war eine Abfrage direkt bei arXiv mit exakten
Einreichungsdaten am 28.10.2026: 199 Paper zu den fünf Themen, eingereicht ab dem 14.10.2026, dazu eine Websuche. Je Thema
hat ein Agent alle Einträge gelesen und die nützlichsten ausgewählt. Ein zweiter Agent hat jede Auswahl gegen die
Abstract-Seite und den Volltext geprüft und Überzeichnungen korrigiert. Angegeben ist das Datum der ersten Einreichung (v1).

Keine der neuen Arbeiten untersucht Sprachwechsel zusammen mit Störungsausbreitung; die Lücke aus Abschnitt 8 besteht
weiter.

### Zu Störungsausbreitung, λ und Checkpoint-Trend

**Ostrow, Qian, Liu, Fiete: A Mathematical Theory of the Jacobian Lens.** 22.10.2026.
[arXiv:2610.31955](https://arxiv.org/abs/2610.31955)

- *Inhalt:* Eine Theorie der „Jacobian Lens“, der über Kontexte und Positionsabstände gemittelten Jacobi-Matrix des
  Residualstroms; die Tiefe spielt die Rolle der Zeit, die Token-Position die des Raums. Unter Modellannahmen findet die
  Lens Richtungen, die sich wie führende Lyapunov-Moden verhalten: Kleine Störungen entlang dieser Richtungen wachsen über
  die Layer, bleiben über Positionen etwa gleich und sind über Kontexte konsistent. Validiert an einem Bayes-Filter und
  einem kleinen Transformer.
- *Bezug:* Ein eng verwandter theoretischer Rahmen: Unser λ entspräche am ehesten dem Tiefenexponenten ihres
  illustrativen Modells. Anders als bei uns hält die Lens die Ausgabeschicht fest und mittelt über Kontexte, während wir
  eine endliche Störung über die Layer 5 bis 12 verfolgen. Ein Test: `lang_dom` und die Zufallsrichtungen mit den
  führenden Singulärrichtungen eines in Pythia-160m geschätzten, gemittelten Jacobians vergleichen.

**Jones: Chaos and bifurcations in continuous language-model feedback.** 20.10.2026.
[arXiv:2610.31692](https://arxiv.org/abs/2610.31692)

- *Inhalt:* Speist man die erwartete Einbettung der eigenen Vorhersage zurück, wird ein Sprachmodell zu einem dynamischen
  System; Temperatur und Steuerung schieben es zwischen sich wiederholendem Text und Chaos hin und her. Die Verstärkung
  zerfällt in die inverse Temperatur, die Streuung der Token-Einbettungen unter der Vorhersageverteilung und die
  Jacobi-Matrix des Netzes. Im chaotischen Bereich ändert schon eine andere Batch-Aufteilung das Ergebnis.
- *Bezug:* Macht den Ausgabeteil (Softmax) zu einem eigenen Verstärker, was zu unserer Lesart der `ent`-Umkehr als
  Ausgabe-Effekt passt, ohne sie zu prüfen. Methodisch ein Anlass, das numerische Grundrauschen von λ bei kleinen Dosen
  mitzumessen.

**Oh, Ahn, Yu: Reader–Writer Coupling in Transformer MLPs.** 21.10.2026.
[arXiv:2610.29340](https://arxiv.org/abs/2610.29340)

- *Inhalt:* In jedem untersuchten MLP-Block gibt es einen kleinen Kern stark gekoppelter Kanäle. Wo die Initialisierung
  beide Vorzeichen gleich wahrscheinlich macht, ist dieser Kern fast vollständig gegenläufig: Die Ausgabe zeigt gegen die
  Richtung, auf die der Eingang anspricht. Das Vorzeichen steht nach wenigen Prozent des Trainings fest, die Kopplung
  wächst weiter; untersucht über die Checkpoints der sieben Pythia-Größen von 70M bis 6.9B. Bei Pythia-160M legt sich ein
  Block erst spät fest. Nebenbefund: Im Repository von Pythia-2.8B tragen die Checkpoints step54000 bis step64000 die Daten
  von step143000, und eine veraltete safetensors-Datei hängt an 77 Checkpoints.
- *Bezug:* Ein möglicher schichtweiser Baustein von λ (nur MLP, ohne Identitätspfad) und ein Anlass zu prüfen, ob der
  Checkpoint-Trend mit dem späten Umbau einzelner Blöcke zusammenfällt. Praktisch: unsere Checkpoints per Prüfsumme auf
  Eindeutigkeit prüfen.

**Yao, Boguraev, Pimentel, Mahowald: The Emergence and Sparsification of a Syntactic Circuit.** 23.10.2026, laut arXiv
EMNLP 2026. [arXiv:2610.33299](https://arxiv.org/abs/2610.33299)

- *Inhalt:* Der Schaltkreis für Subjekt-Verb-Kongruenz in Pythia-1B und zehn Seeds von Pythia-410M (neun davon aus
  PolyPythias) entsteht früh und baut sich danach weiter um, obwohl das Verhalten stabil ist. Zwei Ausreißer-Seeds zeigen
  Umbrüche, die mit Sprüngen im Verlust zusammenfallen.
- *Bezug:* Spricht dafür, den Checkpoint-Trend nicht ohne Weiteres als allgemeine Trainingseigenschaft zu lesen. Er sollte
  an weiteren Seeds von Pythia-160m (PolyPythias) geprüft werden.

### Zu Sprachwechsel und der Kontrolle aus Nachtrag 6.11

**Jeong, Laroza, Yoon u. a.: One Head Away from Language Confusion: Single Attention Heads Across Model Families.**
18.10.2026. [arXiv:2610.25026](https://arxiv.org/abs/2610.25026)

- *Inhalt:* In zehn von zwölf instruktionsgetunten Modellen (1–7B) führt das Abschalten eines einzelnen, per Suche
  gefundenen Attention-Heads deutlich häufiger dazu, dass eine Fortsetzung die erwartete Sprache verlässt, als das
  Abschalten jedes anderen Heads derselben Schicht. In Basismodellen ist der Effekt kleiner.
- *Bezug:* Eine gleiche Suche in Pythia-160m könnte klären, ob es dort einen Head gibt, der die Sprache verfolgt. Ob Δλ
  über ihn läuft, zeigte erst eine erneute λ-Messung bei abgeschaltetem Head.

**Martins, Dunn: Where Are Language-Specific Features Represented in Multilingual LLMs? Everywhere.** 18.10.2026.
[arXiv:2610.25505](https://arxiv.org/abs/2610.25505)

- *Inhalt:* In Llama 3.2 1B lässt sich die Sprache in jeder Schicht gut ablesen, nicht nur in den ersten und letzten. Ein
  vorangestellter Kontext in einer Sprache verschiebt die Darstellung eines mehrdeutigen Satzes zu dieser Sprache.
- *Bezug:* In Analogie zum Priming-Befund könnte auch der Punkt am Ende eines deutschen Satzteils Sprachinformation aus
  dem vorangehenden Kontext tragen. Das hält die lokale Erklärung, die die Kontrolle aus Nachtrag 6.11 prüft, für möglich,
  entscheidet sie aber nicht. Ein einfacher Vortest wäre ein Sprach-Klassifikator an der Injektionsstelle bei K = 0 und
  K > 0.

**Ding: Language Models Keep Abandoned Analyses: Residue and Adaptation after Garden-Path Sentences.** 27.10.2026.
[arXiv:2610.36797](https://arxiv.org/abs/2610.36797)

- *Inhalt:* Nach Holzwegsätzen bleibt ein Rest der verworfenen Lesart, der spätere Vorhersagen beeinflusst; per Activation
  Patching (neun Modelle von 1,4B bis 70B) liest späterer Text diesen Rest aus den Zuständen der Nominalphrase, die vor der
  Auflösung berechnet wurden. In einem zweiten Experiment zu überschriebenen Aussagen zeigt ein Attention-Knockout: In
  sechs Modellen tragen die Positionen nach der Aussage den alten Wert weiter, in drei liest die Antwort ihn direkt.
  Pythia-160M zeigt auf zwei von drei Datensätzen keinen gesicherten Rest.
- *Bezug:* Die Knockout-Logik ließe sich auf LS-NEQ übertragen: Blockiert man die Aufmerksamkeit späterer Positionen auf
  die Wechselstelle, zeigt sich, ob der Schaden über diese weitergetragen wird. Das ergänzt die Kontrolle „lokal oder
  Wechsel“.

**Godoy Ortiz: Meaning Contracts: Local Semantic Geometry in Large Language Models.** 22.10.2026.
[arXiv:2610.30715](https://arxiv.org/abs/2610.30715)

- *Inhalt:* Die rohe Kosinus-Geometrie der Satzdarstellungen ist nach Sprache geordnet (gezeigt an Llama und Qwen); erst
  nach Entfernen der dominanten Richtungen ordnen sich Satzpaare über ein Band von Schichten nach Bedeutung. Bei
  Pythia-2.8B ist diese Ordnung schwach: unter Mittelung über die Tokens in 21 % der Schichten, am letzten Token in keiner.
- *Bezug:* Unser Schaden ist ein Abstand in den rohen Koordinaten. Eine Robustheitsprüfung mit Schaden nach Entfernen der
  dominanten Richtungen würde zeigen, ob der Wechseleffekt auch außerhalb dieser Richtungen besteht.

**Ghosh, Kumar, Garg: Measurement Validity and Representation Dynamics of Confidence Signals in Large Language Models.**
19.10.2026. [arXiv:2610.25846](https://arxiv.org/abs/2610.25846)

- *Inhalt:* Ein präregistriertes Audit von Konfidenzsignalen; eine reine Oberflächen-Baseline erreicht oft dieselbe Güte
  wie Sonden auf internen Zuständen, und zwei von fünf Hypothesen wurden widerlegt.
- *Bezug:* Ein Vorbild für eine Oberflächen-Baseline zur Kontrolle aus Nachtrag 6.11 (λ nur aus der Sprache des
  unmittelbaren Kontexts vorhersagen) und für die getrennte Ausweisung von präregistrierten und nachträglichen Analysen.

### Zur Umkehr der Entropie-Richtung

**Mei, Tian, Yuan u. a.: API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood
Tails.** 27.10.2026. [arXiv:2610.36801](https://arxiv.org/abs/2610.36801)

- *Inhalt:* Ein Prüfverfahren für Sprachmodell-Schnittstellen auf Basis von „Überraschung minus Entropie“. Im
  Methodenteil steht, dass die Ableitung der Log-Wahrscheinlichkeit eines Tokens nach der Temperatur
  (Überraschung − Entropie)/T beträgt: Eine Temperaturerhöhung hebt Tokens, deren Überraschung über der Entropie liegt, und
  senkt die übrigen.
- *Bezug:* Die Regel, mit der sich die `ent`-Umkehr erklären ließe, falls `ent` in erster Näherung wie eine
  Temperaturerhöhung wirkt (siehe den Bericht zu Endpunkt 6). Test: je Variante eine effektive Temperatur schätzen;
  Abweichungen zeigten, dass `ent` mehr tut als eine Temperaturänderung.

**Komanduri, Afshar, Miller, Gao: Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness.**
15.10.2026. [arXiv:2610.22636](https://arxiv.org/abs/2610.22636)

- *Inhalt:* Misst, ab welcher Stärke eines internen Eingriffs sich eine Antwort ändert. Bei falschen und unsicheren
  Antworten genügt ein schwächerer Eingriff, und die Schwelle hängt mit der Entropie der Ausgabe zusammen.
- *Bezug:* Ein Hinweis, die Entropie an der Injektionsstelle in der Kontrolle aus Nachtrag 6.11 als Begleitgröße
  mitzuführen.

### Zu SAE-Transfer und Checkpoints

**Ghosh, Mathai, Babu, Agarwal: Diminishing Returns of Data in Sparse Autoencoders: Functional Behavior does not
Guarantee Feature Reproducibility.** 14.10.2026. [arXiv:2610.20680](https://arxiv.org/abs/2610.20680)

- *Inhalt:* Die Rekonstruktion eines SAE ist oft schon mit wenig Daten gut, während die Eins-zu-eins-Reproduzierbarkeit der
  Merkmale mit mehr Daten weiter steigt (Gemma-2-2B, BatchTopK: 2 % bei 5M Tokens, 29 % bei 50M); ähnlich bei Pythia-160M
  (Layer 6).
- *Bezug:* Unsere FVU von 0,198 ist ein funktionales Maß und zeigt nicht, dass bei step98000 dieselben Merkmale vorliegen.
  Test: ein eigenes SAE auf step98000 trainieren und Merkmal für Merkmal mit dem SAE von step143000 abgleichen.

**He, Shen, Iacob u. a.: Learning to Read Out: Unembedding Dynamics in Language Model Pretraining.** 14.10.2026, laut
arXiv NeurIPS 2026. [arXiv:2610.21025](https://arxiv.org/abs/2610.21025)

- *Inhalt:* Hidden States können eine Information schon tragen, bevor die Ausgabeschicht sie ausdrückt. Ein Wörterbuch über
  die Checkpoints von Pythia datiert einen Umbau der Ausgabeschicht auf ein enges frühes Fenster bei 0,4–1,1 % des
  Trainings. Ein SAE auf den Ausgabegewichten des letzten Checkpoints versagt an frühen Checkpoints (Gewichte, nicht
  Aktivierungen).
- *Bezug:* Über den Abschnitt step98000 bis step143000 sagt das wenig, weil der Befund sehr früh im Training liegt. Ein
  Crosscoder über die Aktivierungen beider Checkpoints würde die Merkmale aber direkt vergleichbar machen.

**Gupta, Zhang, Jin: Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism.** 16.10.2026.
[arXiv:2610.22937](https://arxiv.org/abs/2610.22937)

- *Inhalt:* Tauscht man in Tulu-3-8B und OLMo-2-7B die Attention-Muster von sechs mittleren Schichten zwischen dem
  RL-Checkpoint und seinem DPO-Vorgänger, rückt der Effekt von Hinweisen auf eine vorgeschlagene Antwort größtenteils auf
  das Niveau des jeweils anderen Checkpoints.
- *Bezug:* Methodisch übertragbar: Attention-Muster zwischen step98000 und step143000 tauschen und messen, wie viel vom
  Checkpoint-Unterschied in Δλ mitwandert.

### Etwas älter, aber ebenfalls ganz neu

In den Themenabschnitten stehen zwei Arbeiten von Anfang Oktober 2026, die direkt zu LS-NEQ passen: Finlayson u. a.
(3.10.2026, Abschnitt 1: Becken im Aktivierungsraum) und Tao u. a. (1.10.2026, Abschnitt 5: In sechs Pythia-160M-Läufen
ist die Ausgabereaktion auf eine gleich große relative Störung entlang der Richtung des fehlenden Block-Updates bei
step143000, gemittelt über vier Stärken, kleiner als bei step14000; bei der kleinsten Stärke ist sie größer, und unsere
Dosen liegen noch darunter).

---

## 7. Anregungen für nächste Schritte aus der Literatur

1. **Vergleichsrichtungen:** neben isotropen Zufallsrichtungen auch Mischungs-Richtungen (Differenz zweier
   kovarianzangepasster Zufallsaktivierungen oder zweier anderer echter Aktivierungen; Lee und Heimersheim 2024) und
   Zufallsdifferenz-Richtungen (Ferreira da Silva und Heimersheim 2026).
2. **Normwachstum:** Die Residualnorm wächst über die Layer; Störungen werden deshalb relativ zur Norm des ungestörten
   Zustands gemessen (so auch Chen u. a. 2026 und Tao u. a. 2026). Das ist bei uns schon so: D ist relativ zur Norm des
   ungestörten Zustands (`lsne/dynamics.py`, `DamageMap`).
3. **Störungsgröße und Genauigkeit:** den Kontrast über weitere Dosen und eine zweite Gleitkommagenauigkeit prüfen
   (Islam u. a. 2026; Jones 2026).
4. **Entropie-Richtung:** zuerst die Ziel-Regime-Kontrolle aus dem Bericht zu Endpunkt 6 (die Wirkung von `ent` auf im
   Kontext unwahrscheinliche Wörter auslesen; Vorhersage: positiv). Danach eine je Familie neu berechnete Richtung (Li u. a.
   2026) und messen, wie viel von `ent` im fast wirkungslosen Unterraum der Unembedding-Matrix liegt (Stolfo u. a. 2024;
   Cancedda 2024).
5. **Sprachachse und Neuronen:** prüfen, ob `lang_dom` mit Deutsch-Kontext-Neuronen oder deutschen SAE-Merkmalen
   zusammenfällt und ob beides über die Checkpoints gemeinsam entsteht (Quirke u. a. 2023). Außerdem die einfache
   Mittelwertdifferenz als Vergleichsachse rechnen (Chang u. a. 2022).
6. **SAE über Checkpoints:** FVU desselben SAE bei step143000 als Basis bestimmen, dann FVU und Einsetztest als Funktion
   des Checkpoint-Abstands, mit und ohne Positionen großer Norm; ein Crosscoder für step98000 und step143000 (Kissane u. a.
   2024; Ge u. a. 2026).
7. **Jacobi-Analyse:** Überlapp von `lang_dom` mit den führenden Singulärvektoren der Jacobi-Matrix von Layer 4 bis 12
   (Fernando und Guitchounts 2026; Ostrow u. a. 2026).
8. **Seeds:** den Checkpoint-Trend an weiteren Seeds von Pythia-160m prüfen (PolyPythias; Yao u. a. 2026).
9. **Lokal oder weitergetragen:** mit Attention-Knockout prüfen, ob der Schaden über die Positionen nach der Wechselstelle
   weitergetragen wird (Logik aus Ding 2026); das ergänzt die Kontrolle aus Nachtrag 6.11.
10. **Checkpoints prüfen:** die geladenen Checkpoints per Prüfsumme auf Eindeutigkeit kontrollieren (Oh u. a. 2026 fanden
    im Repository von Pythia-2.8B falsch belegte Checkpoints: Unter step54000 bis step64000 liegen die Daten von
    step143000).

---

## 8. Was die Suche nicht gefunden hat

Die Agenten haben gezielt danach gesucht und nichts gefunden. Das heißt nicht sicher, dass es nichts gibt, aber diese
Punkte scheinen kaum untersucht:

- Eine Arbeit, die einen Lyapunov-artigen Wachstumsexponenten über die Tiefe entlang einer Sprachachse gegen
  Zufallsrichtungen misst, abhängig von der Zahl der Sprachwechsel, oder so etwas entlang einer Sprachachse über
  Pythia-Checkpoints verfolgt. Verwandte Messungen ohne Sprachachse gibt es: Tao u. a. 2026 vergleichen in sechs
  Pythia-160M-Läufen bei step14000 und step143000, wie stark relative Störungen über die Tiefe abklingen, für die Richtung
  des fehlenden Block-Updates gegen Zufallsrichtungen; Chen u. a. 2026 verfolgen Quantisierungsfehler über die Tiefe an
  Checkpoints von Pythia-1.4B und -2.8B. Ferreira da Silva und Heimersheim 2026 vergleichen Merkmals- mit
  Zufallsrichtungen, aber ohne Wachstumsexponent, Wechsel oder Checkpoints; Fernando und Guitchounts 2026 messen
  Jacobi-Spektren an drei OLMo-Checkpoints.
- Eine mechanistische Studie zu deutsch-englisch gemischten Eingaben in kleinen englischdominanten Modellen. Keine Arbeit
  variiert die Zahl der gewechselten Einheiten oder betrachtet jede Wechselstelle einzeln über die Tiefe.
- Eine Arbeit, die mit dem Entropie-Gradienten nach dem Residualstrom steuert oder prüft, ob ein an einem Zustand
  berechneter Entropie-Gradient auf andere Kontexte übertragbar ist. Für den Gradienten eines gelernten Konzept-Scores
  zeigen Li, Li, Huang (2026, Abschnitt 3), dass eine seltener neu berechnete Richtung schwächer steuert.
- Ein veröffentlichter Test, ob ein SAE des letzten Checkpoints die Aktivierungen der Zwischen-Checkpoints desselben
  Modells rekonstruiert. He u. a. (2026, Abschnitt 6) tun das nur für die Gewichte der Ausgabeschicht.

Auch unter den neuesten Arbeiten (Abschnitt 6) untersucht keine Sprachwechsel zusammen mit Störungsausbreitung. Die
Kombination aus kontextabhängigem λ, Richtungsspezifität und Trainingsverlauf scheint demnach neu zu sein.
