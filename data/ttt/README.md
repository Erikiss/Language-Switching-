# `data/ttt/` — Korpus für das Test-Time-Training (DESIGN §4.8d, Stufe K)

Dieses Verzeichnis ist der Ablageort für das **Kontrastkorpus** des Test-Time-Training-Arms
(TTT, „strukturelle Injektion“): fortgesetztes Training eines eingefrorenen Checkpoints
auf einem kleinen Korpus zum **Ethereum/DAO-Fork 2016** (einer Spaltung), gemessen als
Memorisationsgeschwindigkeit pro Textstück sowie als Vorher/Nachher-Vergleich aller
übrigen Kenngrößen (`lsne.ttt`, Notebook `10_ttt_injection`).

**Nichts in diesem Repository enthält das DAO-Fork-Korpus.** Es ist vom Auftraggeber
bereitzustellen (Herkunft, Rechte und Auswahl liegen bei ihm) und wird nicht eingecheckt.
Der Code enthält nur den Loader und einen kleinen, gemeinfreien Platzhalter, damit Tests
und Smoke-Läufe ohne das Korpus funktionieren.

## Wohin das Korpus kommt

```
data/ttt/
├── README.md                       (diese Datei)
├── placeholder_public_domain.txt   (Platzhalter, im Repository)
├── dao_fork/                       <- hier das DAO-Fork-Korpus ablegen (*.txt, *.jsonl)
├── neutral/                        <- Kontrollarm (i): neutrales Korpus gleicher Größe
└── nofork/                         <- Kontrollarm (ii): gleiche Ära/Domäne ohne Spaltungsnarrativ
```

Die Verzeichnisse `dao_fork/`, `neutral/` und `nofork/` werden vom Nutzer angelegt. Wer sie
vor versehentlichem Einchecken schützen will, trägt sie in `.gitignore` ein (z. B.
`data/ttt/dao_fork/`).

## Erwartetes Format

Der Loader `lsne.ttt.load_corpus(pfad, tokenizer, max_docs=None, seq_len=512)` nimmt ein
Verzeichnis (nicht rekursiv, Dateien in sortierter Reihenfolge) oder eine einzelne Datei:

| Endung   | Inhalt |
|----------|--------|
| `*.txt`  | UTF-8-Klartext; **ein Dokument pro Datei** oder mehrere Dokumente, getrennt durch eine Zeile, die nur aus `---` besteht |
| `*.jsonl`| eine JSON-Zeile pro Dokument mit dem Feld `"text"`; Zeilen ohne dieses Feld werden übersprungen |

Jedes Dokument wird ohne Sondertoken tokenisiert und in Stücke von höchstens `seq_len`
Tokens zerlegt (Stücke mit weniger als zwei Tokens entfallen). Ergebnis ist eine Liste
von `LongTensor[T]` (Klasse `Corpus`, eine `list` mit `meta`-Dict: Quelle, Anzahl
Dokumente/Stücke/Tokens, `is_placeholder`, `control`).

Fehlt der Pfad oder enthält er keine Dokumente, liefert der Loader den Platzhalter und
markiert das: `corpus.meta["is_placeholder"] == True` bzw. Modulflag
`lsne.ttt.LAST_LOAD_WAS_PLACEHOLDER`. Ergebnisse mit Platzhalter sind Funktionstests, keine
Befunde.

```python
from lsne.config import ModelSpec
from lsne.model import load_model
from lsne import ttt

mb = load_model(ModelSpec(revision="step98000", dtype="float32"))
corpus = ttt.load_corpus("data/ttt/dao_fork", mb.tokenizer, seq_len=256)
assert not corpus.is_placeholder, "DAO-Korpus fehlt"
controls = {
    "neutral": ttt.load_corpus("data/ttt/neutral", mb.tokenizer, seq_len=256),
    "nofork": ttt.load_corpus("data/ttt/nofork", mb.tokenizer, seq_len=256),
    "permuted": ttt.permute_corpus(corpus, seed=0),
}
table = ttt.ttt_memorisation_table(mb, corpus, controls, checkpoints=["step4000", "step98000", "step143000"],
                                   lr=1e-4, steps=50, threshold=0.5)
mb_ttt = ttt.test_time_train(mb, corpus, lr=1e-4, steps=200, batch=4, seed=0)   # Kopie, mb bleibt unverändert
mb_neu = ttt.test_time_train(mb, corpus, control="neutral", controls=controls, lr=1e-4, steps=200)
```

## Die drei Kontrollarme (DESIGN §4.8d, Punkt 3)

Ein Effekt gilt nur dann als „strukturelle Injektion“, wenn er über **allen drei**
Kontrollarmen liegt; alle Arme erhalten dasselbe Trainingsbudget (gleiche Schrittzahl,
Lernrate, Batchgröße, Seed).

| Arm        | Korpus | Bereitstellung |
|------------|--------|----------------|
| `neutral`  | thematisch neutrales Korpus **gleicher Größe** (Tokenzahl, Stückzahl, Stücklänge) aus einer anderen Domäne | Nutzer: `data/ttt/neutral/` |
| `nofork`   | gleiche Ära und Domäne (Kryptowährungs-/Ethereum-Texte 2016), aber **ohne Spaltungs-/Fork-Narrativ** | Nutzer: `data/ttt/nofork/` |
| `permuted` | das DAO-Korpus mit **permutierten Tokens** je Stück (gleiche Token-Multimenge, zerstörte Reihenfolge) | automatisch: `lsne.ttt.permute_corpus(corpus, seed)`; keine Dateien nötig |

`test_time_train(..., control="neutral" | "nofork")` verlangt das passende Korpus im
Argument `controls={...}` und bricht mit einer klaren Fehlermeldung ab, wenn es fehlt;
`control="permuted"` erzeugt die Permutation selbst.

## Trainingskonventionen

AdamW (Weight Decay einstellbar), Gradientennorm-Clipping bei 1,0, FP32-Masterweights
(ein Modell in BF16/FP16 wird abgelehnt), gesäte und deterministische Batch-Reihenfolge,
Padding mit dem Pad-/EOS-Token des Tokenizers, Attention-Maske, Labels `-100` auf Padding.
`memorisation_speed` arbeitet auf einem State-Dict-Snapshot und stellt ihn in jedem Fall
wieder her; `test_time_train` liefert standardmäßig eine tief kopierte Kopie des Modells.
Das übergebene Modell bleibt unverändert, sofern nicht `inplace=True` gesetzt ist.

## Platzhalter

`placeholder_public_domain.txt` enthält rund 30 kurze Absätze erzählender Prosa, die eigens
für dieses Repository verfasst und als gemeinfrei (CC0) freigegeben wurden. Sie handeln von
einer fiktiven Dorfgemeinschaft, die sich nach einer umstrittenen Entscheidung über ihr
gemeinsames Buch in zwei Gruppen mit zwei Büchern teilt — ein thematischer Stellvertreter
für ein Fork-Ereignis. Es sind keine historischen Dokumente, keine Zitate und keine Texte
aus anderen Quellen. Absätze sind durch `---`-Zeilen getrennt (ein Dokument je Absatz).
