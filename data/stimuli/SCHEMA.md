# Stimulus bank schema (families)

One JSON file per family **and** formulation: `F01A.json`, `F01B.json`, …, `DEV01A.json`.
Formulation B is an independent re-phrasing of the same 10 meaning units (same events,
same order, same proper names, different wording), as required by the dossier.

```json
{
  "family_id": "F01",
  "formulation": "A",
  "title": "Short title",
  "topic": "one line",
  "units": [
    {"id": 1,
     "E": {"subj": "Kraft", "pred": "reads the report", "adj": "in the morning"},
     "D": {"subj": "Kraft", "pred": "liest den Bericht", "adj": "am Morgen"}},
    ...
    {"id": 10, ...}
  ],
  "tail_E": "The last word of the report is",
  "target": " Fewkoombey",
  "target_note": "why this target is predictable only from the context",
  "author": "...",
  "notes": "anything a reviewer must know"
}
```

## Hard rules (checked by `lsne.stimuli` and `tests/test_stimuli.py`)

1. Exactly 10 units; every unit has `E` and `D`, each with non-empty `subj`, `pred`, `adj`.
2. Segments carry **no** leading/trailing whitespace and **no** final punctuation; the
   renderer adds spaces and the final period: `subj + " " + pred + " " + adj + "."`.
3. `subj` is the grammatical subject (a noun phrase), `pred` is the finite verb plus its
   object/complement, `adj` is an adjunct (time, place, manner, prepositional phrase).
   This is what makes the C2 (switch before `adj`) and C3 (switch between `subj` and
   `pred`, i.e. across subject–verb agreement) constructions meaningful.
4. The German `D` version of a unit must have the same meaning, the same proper names,
   the same tense and polarity as `E`. No added or removed information.
5. Unit 1 and unit 10 are always rendered in English in every condition; unit 9 is
   always German in mixed conditions. Write them so that both readings are natural.
6. `tail_E` is an English clause appended after unit 10 (joined by a single space) that
   makes the `target` the natural next token. The `target` must start with a space and
   tokenize to exactly one Pythia token when appended to `tail_E`.
7. The `target` word must not appear anywhere in the units or in `tail_E` (no leak).
8. Avoid the words of the original State 2143 passage (Kafka, Kraft, Brecht, Josefine,
   Fewkoombey) outside the `original_2143` case study, except in the single example
   family taken from the dossier (`DEV01`).
9. Keep units short: 4–9 words in English. Mixed sentences from these segments are
   read by bilingual reviewers for plausibility, but word-for-word naturalness across
   the language boundary is not required.
10. Use proper names and concrete nouns that are stable across languages (names are
    not translated).

## Review file

`data/stimuli/review.csv` has one row per (family_id, formulation, unit): columns
`family_id, formulation, unit, meaning_ok, grammar_E_ok, grammar_D_ok, comment`.
Reviews are written **before** any model results are seen (dossier: Textqualität,
verblindet). Rows with `meaning_ok = 0` exclude the whole family/formulation.
