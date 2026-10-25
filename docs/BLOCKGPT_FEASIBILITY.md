# BlockGPT als Quellmodell: Machbarkeit (Recherche 25.10.2026)

Ergebnis einer Quellenprüfung für Stufe L (DESIGN §4.9). Nur verifizierte Angaben; die
Zeitschätzungen am Ende sind eigene Schätzungen und als solche markiert.

## 1. Gibt es BlockGPT-Gewichte? Nein.

- Die arXiv-Seite (https://arxiv.org/abs/2304.12749) verlinkt weder Code noch Daten; der
  Volltext enthält keine Freigabeerklärung; die Hugging-Face-Modellsuche nach „blockgpt“
  ist leer.
- Die BlockScan-Arbeit (NeurIPS 2025, https://arxiv.org/html/2410.04039v5, Anhang D.1)
  bestätigt unabhängig: der Quellcode war nicht verfügbar; die Autoren haben BlockGPT nach
  Rücksprache mit den Originalautoren selbst nachgebaut.
- Was das Paper selbst festlegt: 8 Transformer-Layer, Kontext 512, Vokabular 100k
  (93 233 Adressen + 6 759 Funktionssignaturen, Rest → [OOV]), Zahlen auf zwei
  signifikante Stellen gerundet, ITR = Call-Trace + State-Trace (SLOAD/SSTORE) + Log-Trace,
  Breitensuche-Linearisierung, Baum-Positionsembeddings, Batch 32, AdamW, eine A100-40GB,
  Daten = alle Transaktionen der 124 kompromittierten DeFi-Anwendungen zwischen Block
  5 470 817 (19.04.2018) und 15 000 000 (21.06.2022). Nicht angegeben: Hidden-Größe,
  Head-Zahl, Parameterzahl, Trainingsdauer, verwendeter Tracer.

## 2. Die beiden inoffiziellen Repositories sind nicht als Quellmodell nutzbar

- **dm4sec/BlockGPT**: ein Commit, keine Lizenz, keine Gewichte, kein Datensatz. Der
  ITR-Builder zieht Traces per `debug_traceTransaction` ohne Tracer-Option (Struct-Logger)
  über 37 öffentliche RPC-Endpunkte, fest verdrahteter Blockbereich 18 689 700–18 689 710.
  Der Trainer baut einen WordPiece-Tokenizer (Vokabular 2 536) und ein Reformer-Modell
  (dim 512, Tiefe 6, kausal), trainiert es aber mit 15 % MLM-Maskierung; mehrere
  importierte Pakete fehlen in den Requirements; bekannte Bugs (undefinierte Globale,
  fehlende Grenzprüfung, doppelt definierte Funktion).
- **BLOCK-GPT-NEW/blockGPT**: archiviert (Juli 2026), benötigt einen modifizierten geth,
  `data/`, `model/`, `vocabulary/` enthalten nur `.gitkeep`; der Encoder ist ein nackter
  bidirektionaler `nn.TransformerEncoder` ohne Embedding-Schicht.
- **BlockScan / tx_fm** (https://github.com/nuwuxian/tx_fm) ist die einzige Codelinie mit
  autorengestütztem BlockGPT-Nachbau: `src_eth/preprocess.py` serialisiert Traces im
  ITR-Stil (`[START] [CALL] from to func gas value [INs] … [OUTs] … [END]`,
  `[START] [LOG] address topic0 … [END]`), `baselines/eth/gpt.py` ist ein
  Causal-LM-Trainingsskript. Das veröffentlichte Modell ist aber ein RoBERTa-MLM (nicht
  kausal) und das Ethereum-Korpus winzig (3 383 + 709 Transaktionen, 2020–2023).

## 3. Öffentliche RPC-Tracing für 2016 (empirisch geprüft)

Nur https://eth.drpc.org beantwortete ohne Schlüssel `debug_traceTransaction` (callTracer)
und `trace_transaction` für die DAO-Drain-Transaktion
0x0ec3f2488a93839524add10ea229e773f6bc891b4eb4794c3337d4495263790b (Block 1 718 497).
publicnode (Token nötig), cloudflare (keine Trace-Methoden), ankr (Schlüssel), 1rpc
(Limit), llamarpc (Fehler 525) fielen aus. Für Stichproben brauchbar, nicht für ein Korpus.
Ein eigener Reth-Archive-Node braucht ≥ 2,8 TB NVMe.

## 4. Empfohlener Weg: eigenes BlockGPT-artiges Modell aus Xatu-Parquet-Dateien

**ethPandaOps Xatu** stellt Parity-artige Call-Traces, Storage-Reads/Diffs, Logs und
ETH-/ERC-20-Transfers für jeden Mainnet-Block seit Genesis bereit, 1 000 Blöcke pro
Datei, CC BY 4.0, ohne Node und ohne API-Schlüssel:

```
https://data.ethpandaops.io/xatu/mainnet/databases/default/<tabelle>/1000/<startblock>.parquet
```

Tabellen: canonical_execution_block, _transaction, _traces, _logs, _contracts,
_storage_diffs, _storage_reads, _native_transfers, _erc20_transfers, _balance_diffs, …
(Übersicht: https://ethpandaops.io/posts/xatu-execution-layer/). Verifiziert wurden die
Chunks 1718000 (DAO-Drain, 50 053 Trace-Zeilen) und 1920000 (Fork-Block, 580 KB); das
gesamte Jahr 2016 (≈ 2 130 Chunks) sind etwa 1–2 GB Traces plus vergleichbar große
Storage-/Log-Tabellen.

Bauplan (Stufe L, Quelladapter):
1. ITR-Serialisierer über Xatu (Traces + Storage + Logs + Transfers) nach dem Tokenformat
   von tx_fm/preprocess.py plus [STATE]-Lese/Schreib-Tokens wie bei dm4sec; Vokabular =
   Spezialtokens + Top-N-Adressen + Top-N-Selektoren + zweistellig gerundete Zahlen.
2. Modell: HF-GPT-2-Konfiguration mit 8 Layern, d = 512, 8 Heads, Kontext 512
   (≈ 25 M Parameter ohne Embeddings), kausales Training.
3. Daten: 2016 (DAO-Ära), optional 2017–2018 für mehr Volumen. Weil BlockGPTs Vokabular
   adressbasiert ist, sähe ein 2018–2022-Modell 2016-Adressen ohnehin als [OOV]; das
   Training direkt auf 2016 ist für die DAO-Fragestellung die richtige Wahl.
4. Aufwand (eigene Schätzung): 1–2 Ingenieurtage für den Serialisierer; Training von
   ≈ 1 M gepackten 512-Token-Sequenzen (≈ 0,5 B Tokens) in der Größenordnung 1–3 Stunden
   auf einer A100; einige hunderttausend Sequenzen unter einer Stunde.

Weitere 2016-Quellen (verifiziert): Google BigQuery `crypto_ethereum` (Traces mit
`trace_type` ∈ {call, create, suicide, reward, genesis, daofork}; 1 TiB Abfragen/Monat
frei; aus dieser Sitzung nicht abfragbar), Hugging Face BlockDB-Exporte
(Transaktionen 2015–2026; ETH-+ERC-20-Transfers mit transfer_type 0/1/2, CC BY 4.0),
Zenodo 3906690 (erste 10 M Blöcke, CC0) und Zenodo 4718440 (Transaktionsgraph, CC BY 4.0).
XBlock-ETH: Spiegel mit abgelaufenem TLS-Zertifikat am 25.10.2026, Felder undokumentiert.

## 5. „Teilen“ auf diesen Daten operationalisieren

Labels werden auf den Transfer-/Trace-Tabellen definiert und auf die ITR-Sequenzen
zurückprojiziert:
- **Viele-zu-viele innerhalb einer Transaktion:** fan_out = Zahl verschiedener Empfänger,
  fan_in = Zahl verschiedener Sender pro Transaktion (native_transfers, erc20_transfers);
  „Verteilung“ bei fan_out ≥ k mit einem Sender, „Pooling“ bei fan_in ≥ k mit einem
  Empfänger. Im Tokenstrom ist das die Zahl der [CALL]-Einträge mit value > 0 bzw. der
  [LOG]-Transfer-Einträge pro Trace, also direkt aus der Sequenz lesbar.
- **Gegenparteien pro Adresse** über Zeitfenster: Grad, Anteil neuer vs. wiederkehrender
  Gegenparteien, Reziprozität.
- **State-Ebene:** Zahl verschiedener Balance-Slots, die eine Transaktion schreibt
  (storage_diffs) = Zahl der [STATE]-Schreibtokens.
- **Fork-Ära vs. Normalbetrieb** (EIP-779, https://eips.ethereum.org/EIPS/eip-779):
  Fork-Block 1 920 000; DAO 0xbb9bc244d798123fde783fcc1c72d3bb8c189413, extraBalance
  0x807640a13483f8ac783c557fcdf27be11ea4ac7a, WithdrawDAO
  0xbf4ed7b27f1d666546e30d74d50d173d20bca754 (Block 1 883 496); Drain ab Block 1 718 497.
  Segmente: vor DAO (< ≈ 1 428 000), DAO aktiv, Drain/Krise (1 718 497–1 919 999),
  Fork-Block, Rückerstattung (Transaktionen an WithdrawDAO). „Teilen“ im DAO-Sinn =
  viele-zu-eins ETH-Zuflüsse vor dem Drain, viele-zu-eins DAO-Token-Transfers nach
  WithdrawDAO nach dem Fork, eins-zu-viele ETH-Rückerstattungen danach.
- Sondensets: {hoher fan_out} vs. {fan_out = 1} (gleiche Ära, ähnliches Gas) und
  {Fork-Ära, DAO-berührend} vs. {gleiche Ära, nicht DAO}; daraus im Quellmodell die
  Difference-of-Means-Richtung bzw. die Zustände an den [CALL]/[LOG]-Positionen für die
  Transplantation.

## 6. Konsequenz für das Projekt

Nicht auf BlockGPT-Gewichte planen. Die Transplantationsmaschinerie (`lsne.transplant`)
ist mit Stand-in-Quellmodellen validiert; der Quelladapter für ein eigenes
BlockGPT-artiges Modell ist der nächste Schritt und braucht eine A100 für einige Stunden
sowie den Xatu-Download. Alternativ kann der Auftraggeber ein bereits trainiertes
Spurenmodell oder vortokenisierte Spuren bereitstellen (Ablage unter `data/s4/` bzw. einem
neuen `data/traces/`).
