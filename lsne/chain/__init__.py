"""Trace source model (Stage L source adapter, docs/TRACE_SOURCE_SPEC.md, DESIGN §4.9).

BlockGPT-style causal LM on 2016 Ethereum execution traces from the public ethPandaOps
Xatu Parquet files (licence CC BY 4.0, attribution "ethPandaOps Xatu").

Submodules (imported explicitly, nothing is loaded eagerly here):

* :mod:`lsne.chain.xatu`   — chunk URLs, resumable download, normalised Parquet reading
* :mod:`lsne.chain.itr`    — call trees (:class:`~lsne.chain.itr.TxRecord`) and ITR serialisation
* :mod:`lsne.chain.text`   — paired English rendering for the target model
* :mod:`lsne.chain.labels` — "sharing" features / labels, DAO eras, matched probe sets
* :mod:`lsne.chain.vocab`, :mod:`lsne.chain.train`, :mod:`lsne.chain.source`,
  :mod:`lsne.chain.cli` — vocabulary, training, SourceModel adapter, command line
"""

__all__ = ["xatu", "itr", "text", "labels", "vocab", "train", "source", "cli"]
