# bioviper2

Biological sequence analysis toolkit built on numpy and pandas without a biopython dependency.

bioviper2 is a sequel to my previous package bioviper, which wrapped biopython functions for processing MSAs, protein structures, and phylogenetic trees with some additional code to make them more usable. I have unfortunately stopped updating this package, partly because dealing with biopython has become unwieldy and I've come to feel that it is more parsimonious to just build directly on top of pandas and inherit directly from dataframe structues rather than create some Frankenstein of biopython and my own whims as a 2nd-year PhD student.

The key goal of this package is just to have a centralized collection of utilities to read and write sequence data from a variety of formats to a pandas dataframe, and additionally to define a data class that stores both the alignment as a dataframe and all of its associated metadata as a secondary data frame.

This package was developed with help from claude code.

## Features

- **MSA class** — memory-efficient multiple sequence alignment backed by a numpy `U1` array, with pandas-style `.loc`/`.iloc` indexing
- **I/O** — read and write FASTA, Stockholm, Clustal, A3M; unaligned sequences to/from DataFrame; CSV export
- **Analysis** — per-position conservation and Shannon entropy, per-sequence coverage, pairwise identity matrix

## Installation

```bash
pip install -e .
```

## Quick start

```python
import bioviper2 as bv

# Load an alignment
msa = bv.read("alignment.fasta")

# Subset
msa.loc["seq1":"seq10"]          # label slice (rows)
msa.iloc[:, 0:50]                # first 50 positions

# Analysis
msa.conservation()               # per-position fraction identical
msa.entropy()                    # per-position Shannon entropy (bits)
msa.coverage()                   # per-sequence fraction non-gap
msa.pairwise_identity()          # n×n identity DataFrame

# Load unaligned sequences
seqs = bv.read_sequences("proteins.fasta")   # returns pd.DataFrame
seqs[seqs.length > 200]                      # standard pandas filtering

# Write
bv.write(msa, "out.sto")                     # Stockholm
bv.write_sequences(seqs, "out.csv")          # CSV
```

## Supported formats

| Format | Read | Write |
|---|---|---|
| FASTA (`.fa`, `.fasta`, `.fna`, `.faa`) | ✓ | ✓ |
| Stockholm (`.sto`, `.stk`) | ✓ | ✓ |
| Clustal (`.aln`, `.clw`) | ✓ | ✓ |
| A3M (`.a3m`) | ✓ | ✓ |
| CSV (`.csv`) | — | ✓ |
