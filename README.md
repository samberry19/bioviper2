# bioviper2

Biological sequence analysis toolkit built on numpy and pandas without a biopython dependency.

bioviper2 is a sequel to my previous package bioviper, which wrapped biopython functions for processing MSAs, protein structures, and phylogenetic trees with some additional code to make them more usable. I have unfortunately stopped updating this package, partly because dealing with biopython has become unwieldy and I've come to feel that it is more parsimonious to just build directly on top of pandas and inherit directly from dataframe structues rather than create some Frankenstein of biopython and my own whims as a 2nd-year PhD student.

The key goal of this package is just to have a centralized collection of utilities to read and write sequence data from a variety of formats to a pandas dataframe, and additionally to define a data class that stores both the alignment as a dataframe and all of its associated metadata as a secondary data frame.

This package was developed with help from claude code.

## Features

- **MSA class** — memory-efficient multiple sequence alignment backed by a numpy `U1` array, with pandas-style `.loc`/`.iloc` indexing
- **Structure class** — protein structure backed by a numpy coordinate array and a parallel atom table; reads PDB and mmCIF; `.select()` by chain, residue, atom, element, or hetero flag
- **DistanceMatrix class** — labeled pairwise distance matrix with re-selectable axes and MultiIndex pandas access; supports CA–CA, CB–CB, all-atom, and residue minimum-distance modes
- **I/O** — read and write FASTA, Stockholm, Clustal, A3M, PDB, mmCIF; unaligned sequences to/from DataFrame; CSV export
- **Analysis** — per-position conservation, Shannon entropy, and alignment depth; per-sequence occupancy; pairwise identity matrix

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
msa.coverage()                   # per-position depth (fraction of sequences non-gap)
msa.coverage(per='sequence')     # per-sequence occupancy (fraction of columns non-gap)
msa.pairwise_identity()          # n×n identity DataFrame

# Load unaligned sequences
seqs = bv.read_sequences("proteins.fasta")   # returns pd.DataFrame
seqs[seqs.length > 200]                      # standard pandas filtering

# Write
bv.write(msa, "out.sto")                     # Stockholm
bv.write_sequences(seqs, "out.csv")          # CSV
```

## Structure and distance matrices

```python
# Load a structure
s = bv.read_structure("1abc.pdb")   # or .cif / .mmcif
s = bv.read_pdb("1abc.pdb")
s = bv.read_mmcif("1abc.cif")

# Inspect
s                           # Structure(1234 atoms, 2 chains [A, B], 156 residues, 1 model)
s.chains                    # array(['A', 'B'])
s.n_residues                # 156
s.atoms                     # pandas DataFrame — one row per atom
s.coords                    # numpy (n_atoms, 3) float64 array
s.atoms["b_factor"]         # B-factors / AlphaFold pLDDT scores
s.plddt                     # same, named alias for AlphaFold files

# Selection — mirrors pandas semantics, returns a sub-Structure
s.select(chain="A")
s.select(chain="A", atom="CA")
s.select(resi=range(50, 61))         # residues 50–60 by author number
s.select(hetero=False)               # drop waters and ligands
s.select(element="FE")
s.iloc[:100]                         # first 100 atoms by position

# Export
s.to_dataframe()             # atoms + x/y/z columns in one flat DataFrame
bv.write_structure(s, "out.pdb")
bv.write_structure(s, "out.cif")

# Distance matrices — axes labeled by real residue/atom identifiers
dm = s.distance_matrix("ca")          # Cα–Cα, one row/col per residue
dm = s.distance_matrix("cb")          # Cβ–Cβ (CA fallback for glycine)
dm = s.distance_matrix("all_atom")    # every atom × every atom
dm = s.distance_matrix("min")         # residue×residue minimum inter-atom distance

dm.shape                     # (n, n)
dm.values                    # raw numpy array
dm.labels                    # DataFrame describing each row/col

# Pandas MultiIndex access by real residue number (not array position)
df = dm.to_dataframe()
df.loc[("A", 50, ""), ("A", 60, "")]  # distance between res 50 and 60

# Re-select both axes simultaneously — same API as Structure.select
dm.select(chain="A")
dm.select(chain="A", resi=range(50, 61))
dm.select(resi=2)                     # filters by author res_seq, not array index

# atom-level matrix allows atom/element/hetero filtering too
s.distance_matrix("all_atom").select(chain="A", atom="CA")
```

## Supported formats

| Format | Read | Write |
|---|---|---|
| FASTA (`.fa`, `.fasta`, `.fna`, `.faa`) | ✓ | ✓ |
| Stockholm (`.sto`, `.stk`) | ✓ | ✓ |
| Clustal (`.aln`, `.clw`) | ✓ | ✓ |
| A3M (`.a3m`) | ✓ | ✓ |
| PDB (`.pdb`, `.ent`) | ✓ | ✓ |
| mmCIF (`.cif`, `.mmcif`) | ✓ | ✓ |
| CSV (`.csv`) | — | ✓ |
