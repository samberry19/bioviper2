"""Wrappers for HMMER 3 command-line tools (hmmalign, hmmbuild, hmmsearch).

Each function invokes the corresponding HMMER binary as a subprocess, handles
temporary files transparently, and returns native bioviper2 / pandas objects.
HMMER 3 must be installed and available on PATH.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..msa import MSA
from ..io.fasta import write_fasta_sequences
from ..io.stockholm import read_stockholm, write_stockholm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_seq_series(sequences) -> pd.Series:
    """Normalise various sequence input formats to a Series(id → seq_str)."""
    if isinstance(sequences, pd.DataFrame):
        if "sequence" not in sequences.columns:
            raise ValueError(
                "DataFrame input must have a 'sequence' column "
                "(as returned by read_fasta_sequences)."
            )
        return sequences["sequence"]
    if isinstance(sequences, pd.Series):
        return sequences
    if isinstance(sequences, dict):
        return pd.Series(sequences)
    if isinstance(sequences, list):
        return pd.Series({f"seq{i}": s for i, s in enumerate(sequences)})
    if isinstance(sequences, str):
        return pd.Series({"seq0": sequences})
    raise TypeError(
        f"sequences must be a DataFrame, Series, dict, list, or str; "
        f"got {type(sequences).__name__}."
    )


def _run(cmd: list[str], tool: str) -> None:
    """Run a subprocess, raising informative errors on failure."""
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{tool}' not found on PATH.  Install HMMER (https://hmmer.org)."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{tool} exited with status {exc.returncode}:\n{exc.stderr.strip()}"
        ) from exc


def _parse_tblout(path: Path) -> pd.DataFrame:
    """Parse an hmmsearch/hmmscan --tblout file into a DataFrame.

    Columns: target, evalue, score, bias, evalue_best_domain,
             score_best_domain, description.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 19:
                continue
            rows.append({
                "target":             fields[0],
                "evalue":             float(fields[4]),
                "score":              float(fields[5]),
                "bias":               float(fields[6]),
                "evalue_best_domain": float(fields[7]),
                "score_best_domain":  float(fields[8]),
                "description":        " ".join(fields[18:]),
            })

    cols = ["target", "evalue", "score", "bias",
            "evalue_best_domain", "score_best_domain", "description"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("evalue")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hmmalign(
    hmm: Union[str, Path],
    sequences,
    *,
    trim: bool = False,
    seqtype: Optional[str] = None,
    mapali: Optional[Union[str, Path]] = None,
) -> MSA:
    """Align sequences to a profile HMM using ``hmmalign``.

    Parameters
    ----------
    hmm:
        Path to an HMM file (built with :func:`hmmbuild` or downloaded from
        Pfam, etc.).
    sequences:
        Sequences to align.  Accepts:

        * ``pd.DataFrame`` with a ``'sequence'`` column (as returned by
          :func:`~bioviper2.read_fasta_sequences`)
        * ``pd.Series`` mapping sequence ID → sequence string
        * ``dict`` mapping sequence ID → sequence string
        * ``list`` of sequence strings (auto-named ``seq0``, ``seq1``, …)
        * a single sequence string (named ``seq0``)

    trim:
        Remove unaligned terminal tails from the output alignment.
    seqtype:
        Force alphabet detection: ``'amino'``, ``'dna'``, or ``'rna'``.
        If ``None``, hmmalign auto-detects.
    mapali:
        Path to the seed alignment used to build the HMM.  Those sequences
        are included in the output alongside the new sequences (``--mapali``).

    Returns
    -------
    MSA
        Aligned sequences as an :class:`~bioviper2.MSA` object.

    Raises
    ------
    FileNotFoundError
        If ``hmmalign`` is not found on ``$PATH``.
    RuntimeError
        If ``hmmalign`` exits with a non-zero status.
    """
    seq_series = _to_seq_series(sequences)
    seq_df = seq_series.rename("sequence").to_frame()

    cmd = ["hmmalign"]
    if trim:
        cmd.append("--trim")
    if seqtype in ("amino", "dna", "rna"):
        cmd.append(f"--{seqtype}")
    if mapali is not None:
        cmd.extend(["--mapali", str(mapali)])
    cmd.extend(["--outformat", "Stockholm"])

    seq_tmp = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".sto", delete=False)
    seq_path = Path(seq_tmp.name)
    out_path = Path(out_tmp.name)
    seq_tmp.close()
    out_tmp.close()

    try:
        write_fasta_sequences(seq_df, seq_path)
        cmd.extend(["-o", str(out_path), str(hmm), str(seq_path)])
        _run(cmd, "hmmalign")
        return read_stockholm(out_path)
    finally:
        seq_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def hmmbuild(
    msa: Union[MSA, str, Path],
    output: Union[str, Path],
    *,
    name: Optional[str] = None,
    seqtype: Optional[str] = None,
) -> Path:
    """Build a profile HMM from a multiple sequence alignment using ``hmmbuild``.

    Parameters
    ----------
    msa:
        Input alignment as an :class:`~bioviper2.MSA` object, or a path to an
        alignment file in any format recognised by hmmbuild (Stockholm, FASTA,
        A2M, …).
    output:
        Path where the HMM file will be written.
    name:
        Name for the HMM profile.  Defaults to the output file stem.
    seqtype:
        Force alphabet: ``'amino'``, ``'dna'``, or ``'rna'``.

    Returns
    -------
    Path
        Absolute path to the written HMM file.

    Raises
    ------
    FileNotFoundError
        If ``hmmbuild`` is not found on ``$PATH``.
    RuntimeError
        If ``hmmbuild`` exits with a non-zero status.
    """
    output = Path(output)

    cmd = ["hmmbuild", "-o", "/dev/null"]  # suppress summary stdout
    if name:
        cmd.extend(["-n", name])
    if seqtype in ("amino", "dna", "rna"):
        cmd.append(f"--{seqtype}")

    if isinstance(msa, MSA):
        msa_tmp = tempfile.NamedTemporaryFile(suffix=".sto", delete=False)
        msa_path = Path(msa_tmp.name)
        msa_tmp.close()
        try:
            write_stockholm(msa, msa_path)
            cmd.extend([str(output), str(msa_path)])
            _run(cmd, "hmmbuild")
        finally:
            msa_path.unlink(missing_ok=True)
    else:
        cmd.extend([str(output), str(msa)])
        _run(cmd, "hmmbuild")

    return output.resolve()


def hmmsearch(
    hmm: Union[str, Path],
    sequences,
    *,
    evalue: float = 10.0,
    extra_args: Optional[list] = None,
) -> pd.DataFrame:
    """Search a sequence database with a profile HMM using ``hmmsearch``.

    Parameters
    ----------
    hmm:
        Path to an HMM file.
    sequences:
        Sequences to search.  Accepts the same types as :func:`hmmalign`, or
        a path (``str`` / :class:`~pathlib.Path`) to a FASTA file on disk.
    evalue:
        Full-sequence E-value reporting threshold (``-E``).
    extra_args:
        Additional flags passed verbatim to ``hmmsearch``
        (e.g. ``['--cut_ga']``).

    Returns
    -------
    pd.DataFrame
        One row per hit, sorted by E-value ascending, with columns:

        * ``target`` — target sequence name
        * ``evalue`` — full-sequence E-value
        * ``score`` — full-sequence bit-score
        * ``bias`` — composition bias correction
        * ``evalue_best_domain`` — best single-domain E-value
        * ``score_best_domain`` — best single-domain bit-score
        * ``description`` — target sequence description

        Returns an empty DataFrame if no hits pass the threshold.

    Raises
    ------
    FileNotFoundError
        If ``hmmsearch`` is not found on ``$PATH``.
    RuntimeError
        If ``hmmsearch`` exits with a non-zero status.
    """
    need_tmp = not isinstance(sequences, (str, Path))
    seq_path: Optional[Path] = None

    cmd = ["hmmsearch", "-E", str(evalue), "--noali"]
    if extra_args:
        cmd.extend(extra_args)

    tbl_tmp = tempfile.NamedTemporaryFile(suffix=".tbl", delete=False)
    tbl_path = Path(tbl_tmp.name)
    tbl_tmp.close()

    try:
        if need_tmp:
            seq_series = _to_seq_series(sequences)
            seq_df = seq_series.rename("sequence").to_frame()
            seq_tmp = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
            seq_path = Path(seq_tmp.name)
            seq_tmp.close()
            write_fasta_sequences(seq_df, seq_path)
            db_path = str(seq_path)
        else:
            db_path = str(sequences)

        cmd.extend(["-o", "/dev/null", "--tblout", str(tbl_path), str(hmm), db_path])
        _run(cmd, "hmmsearch")
        return _parse_tblout(tbl_path)
    finally:
        tbl_path.unlink(missing_ok=True)
        if seq_path is not None:
            seq_path.unlink(missing_ok=True)
