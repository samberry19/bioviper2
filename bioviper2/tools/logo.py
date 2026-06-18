"""Sequence logo plotting via logomaker.

logomaker is a soft dependency — it is imported only inside :func:`sequence_logo`
so the rest of the package works without it installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.axes
    import logomaker as _lm


def _to_logo_df(matrix) -> pd.DataFrame:
    """Convert various matrix types to a logomaker-compatible DataFrame.

    logomaker expects a DataFrame whose index is alignment positions and
    whose columns are single characters (the alphabet).
    """
    if isinstance(matrix, pd.DataFrame):
        return matrix.copy()

    # xarray DataArray — import lazily so xarray is also a soft dependency
    try:
        import xarray as xr  # noqa: F401
    except ImportError:
        pass
    else:
        if isinstance(matrix, xr.DataArray):
            if matrix.ndim != 2:
                raise ValueError(
                    f"DataArray must be 2-D (positions × characters), "
                    f"got {matrix.ndim} dimensions."
                )
            df = matrix.to_pandas()
            # Ensure orientation: rows = positions (many), cols = alphabet (few).
            # Heuristic: if the columns look like single characters, we're good;
            # otherwise transpose.
            if df.shape[1] > df.shape[0]:
                df = df.T
            return df

    raise TypeError(
        f"matrix must be a pd.DataFrame or xr.DataArray, "
        f"got {type(matrix).__name__}.  "
        "Pass msa.site_frequencies() or msa.pairwise_frequencies().site directly."
    )


def sequence_logo(
    matrix,
    *,
    from_type: str = "probability",
    to_type: str = "information",
    ax: Optional["matplotlib.axes.Axes"] = None,
    figsize: Tuple[float, float] = (10.0, 2.5),
    **logo_kwargs,
) -> "_lm.Logo":
    """Draw a sequence logo from a frequency or probability matrix.

    Parameters
    ----------
    matrix:
        Position × character frequency or probability matrix.  Accepts:

        * ``pd.DataFrame`` — rows are positions, columns are characters.
          :meth:`~bioviper2.MSA.site_frequencies` returns this directly.
        * ``xr.DataArray`` — 2-D labelled array; transposed automatically
          if columns outnumber rows.

    from_type:
        Interpretation of the input values.  Passed to
        ``logomaker.transform_matrix``.  One of:

        * ``'probability'`` *(default)* — rows sum to 1.
        * ``'counts'`` — raw character counts per position.
        * ``'information'`` — already in bits; no transform applied.
        * ``'weight'`` — log-odds weight matrix.

    to_type:
        Output scale for the logo heights.  One of:

        * ``'information'`` *(default)* — bits (0 → log₂ A); emphasises
          conserved positions.
        * ``'probability'`` — stacked bars sum to 1 at every position.
        * ``'counts'`` — raw counts.
        * ``'weight'`` — log-odds.

        Set ``from_type`` and ``to_type`` to the same value to skip the
        transform and plot the matrix as-is.

    ax:
        Matplotlib :class:`~matplotlib.axes.Axes` to draw on.  If ``None``
        (default) a new figure of size *figsize* is created automatically.

    figsize:
        ``(width, height)`` in inches for the auto-created figure.
        Ignored when *ax* is supplied.

    **logo_kwargs:
        Extra keyword arguments forwarded verbatim to
        :class:`logomaker.Logo` (e.g. ``color_scheme``, ``font_name``,
        ``stack_order``, ``show_spines``).

    Returns
    -------
    logomaker.Logo
        The Logo object.  Access ``logo.ax`` for the Matplotlib axis and
        ``logo.ax.figure`` for the figure (e.g. to save or show).

    Raises
    ------
    ImportError
        If ``logomaker`` is not installed.
    TypeError
        If *matrix* is not a DataFrame or DataArray.

    Examples
    --------
    Basic usage from an MSA::

        freqs = msa.site_frequencies()
        logo  = bv.sequence_logo(freqs)

    Plot onto an existing axis in a multi-panel figure::

        fig, axes = plt.subplots(1, 2, figsize=(14, 2.5))
        bv.sequence_logo(freqs1, ax=axes[0])
        bv.sequence_logo(freqs2, ax=axes[1])
        plt.tight_layout()

    Use raw counts and keep the count scale::

        logo = bv.sequence_logo(msa.site_frequencies() * msa.n_seqs,
                                 from_type='counts', to_type='counts')
    """
    try:
        import logomaker
    except ImportError as exc:
        raise ImportError(
            "logomaker is required for sequence_logo().  "
            "Install it with:  pip install logomaker"
        ) from exc

    df = _to_logo_df(matrix)

    # Drop all-zero columns (characters absent from the alignment)
    df = df.loc[:, (df != 0).any(axis=0)]

    # Transform matrix unless the caller has already done it
    if from_type != to_type:
        df = logomaker.transform_matrix(df, from_type=from_type, to_type=to_type)

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=figsize)

    logo = logomaker.Logo(df, ax=ax, **logo_kwargs)
    return logo
