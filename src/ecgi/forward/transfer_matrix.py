"""Optional sparse-factorization capability check used by production Tikhonov."""

from __future__ import annotations

import scipy.sparse as sp


def _cholmod_usable() -> bool:
    """Return whether scikit-sparse exposes a callable CHOLMOD factor."""
    try:
        from sksparse.cholmod import cholesky

        return callable(cholesky(sp.eye(3, format="csc")))
    except Exception:
        return False
