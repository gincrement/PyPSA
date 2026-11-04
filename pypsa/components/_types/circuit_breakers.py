# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Circuit breakers components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Branch components a circuit breaker can be attached to.
BREAKER_BRANCH_COMPONENTS = ("Line", "Transformer")


@patch_add_docstring
class CircuitBreakers(Components):
    """Circuit breakers components class.

    A circuit breaker is attached to a [Line][pypsa.components.Lines] or
    [Transformer][pypsa.components.Transformers] and opens or closes it. With a
    fixed `status`, an open breaker takes its branch out of power flow and
    optimisation in that snapshot. With `switchable=True`, the optimisation decides
    the status in each snapshot (optimal transmission switching). With
    `extendable=True`, it also decides whether to install the breaker at
    `capital_cost`; a branch can only be opened by an installed breaker.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.circuit_breakers
    Empty 'CircuitBreaker' Components

    """

    def branch_status(self, snapshots: Sequence | None = None) -> pd.DataFrame:
        """Get the fixed status of the branches with active circuit breakers.

        Parameters
        ----------
        snapshots : Sequence, optional
            Snapshots for which to return the status, defaults to all snapshots.

        Returns
        -------
        pandas.DataFrame
            Status (1 closed, 0 open) per snapshot with a column MultiIndex of
            (`component`, `branch`).

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["a", "b"])
        >>> n.add("Line", "l", bus0="a", bus1="b", x=0.1)
        >>> n.add("CircuitBreaker", "cb", branch="l", status=0)
        >>> n.c.circuit_breakers.branch_status()
        component  Line
        branch        l
        snapshot
        now         0.0

        """
        n = self.n_save
        if snapshots is None:
            snapshots = n.snapshots
        static = self.static[self.static.active]
        status = n.get_switchable_as_dense(
            "CircuitBreaker", "status", snapshots, inds=static.index
        )
        status.columns = pd.MultiIndex.from_arrays(
            [static.component, static.branch], names=["component", "branch"]
        )
        return status.astype(float)

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **kwargs: Any,
    ) -> pd.Index | None:
        """Wrap Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            return_names=return_names,
            **kwargs,
        )
