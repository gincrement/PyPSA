# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Synchronous condensers components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import xarray as xr


@patch_add_docstring
class SynchronousCondensers(Components):
    """Synchronous condensers components class.

    A synchronous condenser is a synchronous machine without prime mover. It provides
    reactive power for voltage control in the power flow and inertia and short-circuit
    power to the system. In the optimisation, its operational variable `s` is the
    online capacity, which draws active power losses `loss * s` from the bus and
    contributes to global constraints of type `inertia` and `short_circuit_power`.
    These requirements drive its dispatch and expansion.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.synchronous_condensers
    Empty 'SynchronousCondenser' Components

    """

    _operational_variables = ["s"]

    def get_bounds_pu(
        self,
        attr: str = "s",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds of the online capacity of synchronous condensers.

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "s"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = (
                "Bounds can only be retrieved for operational attributes. For "
                "synchronous condensers those are: "
                f"{list_as_string(self._operational_variables)}."
            )
            raise ValueError(msg)

        return self.da.s_min_pu, self.da.s_max_pu

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
