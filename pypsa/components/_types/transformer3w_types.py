# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""3-winding transformer types components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


@patch_add_docstring
class Transformer3WTypes(Components):
    """3-winding transformer types components class.

    This class is used for 3-winding transformer type components. All functionality
    specific to 3-winding transformer types is implemented here. Functionality for all
    components is implemented in the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.transformer3w_types
    'Transformer3WType' Components
    ------------------------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 2

    """

    _list_name = "transformer3w_types"

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
