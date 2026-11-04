# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""3-winding transformer components module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network

logger = logging.getLogger(__name__)

#: Suffix of the auxiliary star bus created for each 3-winding transformer.
STAR_BUS_SUFFIX = " star"
#: Suffixes of the three 2-winding legs created for each 3-winding transformer.
LEG_SUFFIXES = (" leg0", " leg1", " leg2")
#: Columns of the leg parameters returned by `Transformers3W.star_equivalent`.
LEG_COLUMNS = [
    "bus0",
    "bus1",
    "model",
    "s_nom",
    "r",
    "x",
    "g",
    "b",
    "tap_ratio",
    "tap_side",
    "phase_shift",
    "active",
    "s_nom_extendable",
    "s_nom_min",
    "s_nom_max",
    "s_nom_set",
    "s_nom_mod",
    "capital_cost",
    "overnight_cost",
    "discount_rate",
    "fom_cost",
    "build_year",
    "lifetime",
]


def star_bus_names(names: pd.Index) -> pd.Index:
    """Names of the auxiliary star buses for the given 3-winding transformers."""
    return pd.Index(names.astype(str) + STAR_BUS_SUFFIX, name="name")


def leg_names(names: pd.Index) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Names of the three 2-winding legs for the given 3-winding transformers."""
    return tuple(  # type: ignore[return-value]
        pd.Index(names.astype(str) + suffix, name="name") for suffix in LEG_SUFFIXES
    )


def delta_to_star(
    z01: pd.Series | complex, z12: pd.Series | complex, z02: pd.Series | complex
) -> tuple[Any, Any, Any]:
    """Convert pairwise short-circuit impedances to star equivalent impedances.

    All impedances must be given on a common base.

    Parameters
    ----------
    z01, z12, z02 : pandas.Series or complex
        Short-circuit impedance between windings 0-1, 1-2 and 0-2.

    Returns
    -------
    tuple
        Star impedances (z0, z1, z2) of the legs attached to windings 0, 1 and 2.

    Examples
    --------
    >>> from pypsa.components._types.transformers3w import delta_to_star
    >>> [round(z, 3) for z in delta_to_star(0.1, 0.2, 0.15)]
    [0.025, 0.075, 0.125]

    """
    z0 = (z01 + z02 - z12) / 2
    z1 = (z01 + z12 - z02) / 2
    z2 = (z02 + z12 - z01) / 2
    return z0, z1, z2


def apply_transformer3w_types(n: Network) -> None:
    """Calculate 3-winding transformer parameters from standard types.

    Sets `r01`, `x01`, `r12`, `x12`, `r02`, `x02`, `g`, `b` (per unit on `s_nom0`),
    `s_nom0`, `s_nom1`, `s_nom2`, `tap_side`, `tap_ratio`, `phase_shift1` and
    `phase_shift2` of all 3-winding transformers with a non-empty `type`.
    """
    # Avoid circular import
    from pypsa.network.power_flow import _warn_type_param_override  # noqa: PLC0415

    c = n.c.transformers3w
    with_types_b = c.static.type != ""
    if not with_types_b.any():
        return

    types_used = c.static.loc[with_types_b, "type"].unique()
    missing_types = pd.Index(types_used).difference(n.c.transformer3w_types.names)
    if not missing_types.empty:
        msg = (
            f"The type(s) {', '.join(missing_types)} do(es) not exist in "
            f"n.transformer3w_types"
        )
        raise ValueError(msg)

    t = c.static.loc[with_types_b, ["type", "tap_position"]].join(
        n.c.transformer3w_types.static, on="type"
    )

    s_nom0 = t["s_nom0"]
    for pair, (i, j) in {"01": (0, 1), "12": (1, 2), "02": (0, 2)}.items():
        # short circuit voltages are referred to the smaller rating of the pair,
        # rescale them to the common base s_nom0
        s_ref = t[[f"s_nom{i}", f"s_nom{j}"]].min(axis=1)
        z = t[f"vsc{pair}"] / 100.0 * s_nom0 / s_ref
        r = t[f"vscr{pair}"] / 100.0 * s_nom0 / s_ref
        t[f"r{pair}"] = r
        t[f"x{pair}"] = np.sqrt(z**2 - r**2)

    # NB: g and b are per unit of s_nom0
    t["g"] = t["pfe"] / (1000.0 * s_nom0)
    t["b"] = -np.sqrt(((t["i0"] / 100.0) ** 2 - t["g"] ** 2).clip(lower=0))

    t["tap_ratio"] = 1.0 + (t["tap_position"] - t["tap_neutral"]) * (
        t["tap_step"] / 100.0
    )

    impedance_attrs = ["r01", "x01", "r12", "x12", "r02", "x02", "g", "b"]
    rating_attrs = ["s_nom0", "s_nom1", "s_nom2"]
    attrs = [
        *impedance_attrs,
        *rating_attrs,
        "tap_side",
        "tap_ratio",
        "phase_shift1",
        "phase_shift2",
    ]
    _warn_type_param_override(
        "Transformer3W(s)",
        c.static.loc[with_types_b],
        t,
        impedance_attrs + rating_attrs,
    )
    c.static.loc[with_types_b, attrs] = t[attrs].astype(c.static[attrs].dtypes)


@patch_add_docstring
class Transformers3W(Components):
    """3-winding transformers components class.

    This class is used for 3-winding transformer components. All functionality
    specific to 3-winding transformers is implemented here. Functionality for all
    components is implemented in the abstract base class.

    A 3-winding transformer is modelled by its star equivalent: an auxiliary star bus
    with the voltage level of `bus0`, connected to `bus0`, `bus1` and `bus2` by three
    2-winding [pypsa.components.Transformers][] (the legs). The legs are created by
    [`expand`][pypsa.components.Transformers3W.expand], after which power flow and
    optimisation treat them like any other transformer. Results of the legs are
    mapped back to `p0`, `p1`, `p2` (and `q0`, `q1`, `q2`) after `n.pf()`, `n.lpf()`
    and `n.optimize()`.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.transformers3w
    Empty 'Transformer3W' Components

    """

    _list_name = "transformers3w"

    def star_equivalent(self) -> pd.DataFrame:
        """Calculate the parameters of the three 2-winding legs of the star equivalent.

        Standard types are applied first. Impedances of each leg are returned per
        unit on the leg's own rating, i.e. in the convention of
        [pypsa.components.Transformers][].

        Returns
        -------
        pandas.DataFrame
            Leg parameters indexed by leg name, with the electrical parameters `bus0`,
            `bus1`, `model`, `s_nom`, `r`, `x`, `g`, `b`, `tap_ratio`, `tap_side` and
            `phase_shift`, as well as `active` and the expansion parameters. For an
            extendable 3-winding transformer all legs are extendable with bounds
            scaled by the ratio of their rating to `s_nom0`, while the costs and the
            modular unit size are put on leg 0.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["HV", "MV", "LV"], v_nom=[110.0, 20.0, 10.0])
        >>> n.add("Transformer3W", "T", bus0="HV", bus1="MV", bus2="LV",
        ...       type="63/25/38 MVA 110/20/10 kV")
        >>> n.c.transformers3w.star_equivalent()[["bus0", "bus1", "s_nom"]]
                      bus0      bus1  s_nom
        name
        T leg0          HV    T star   63.0
        T leg1      T star        MV   25.0
        T leg2      T star        LV   38.0

        """
        n = self.n_save
        if n.has_scenarios:
            msg = (
                "3-winding transformers are not yet supported for stochastic networks."
            )
            raise NotImplementedError(msg)

        apply_transformer3w_types(n)
        static = self.static
        if static.empty:
            return pd.DataFrame(columns=LEG_COLUMNS)

        s_nom = static[["s_nom0", "s_nom1", "s_nom2"]]
        if (s_nom <= 0).any(axis=None):
            invalid = static.index[(s_nom <= 0).any(axis=1)]
            msg = (
                "3-winding transformers need positive ratings `s_nom0`, `s_nom1` and "
                f"`s_nom2`. Invalid: {', '.join(invalid.astype(str))}."
            )
            raise ValueError(msg)

        invalid_side = ~static.tap_side.isin([0, 1, 2])
        if invalid_side.any():
            msg = (
                "`tap_side` of 3-winding transformers must be 0, 1 or 2. Invalid: "
                f"{', '.join(static.index[invalid_side].astype(str))}."
            )
            raise ValueError(msg)

        z01 = static.r01 + 1j * static.x01
        z12 = static.r12 + 1j * static.x12
        z02 = static.r02 + 1j * static.x02
        z_star = delta_to_star(z01, z12, z02)  # per unit on s_nom0

        star = star_bus_names(static.index)
        names = leg_names(static.index)

        legs = []
        for i in range(3):
            # rescale from base s_nom0 to the leg's own rating s_nom{i}
            z = z_star[i] * static[f"s_nom{i}"] / static.s_nom0
            if i == 0:
                bus0, bus1 = static.bus0.to_numpy(), star.to_numpy()
                tap_side = 0
                g, b = static.g, static.b
                phase_shift = pd.Series(0.0, index=static.index)
            else:
                bus0, bus1 = star.to_numpy(), static[f"bus{i}"].to_numpy()
                tap_side = 1
                g = b = pd.Series(0.0, index=static.index)
                phase_shift = static[f"phase_shift{i}"]
            tap_ratio = static.tap_ratio.where(static.tap_side == i, 1.0)
            # expansion: bounds scale with the winding rating, costs sit on leg 0
            ratio = static[f"s_nom{i}"] / static.s_nom0
            if i == 0:
                costs = static[
                    [
                        "s_nom_set",
                        "s_nom_mod",
                        "capital_cost",
                        "overnight_cost",
                        "discount_rate",
                        "fom_cost",
                    ]
                ]
            else:
                costs = pd.DataFrame(
                    {
                        "s_nom_set": np.nan,
                        "s_nom_mod": 0.0,
                        "capital_cost": 0.0,
                        "overnight_cost": np.nan,
                        "discount_rate": np.nan,
                        "fom_cost": 0.0,
                    },
                    index=static.index,
                )
            legs.append(
                pd.DataFrame(
                    {
                        "bus0": bus0,
                        "bus1": bus1,
                        "model": static.model.to_numpy(),
                        "s_nom": static[f"s_nom{i}"].to_numpy(),
                        "r": np.real(z.to_numpy()),
                        "x": np.imag(z.to_numpy()),
                        "g": g.to_numpy(),
                        "b": b.to_numpy(),
                        "tap_ratio": tap_ratio.to_numpy(),
                        "tap_side": tap_side,
                        "phase_shift": phase_shift.to_numpy(),
                        "active": static.active.to_numpy(),
                        "s_nom_extendable": static.s_nom_extendable.to_numpy(),
                        "s_nom_min": (static.s_nom_min * ratio).to_numpy(),
                        "s_nom_max": (static.s_nom_max * ratio).to_numpy(),
                        **{col: costs[col].to_numpy() for col in costs},
                        "build_year": static.build_year.to_numpy(),
                        "lifetime": static.lifetime.to_numpy(),
                    },
                    index=names[i],
                )
            )

        # order legs per 3-winding transformer
        return pd.concat(legs)[LEG_COLUMNS].loc[
            pd.Index(np.column_stack(names).ravel(), name="name")
        ]

    def expand(self) -> pd.Index:
        """Add the star equivalent of all 3-winding transformers to the network.

        For each 3-winding transformer `<name>`, an auxiliary bus `<name> star` and
        the three 2-winding transformers `<name> leg0` (`bus0` to star bus),
        `<name> leg1` (star bus to `bus1`) and `<name> leg2` (star bus to `bus2`) are
        added. Existing components with these names are overwritten, so `expand` can
        be called again after changing the 3-winding transformer data.

        The star bus takes `v_nom`, `carrier` and the coordinates of `bus0`. The
        magnetising shunt `g`, `b` is placed on leg 0, while the tap changer is
        placed on the leg given by `tap_side`, at the side of the outer bus.

        Returns
        -------
        pandas.Index
            Names of the created 2-winding leg transformers.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["HV", "MV", "LV"], v_nom=[110.0, 20.0, 10.0])
        >>> n.add("Transformer3W", "T", bus0="HV", bus1="MV", bus2="LV",
        ...       type="63/25/38 MVA 110/20/10 kV")
        >>> list(n.c.transformers3w.expand())
        ['T leg0', 'T leg1', 'T leg2']
        >>> n.c.transformers3w.static.loc["T", "star_bus"]
        'T star'

        """
        n = self.n_save
        legs = self.star_equivalent()
        if legs.empty:
            return pd.Index([], name="name")

        static = self.static
        unknown = pd.Index(
            pd.unique(static[["bus0", "bus1", "bus2"]].to_numpy().ravel())
        ).difference(n.c.buses.static.index)
        if not unknown.empty:
            msg = (
                "3-winding transformers are attached to buses which are not defined: "
                f"{', '.join(unknown.astype(str))}."
            )
            raise ValueError(msg)

        star = star_bus_names(static.index)
        bus0 = n.c.buses.static.loc[static.bus0]
        n.add(
            "Bus",
            star,
            v_nom=bus0.v_nom.to_numpy(),
            carrier=bus0.carrier.to_numpy(),
            x=bus0.x.to_numpy(),
            y=bus0.y.to_numpy(),
            overwrite=True,
        )
        n.add("Transformer", legs.index, overwrite=True, **legs)
        static["star_bus"] = star.to_numpy()
        return legs.index

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


def _expanded_transformers3w(n: Network) -> pd.Index:
    """3-winding transformers whose star equivalent is present in the network."""
    c = n.c.transformers3w
    if c.static.empty or n.has_scenarios:
        return pd.Index([], name="name")
    expanded = c.static.index[c.static.star_bus != ""]
    present = np.all(
        [names.isin(n.c.transformers.static.index) for names in leg_names(expanded)],
        axis=0,
    )
    return expanded[present] if len(expanded) else expanded


def _warn_unexpanded_transformers3w(n: Network) -> None:
    """Warn about 3-winding transformers which are ignored since not expanded."""
    c = n.c.transformers3w
    if c.static.empty or n.has_scenarios:
        return
    missing = c.static.index.difference(_expanded_transformers3w(n))
    if not missing.empty:
        logger.warning(
            "The 3-winding transformer(s) %s are not represented in the network and "
            "are ignored in power flow and optimisation. Call "
            "`n.c.transformers3w.expand()` to add their star equivalent.",
            ", ".join(missing.astype(str)),
        )


def _assign_transformer3w_results(n: Network) -> None:
    """Map power flows of the star equivalent legs to the 3-winding transformers."""
    expanded = _expanded_transformers3w(n)
    if expanded.empty:
        return

    c = n.c.transformers3w
    trafo_dynamic = n.c.transformers.dynamic
    names = leg_names(expanded)
    # (3-winding attribute, leg, leg attribute)
    mapping = [("0", 0, "0"), ("1", 1, "1"), ("2", 2, "1")]
    for quantity in ["p", "q"]:
        for port, leg, leg_port in mapping:
            leg_attr = f"{quantity}{leg_port}"
            if leg_attr not in trafo_dynamic or trafo_dynamic[leg_attr].empty:
                continue
            leg_flows = trafo_dynamic[leg_attr]
            if not names[leg].isin(leg_flows.columns).all():
                continue
            values = leg_flows.loc[:, names[leg]].set_axis(expanded, axis=1)
            values.columns.name = "name"
            c.dynamic[f"{quantity}{port}"] = values


def _assign_transformer3w_capacities(n: Network) -> None:
    """Map optimised ratings of the star equivalent legs to the 3-winding transformers."""
    expanded = _expanded_transformers3w(n)
    if expanded.empty:
        return

    c = n.c.transformers3w
    legs = n.c.transformers.static
    for i, names in enumerate(leg_names(expanded)):
        c.static.loc[expanded, f"s_nom{i}_opt"] = legs.loc[
            names, "s_nom_opt"
        ].to_numpy()


def _fix_optimal_transformer3w_capacities(n: Network) -> None:
    """Fix ratings of extendable 3-winding transformers to their optimised values."""
    c = n.c.transformers3w
    if c.static.empty:
        return
    ext_i = c.static.index[c.static.s_nom_extendable].intersection(
        _expanded_transformers3w(n)
    )
    for i in range(3):
        c.static.loc[ext_i, f"s_nom{i}"] = c.static.loc[ext_i, f"s_nom{i}_opt"]
    c.static["s_nom_extendable"] = False
