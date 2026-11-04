# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Optimisation of grid control devices.

Covers the costs of phase-shifting transformers and the switching and installation
decisions of circuit breakers (optimal transmission switching).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from xarray import DataArray

from pypsa.components._types.circuit_breakers import BREAKER_BRANCH_COMPONENTS

if TYPE_CHECKING:
    from linopy import LinearExpression

    from pypsa import Network

PHASE_SHIFT = "Transformer-phase_shift"
PHASE_SHIFT_ABS = "Transformer-phase_shift_abs"
BREAKER_STATUS = "CircuitBreaker-status_opt"
BREAKER_INSTALLED = "CircuitBreaker-installed"
BREAKER_SLACK = "CircuitBreaker-v_ang_open"
BREAKER_SWITCH = "CircuitBreaker-switch"


# Phase-shifting transformers
# ---------------------------


def _costly_phase_shifts(n: Network, sns: pd.Index) -> pd.Index:
    """Optimised phase shifts with a non-zero `phase_shift_cost`."""
    m = n.model
    if PHASE_SHIFT not in m.variables:
        return pd.Index([], name="name")
    names = m[PHASE_SHIFT].indexes["name"]
    cost = n.c.transformers.da.phase_shift_cost.sel(name=names, snapshot=sns)
    if (cost < 0).any():
        msg = "The `phase_shift_cost` of transformers must not be negative."
        raise ValueError(msg)
    costly = (cost != 0).any([d for d in cost.dims if d != "name"]).to_series()
    return names[costly.reindex(names).to_numpy(dtype=bool)]


def define_phase_shift_cost_variables(n: Network, sns: pd.Index) -> None:
    """Define the absolute phase shift of optimised phase-shifting transformers.

    The variable is only created for transformers with a non-zero
    `phase_shift_cost`; it is bounded below by the positive and negative phase
    shift, so that minimising its cost yields the absolute phase shift.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    sns : pandas.Index
        Snapshots.

    """
    names = _costly_phase_shifts(n, sns)
    if names.empty:
        return
    m = n.model
    ps = m[PHASE_SHIFT].sel(name=names)
    abs_ps = m.add_variables(lower=0, coords=ps.coords, name=PHASE_SHIFT_ABS)
    m.add_constraints(abs_ps - ps >= 0, name=f"{PHASE_SHIFT_ABS}-upper")
    m.add_constraints(abs_ps + ps >= 0, name=f"{PHASE_SHIFT_ABS}-lower")


def phase_shift_cost_terms(n: Network, sns: pd.Index, weight: Any) -> list:
    """Objective terms for the cost of shifting the phase of transformers."""
    m = n.model
    if PHASE_SHIFT_ABS not in m.variables:
        return []
    abs_ps = m[PHASE_SHIFT_ABS]
    names = abs_ps.indexes["name"]
    cost = n.c.transformers.da.phase_shift_cost.sel(name=names, snapshot=sns)
    return [(abs_ps * cost * weight).sum(dim=["name", "snapshot"])]


# Circuit breakers
# ----------------


def active_circuit_breakers(n: Network) -> pd.DataFrame:
    """Validate and return the active circuit breakers of the network.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.

    Returns
    -------
    pandas.DataFrame
        Static data of the active circuit breakers.

    """
    c = n.c.circuit_breakers
    if c.static.empty:
        return c.static
    if n.has_scenarios:
        msg = "Circuit breakers are not yet supported for stochastic networks."
        raise NotImplementedError(msg)

    static = c.static[c.static.active]

    invalid = ~static.component.isin(BREAKER_BRANCH_COMPONENTS)
    if invalid.any():
        msg = (
            "The `component` of circuit breakers must be one of "
            f"{', '.join(BREAKER_BRANCH_COMPONENTS)}. Invalid: "
            f"{', '.join(static.index[invalid].astype(str))}."
        )
        raise ValueError(msg)

    for comp_key, group in static.groupby("component"):
        comp = str(comp_key)
        missing = pd.Index(group.branch).difference(n.c[comp].static.index)
        if not missing.empty:
            msg = (
                f"Circuit breakers are attached to {comp} components which do not "
                f"exist: {', '.join(missing.astype(str))}."
            )
            raise ValueError(msg)

    duplicated = static.duplicated(["component", "branch"], keep=False)
    if duplicated.any():
        msg = (
            "At most one active circuit breaker can be attached to a branch. "
            f"Duplicated: {', '.join(static.index[duplicated].astype(str))}."
        )
        raise ValueError(msg)

    not_switchable = static.extendable & ~static.switchable
    if not_switchable.any():
        msg = (
            "Extendable circuit breakers must be switchable. Invalid: "
            f"{', '.join(static.index[not_switchable].astype(str))}."
        )
        raise ValueError(msg)

    return static


def _optimised_breakers(n: Network, sns: pd.Index) -> pd.DataFrame:
    """Active breakers on active branches which can open in the snapshots."""
    static = active_circuit_breakers(n)
    if static.empty:
        return static

    on_active_branch = pd.Series(False, index=static.index)
    for comp_key, group in static.groupby("component"):
        comp = str(comp_key)
        active = n.c[comp].active_assets
        on_active_branch[group.index] = group.branch.isin(active)
    static = static[on_active_branch]

    status = n.get_switchable_as_dense(
        "CircuitBreaker", "status", inds=static.index
    ).loc[sns]
    can_open = static.switchable | (status != 1).any()
    return static[can_open]


def _breaker_status(n: Network, names: pd.Index, sns: pd.Index) -> DataArray:
    return n.c.circuit_breakers.da.status.sel(name=names, snapshot=sns)


def _branch_flow(n: Network, breakers: pd.DataFrame, sns: pd.Index) -> dict:
    """Flow variables of the branches of the breakers, indexed by breaker name."""
    flows = {}
    for comp_key, group in breakers.groupby("component"):
        comp = str(comp_key)
        s = n.model[f"{comp}-s"].sel(name=group.branch.to_list(), snapshot=sns)
        flows[comp] = s.assign_coords(name=group.index.to_numpy())
    return flows


def _branch_flow_capacity(n: Network, breakers: pd.DataFrame) -> pd.Series:
    """Upper bound of the absolute flow of the branches of the breakers."""
    capacity = pd.Series(np.nan, index=breakers.index)
    for comp_key, group in breakers.groupby("component"):
        comp = str(comp_key)
        static = n.c[comp].static.loc[group.branch]
        s_max_pu = n.get_switchable_as_dense(
            comp, "s_max_pu", inds=pd.Index(group.branch)
        ).max()
        s_nom = static.s_nom.where(~static.s_nom_extendable, static.s_nom_max)
        capacity[group.index] = (s_nom * s_max_pu).to_numpy()
    infinite = ~np.isfinite(capacity)
    if infinite.any():
        msg = (
            "Circuit breakers on extendable branches need a finite `s_nom_max` of the "
            f"branch. Invalid: {', '.join(capacity.index[infinite].astype(str))}."
        )
        raise ValueError(msg)
    return capacity


def define_circuit_breaker_variables(n: Network, sns: pd.Index) -> None:
    """Define the variables of circuit breakers.

    * `CircuitBreaker-status_opt`: binary status (1 closed, 0 open) of switchable
      breakers.
    * `CircuitBreaker-installed`: binary installation decision of extendable
      breakers.
    * `CircuitBreaker-v_ang_open`: voltage angle difference across open breakers,
      which relaxes Kirchhoff's voltage law of the open branch.
    * `CircuitBreaker-switch`: switching operations of switchable breakers with
      switching costs.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    sns : pandas.Index
        Snapshots.

    """
    breakers = _optimised_breakers(n, sns)
    if breakers.empty:
        return
    m = n.model

    big_m = DataArray(np.deg2rad(breakers.v_ang_max_open).rename_axis("name"))
    status = _breaker_status(n, breakers.index, sns)

    # switchable: status is a decision
    switchable = breakers.index[breakers.switchable]
    if not switchable.empty:
        coords = status.sel(name=switchable).coords
        m.add_variables(coords=coords, name=BREAKER_STATUS, binary=True)

        extendable = breakers.index[breakers.extendable]
        if not extendable.empty:
            m.add_variables(
                coords=[extendable.rename("name")], name=BREAKER_INSTALLED, binary=True
            )

        cost = n.c.circuit_breakers.da.switching_cost.sel(name=switchable, snapshot=sns)
        if (cost < 0).any():
            msg = "The `switching_cost` of circuit breakers must not be negative."
            raise ValueError(msg)
        costly = (cost != 0).any("snapshot").to_series().reindex(switchable)
        if costly.any():
            names = switchable[costly.to_numpy()]
            m.add_variables(
                lower=0, coords=status.sel(name=names).coords, name=BREAKER_SWITCH
            )

    # angle difference across the open breaker; fixed breakers can only open when
    # their status is 0
    is_fixed = DataArray(~breakers.switchable.rename_axis("name"))
    bound = big_m * (1 - status.where(is_fixed, 0))
    m.add_variables(lower=-bound, upper=bound, name=BREAKER_SLACK)


def define_circuit_breaker_constraints(n: Network, sns: pd.Index) -> None:
    r"""Define the constraints of circuit breakers.

    For a breaker with status $u_t$ (fixed or binary) on a branch with flow $s_t$ and
    flow bound $\bar{s}$, the flow vanishes when the breaker is open and Kirchhoff's
    voltage law of the branch is relaxed by the angle difference $\delta_t$ across
    the open breaker:

    $$-\bar{s} u_t \leq s_t \leq \bar{s} u_t, \qquad
    -M (1 - u_t) \leq \delta_t \leq M (1 - u_t)$$

    An extendable breaker can only open if installed, $u_t \geq 1 - i$, and
    switching operations $w_t \geq |u_t - u_{t-1}|$ are priced with
    `switching_cost`.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    sns : pandas.Index
        Snapshots.

    """
    m = n.model
    breakers = _optimised_breakers(n, sns)
    if breakers.empty:
        return

    flows = _branch_flow(n, breakers, sns)
    capacity = _branch_flow_capacity(n, breakers)

    # fixed breakers: no flow while open
    fixed = breakers.index[~breakers.switchable]
    for comp, s in flows.items():
        names = s.indexes["name"].intersection(fixed)
        if names.empty:
            continue
        open_ = _breaker_status(n, names, sns) != 1
        m.add_constraints(
            s.sel(name=names) == 0,
            name=f"CircuitBreaker-{comp}-open",
            mask=open_,
        )

    switchable = breakers.index[breakers.switchable]
    if switchable.empty:
        return

    status = m[BREAKER_STATUS]
    cap = DataArray(capacity.rename_axis("name"))
    big_m = DataArray(np.deg2rad(breakers.v_ang_max_open).rename_axis("name"))

    for comp, s in flows.items():
        names = s.indexes["name"].intersection(switchable)
        if names.empty:
            continue
        flow = s.sel(name=names)
        u = status.sel(name=names)
        c_sel = cap.sel(name=names)
        m.add_constraints(
            flow - c_sel * u <= 0, name=f"CircuitBreaker-{comp}-flow-upper"
        )
        m.add_constraints(
            flow + c_sel * u >= 0, name=f"CircuitBreaker-{comp}-flow-lower"
        )

    slack = m[BREAKER_SLACK].sel(name=switchable)
    bm = big_m.sel(name=switchable)
    m.add_constraints(slack + bm * status <= bm, name=f"{BREAKER_SLACK}-upper")
    m.add_constraints(slack - bm * status >= -bm, name=f"{BREAKER_SLACK}-lower")

    if BREAKER_INSTALLED in m.variables:
        installed = m[BREAKER_INSTALLED]
        names = installed.indexes["name"]
        m.add_constraints(
            status.sel(name=names) + installed >= 1,
            name="CircuitBreaker-installed-status",
        )

    if BREAKER_SWITCH in m.variables:
        switch = m[BREAKER_SWITCH]
        names = switch.indexes["name"]
        u = status.sel(name=names)
        u_prev = u.shift(snapshot=1)
        initial = DataArray(
            breakers.loc[names, "status_initial"].astype(float).rename_axis("name")
        )
        first = DataArray(
            np.arange(len(u.indexes["snapshot"])) == 0,
            coords={"snapshot": u.coords["snapshot"]},
            dims="snapshot",
        )
        rhs_open = (-initial).where(first, 0)
        rhs_close = initial.where(first, 0)
        m.add_constraints(
            switch - u + u_prev >= rhs_open, name=f"{BREAKER_SWITCH}-open"
        )
        m.add_constraints(
            switch + u - u_prev >= rhs_close, name=f"{BREAKER_SWITCH}-close"
        )


def circuit_breaker_kvl_term(
    n: Network, c_plain: pd.DataFrame, snapshots: pd.Index
) -> LinearExpression | None:
    """Relaxation of the cycle constraints by the angle differences of open breakers.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    c_plain : pandas.DataFrame
        Unweighted cycle matrix with a (type, name) branch index.
    snapshots : pandas.Index
        Snapshots of the investment period.

    Returns
    -------
    linopy.LinearExpression or None
        Term to add to the left-hand side of Kirchhoff's voltage law (scaled by
        1e5 like the other terms), or None if there are no open-capable breakers.

    """
    m = n.model
    if BREAKER_SLACK not in m.variables or c_plain.empty:
        return None
    slack = m[BREAKER_SLACK]
    static = n.c.circuit_breakers.static.loc[slack.indexes["name"]]

    terms = []
    for comp_key, group in static.groupby("component"):
        comp = str(comp_key)
        if comp not in c_plain.index.unique("type"):
            continue
        rows = c_plain.loc[comp]
        group = group[group.branch.isin(rows.index)]
        if group.empty:
            continue
        C = DataArray(rows.loc[group.branch])
        eps = slack.sel(name=group.index.to_list(), snapshot=snapshots).assign_coords(
            name=group.branch.to_numpy()
        )
        terms.append((eps @ C) * 1e5)
    return sum(terms) if terms else None


def circuit_breaker_cost_terms(
    n: Network, sns: pd.Index, event_weight: Any, period_weight: float
) -> tuple[list, list]:
    """Objective terms for installing and switching circuit breakers.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    sns : pandas.Index
        Snapshots.
    event_weight : float or xarray.DataArray
        Weight of switching events per snapshot (investment period weighting).
    period_weight : float
        Weight of periodic capital costs (sum of investment period weightings).

    Returns
    -------
    tuple[list, list]
        CAPEX and OPEX terms.

    """
    m = n.model
    capex, opex = [], []
    c = n.c.circuit_breakers
    if BREAKER_INSTALLED in m.variables:
        installed = m[BREAKER_INSTALLED]
        names = installed.indexes["name"]
        cost = DataArray(c.static.loc[names, "capital_cost"].rename_axis("name"))
        if (cost != 0).any():
            capex.append((installed * cost * period_weight).sum("name"))
    if BREAKER_SWITCH in m.variables:
        switch = m[BREAKER_SWITCH]
        names = switch.indexes["name"]
        cost = c.da.switching_cost.sel(name=names, snapshot=sns)
        opex.append((switch * cost * event_weight).sum(dim=["name", "snapshot"]))
    return capex, opex


def assign_circuit_breaker_results(n: Network, sns: pd.Index) -> None:
    """Write status and installation results of circuit breakers to the network."""
    c = n.c.circuit_breakers
    if c.static.empty or n.has_scenarios:
        return
    m = n.model

    status = n.get_switchable_as_dense("CircuitBreaker", "status").loc[sns]
    status = status.astype(float)
    if BREAKER_STATUS in m.variables:
        sol = m[BREAKER_STATUS].solution.to_pandas().round()
        status.loc[sol.index, sol.columns] = sol.to_numpy()
    current = c.dynamic.get("status_opt", pd.DataFrame())
    c.dynamic["status_opt"] = (
        status.combine_first(current) if not current.empty else status
    ).reindex(index=n.snapshots, columns=c.static.index)

    installed = pd.Series(1.0, index=c.static.index)
    if BREAKER_INSTALLED in m.variables:
        sol = m[BREAKER_INSTALLED].solution.to_pandas().round()
        installed[sol.index] = sol.to_numpy()
    c.static["installed_opt"] = installed
