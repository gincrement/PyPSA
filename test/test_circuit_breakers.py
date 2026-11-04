# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for circuit breakers and optimal transmission switching.

Physical setup: a triangle A-B-C with cheap supply at A and demand of 100 MW at C.
All lines have the same reactance, so with all lines closed 2/3 of the transfer
takes the direct line A-C and 1/3 the path A-B-C. Line B-C is limited to 30 MW,
which limits the transfer to 90 MW. Opening B-C sends everything over A-C, which
carries the full 100 MW (Braess-like paradox).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import pypsa

BENEFIT = 10.0 * (100.0 - 10.0)  # 10 MW shifted from expensive to cheap supply


def _triangle(snapshots: int = 1, bc_component: str = "Line") -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(snapshots))
    n.add("Bus", ["A", "B", "C"], v_nom=380.0)
    n.add("Generator", "cheap", bus="A", p_nom=500, marginal_cost=10, control="Slack")
    n.add("Generator", "dear", bus="C", p_nom=500, marginal_cost=100)
    n.add("Load", "load", bus="C", p_set=100.0)
    n.add("Line", "AC", bus0="A", bus1="C", x=1.0, r=0.01, s_nom=100.0)
    n.add("Line", "AB", bus0="A", bus1="B", x=1.0, r=0.01, s_nom=100.0)
    if bc_component == "Line":
        n.add("Line", "BC", bus0="B", bus1="C", x=1.0, r=0.01, s_nom=30.0)
    else:
        # same per unit reactance as the lines: x_pu = 1 / 380**2
        n.add(
            "Transformer",
            "BC",
            bus0="B",
            bus1="C",
            x=30.0 / 380.0**2,
            r=1e-8,
            s_nom=30.0,
        )
    return n


def test_all_closed_baseline():
    n = _triangle()
    n.optimize()
    assert n.objective == pytest.approx(90 * 10 + 10 * 100)


@pytest.mark.parametrize("component", ["Line", "Transformer"])
def test_optimal_switching_opens_branch(component):
    n = _triangle(bc_component=component)
    n.add("CircuitBreaker", "cb", component=component, branch="BC", switchable=True)
    status, _ = n.optimize()
    assert status == "ok"

    assert n.objective == pytest.approx(100 * 10)
    assert n.c.circuit_breakers.dynamic.status_opt.iloc[0]["cb"] == 0
    assert n.c[component].dynamic.p0.iloc[0]["BC"] == pytest.approx(0, abs=1e-6)
    assert n.c.lines.dynamic.p0.iloc[0]["AC"] == pytest.approx(100.0)
    # the angle difference across the open breaker relaxes KVL
    assert n.c.circuit_breakers.dynamic.v_ang_open.iloc[0]["cb"] != pytest.approx(0)


def test_switching_keeps_useful_branch_closed():
    """A breaker on A-C would only make things worse."""
    n = _triangle()
    n.add("CircuitBreaker", "cb", branch="AC", switchable=True)
    n.optimize()
    assert n.c.circuit_breakers.dynamic.status_opt.iloc[0]["cb"] == 1
    assert n.objective == pytest.approx(90 * 10 + 10 * 100)


def test_fixed_open_breaker_in_optimisation_and_power_flow():
    n = _triangle(snapshots=3)
    n.add("CircuitBreaker", "cb", branch="BC", status=[1, 0, 1])
    n.optimize()

    p0 = n.c.lines.dynamic.p0
    assert_allclose(p0["BC"], [30.0, 0.0, 30.0], atol=1e-6)
    assert_allclose(p0["AC"], [60.0, 100.0, 60.0], atol=1e-6)
    assert_allclose(n.c.circuit_breakers.dynamic.status_opt["cb"], [1, 0, 1])
    assert "CircuitBreaker-status_opt" not in n.model.variables

    n.optimize.fix_optimal_dispatch()
    n.lpf()
    assert_allclose(n.c.lines.dynamic.p0["BC"], [30.0, 0.0, 30.0], atol=1e-6)
    assert_allclose(n.c.lines.dynamic.p0["AC"], [60.0, 100.0, 60.0], atol=1e-6)

    result = n.pf()
    assert result["converged"].all().all()
    assert_allclose(n.c.lines.dynamic.p0.loc[1, "BC"], 0.0)
    assert n.c.lines.dynamic.p0.loc[1, "AC"] == pytest.approx(100.0, rel=1e-3)

    # the base topology and active flags are restored
    assert n.c.lines.static.active.all()
    assert len(n.c.sub_networks.static) == 1


def test_open_breaker_splits_sub_network_in_power_flow():
    n = pypsa.Network()
    n.add("Bus", ["A", "B"])
    n.add("Generator", "gA", bus="A", control="Slack", p_set=0)
    n.add("Generator", "gB", bus="B", p_set=10)
    n.add("Load", "load", bus="B", p_set=10)
    n.add("Line", "L", bus0="A", bus1="B", x=0.1, r=0.01)
    n.add("CircuitBreaker", "cb", branch="L", status=0)
    n.lpf()
    assert n.c.lines.dynamic.p0.iloc[0]["L"] == 0.0


def test_optimal_switching_then_power_flow():
    n = _triangle()
    n.add("CircuitBreaker", "cb", branch="BC", switchable=True)
    n.optimize()
    n.optimize.fix_optimal_dispatch()

    breakers = n.c.circuit_breakers
    assert not breakers.static.at["cb", "switchable"]
    assert breakers.dynamic.status.iloc[0]["cb"] == 0
    n.lpf()
    assert n.c.lines.dynamic.p0.iloc[0]["AC"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("capital_cost", "installed"), [(0.5 * BENEFIT, 1), (2 * BENEFIT, 0)]
)
def test_breaker_expansion(capital_cost, installed):
    n = _triangle()
    n.add(
        "CircuitBreaker",
        "cb",
        branch="BC",
        switchable=True,
        extendable=True,
        capital_cost=capital_cost,
    )
    n.optimize()
    breakers = n.c.circuit_breakers
    assert breakers.static.at["cb", "installed_opt"] == installed
    assert breakers.dynamic.status_opt.iloc[0]["cb"] == 1 - installed
    expected = 1000.0 + capital_cost if installed else 1900.0
    assert n.objective == pytest.approx(expected)


@pytest.mark.parametrize(
    ("switching_cost", "switches"), [(0.5 * BENEFIT, 1.0), (3 * BENEFIT, 0.0)]
)
def test_switching_cost(switching_cost, switches):
    n = _triangle(snapshots=2)
    n.add(
        "CircuitBreaker",
        "cb",
        branch="BC",
        switchable=True,
        switching_cost=switching_cost,
        status_initial=1,
    )
    n.optimize()
    breakers = n.c.circuit_breakers
    assert breakers.dynamic.switch["cb"].sum() == pytest.approx(switches)
    assert_allclose(breakers.dynamic.status_opt["cb"], [1 - switches] * 2)


def test_switching_with_investment_periods():
    n = _triangle()
    n.set_snapshots(pd.MultiIndex.from_product([[2030, 2040], [0]]))
    n.investment_periods = [2030, 2040]
    n.c.loads.dynamic.p_set = pd.DataFrame({"load": [80.0, 100.0]}, index=n.snapshots)
    n.add("CircuitBreaker", "cb", branch="BC", switchable=True)
    n.optimize(multi_investment_periods=True)
    # only the congested period benefits from opening
    status = n.c.circuit_breakers.dynamic.status_opt["cb"]
    assert status.loc[(2040, 0)] == 0
    assert n.c.lines.dynamic.p0.loc[(2040, 0), "AC"] == pytest.approx(100.0)


def test_meshed_network_with_parallel_cycles():
    """An open branch in one cycle keeps Kirchhoff's voltage law in the others."""
    n = _triangle()
    n.add("Bus", "D", v_nom=380.0)
    n.add("Line", "AD", bus0="A", bus1="D", x=1.0, r=0.01, s_nom=500.0)
    n.add("Line", "DC", bus0="D", bus1="C", x=1.0, r=0.01, s_nom=500.0)
    n.add("CircuitBreaker", "cb", branch="BC", status=0)
    n.optimize()
    p0 = n.c.lines.dynamic.p0.iloc[0]
    # with B-C open, A-C (x=1) and A-D-C (x=2) split 2:1 and B carries nothing
    assert p0["BC"] == pytest.approx(0.0, abs=1e-6)
    assert p0["AB"] == pytest.approx(0.0, abs=1e-6)
    assert p0["AC"] == pytest.approx(2 * p0["AD"])
    assert p0["AC"] + p0["AD"] == pytest.approx(100.0)


# Validation
# ----------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"branch": "missing"}, "do not exist"),
        ({"branch": "BC", "component": "Link"}, "must be one of"),
        ({"branch": "BC", "extendable": True}, "must be switchable"),
    ],
)
def test_invalid_breakers_raise(kwargs, match):
    n = _triangle()
    n.add("CircuitBreaker", "cb", **kwargs)
    with pytest.raises(ValueError, match=match):
        n.optimize()


def test_duplicate_breakers_raise():
    n = _triangle()
    n.add("CircuitBreaker", ["cb1", "cb2"], branch="BC", status=0)
    with pytest.raises(ValueError, match="At most one"):
        n.lpf()


def test_extendable_branch_needs_finite_s_nom_max():
    n = _triangle()
    n.c.lines.static.loc["BC", "s_nom_extendable"] = True
    n.add("CircuitBreaker", "cb", branch="BC", switchable=True)
    with pytest.raises(ValueError, match="finite `s_nom_max`"):
        n.optimize()

    n.c.lines.static.loc["BC", "s_nom_max"] = 50.0
    status, _ = n.optimize()
    assert status == "ok"


def test_negative_switching_cost_raises():
    n = _triangle()
    n.add("CircuitBreaker", "cb", branch="BC", switchable=True, switching_cost=-1)
    with pytest.raises(ValueError, match="switching_cost"):
        n.optimize()


def test_inactive_breaker_is_ignored():
    n = _triangle()
    n.add("CircuitBreaker", "cb", branch="BC", status=0, active=False)
    n.optimize()
    assert "CircuitBreaker-v_ang_open" not in n.model.variables
    assert n.objective == pytest.approx(1900.0)


def test_stochastic_not_supported():
    n = _triangle()
    n.add("CircuitBreaker", "cb", branch="BC", switchable=True)
    n.set_scenarios({"a": 0.5, "b": 0.5})
    with pytest.raises(NotImplementedError, match="stochastic"):
        n.optimize()


def test_netcdf_roundtrip(tmp_path):
    n = _triangle(snapshots=2)
    n.add("CircuitBreaker", "cb", branch="BC", status=[1, 0], switchable=False)
    fn = tmp_path / "breakers.nc"
    n.export_to_netcdf(fn)
    m = pypsa.Network(fn)
    assert_allclose(m.c.circuit_breakers.dynamic.status["cb"], [1, 0])
    assert m.c.circuit_breakers.static.at["cb", "branch"] == "BC"
    assert np.isclose(m.c.circuit_breakers.static.at["cb", "v_ang_max_open"], 180)
