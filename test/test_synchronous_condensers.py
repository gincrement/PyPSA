# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for synchronous condensers and system strength requirements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import pypsa


def _network(**syncon) -> pypsa.Network:
    """Two buses with demand at B, cheap supply at A."""
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Carrier", ["AC", "gas", "syncon"])
    n.add("Bus", ["A", "B"], v_nom=110.0, carrier="AC")
    n.add("Generator", "gas", bus="A", carrier="gas", p_nom=500, marginal_cost=10)
    n.add("Load", "load", bus="B", p_set=[100.0, 60.0])
    n.add("Line", "L", bus0="A", bus1="B", x=1.0, r=0.1, s_nom=500, carrier="AC")
    params = {
        "s_nom_extendable": True,
        "capital_cost": 5.0,
        "loss": 0.01,
        "inertia_constant": 2.0,
        "short_circuit_factor": 6.0,
        "carrier": "syncon",
        **syncon,
    }
    n.add("SynchronousCondenser", "sc", bus="B", **params)
    return n


# Optimisation
# ------------


def test_no_requirement_no_expansion():
    n = _network()
    n.optimize()
    assert n.c.synchronous_condensers.static.at["sc", "s_nom_opt"] == pytest.approx(0)


def test_inertia_requirement_drives_expansion_and_dispatch():
    n = _network()
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    status, _ = n.optimize()
    assert status == "ok"

    sc = n.c.synchronous_condensers
    # 100 MWs / 2 s
    assert sc.static.at["sc", "s_nom_opt"] == pytest.approx(50.0)
    assert_allclose(sc.dynamic.s["sc"], 50.0)
    # losses are drawn from the bus and supplied by the generator
    assert_allclose(sc.dynamic.p["sc"], -0.5)
    assert_allclose(n.c.generators.dynamic.p["gas"], [100.5, 60.5])
    assert_allclose(n.c.buses.dynamic.p["B"], [-100.5, -60.5])

    # shadow price of the requirement per snapshot
    mu = n.c.global_constraints.dynamic.mu_snapshot["H"]
    assert (mu > 0).all()


def test_existing_capacity_dispatched_only_as_needed():
    n = _network(s_nom_extendable=False, s_nom=200.0, marginal_cost=1.0)
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()
    assert_allclose(n.c.synchronous_condensers.dynamic.s["sc"], 50.0)


def test_short_circuit_requirement_at_bus():
    n = _network()
    n.add(
        "SynchronousCondenser",
        "sc A",
        bus="A",
        s_nom_extendable=True,
        capital_cost=1.0,
        short_circuit_factor=6.0,
    )
    n.add(
        "GlobalConstraint",
        "SCL B",
        type="short_circuit_power",
        bus="B",
        sense=">=",
        constant=300.0,
    )
    n.optimize()

    s_nom = n.c.synchronous_condensers.static.s_nom_opt
    # the cheaper condenser at A does not count for bus B
    assert s_nom["sc"] == pytest.approx(50.0)
    assert s_nom["sc A"] == pytest.approx(0.0)


def test_generators_contribute_via_carrier_attribute():
    n = _network()
    n.c.carriers.static["inertia_constant"] = 0.0
    n.c.carriers.static.loc["gas", "inertia_constant"] = 0.5
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()

    # gas provides 0.5 * p, the condenser covers the rest of 100 MWs at the peak
    p_gas = n.c.generators.dynamic.p["gas"]
    s = n.c.synchronous_condensers.dynamic.s["sc"]
    assert ((0.5 * p_gas + 2.0 * s) >= 100.0 - 1e-6).all()
    assert n.c.synchronous_condensers.static.at["sc", "s_nom_opt"] < 50.0


def test_requirement_per_investment_period():
    n = _network()
    n.set_snapshots(pd.MultiIndex.from_product([[2030, 2040], [0, 1]]))
    n.investment_periods = [2030, 2040]
    n.c.loads.dynamic.p_set = pd.DataFrame({"load": 80.0}, index=n.snapshots)
    n.add(
        "GlobalConstraint",
        "H 2040",
        type="inertia",
        sense=">=",
        constant=100.0,
        investment_period=2040,
    )
    n.optimize(multi_investment_periods=True)
    s = n.c.synchronous_condensers.dynamic.s["sc"]
    assert_allclose(s.loc[2040], 50.0)
    assert_allclose(s.loc[2030], 0.0)


def test_requirement_without_contributors_is_skipped(caplog):
    n = _network(inertia_constant=0.0)
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()
    assert "has no contributing components" in caplog.text
    assert "GlobalConstraint-H" not in n.model.constraints


def test_modular_expansion():
    n = _network(s_nom_mod=40.0)
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()
    assert n.c.synchronous_condensers.static.at["sc", "s_nom_opt"] == pytest.approx(80)


def test_statistics():
    n = _network()
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()
    capex = n.statistics.capex()
    assert capex.loc["SynchronousCondenser"].sum() == pytest.approx(250.0)
    balance = n.statistics.energy_balance()
    assert balance.loc["SynchronousCondenser"].sum() == pytest.approx(-1.0)


def test_stochastic_requirement_not_supported():
    n = _network()
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.set_scenarios({"a": 0.5, "b": 0.5})
    with pytest.raises(NotImplementedError, match="stochastic"):
        n.optimize()


# Power flow
# ----------


def _pf_network(**syncon) -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", ["A", "B"], v_nom=110.0, v_mag_pu_set=1.0)
    n.add("Generator", "grid", bus="A", control="Slack")
    n.add("Line", "L", bus0="A", bus1="B", x=10.0, r=1.0)
    n.add("Load", "load", bus="B", p_set=20.0, q_set=10.0)
    n.add("SynchronousCondenser", "sc", bus="B", s_nom=30.0, **syncon)
    return n


def test_pf_voltage_control():
    n = _pf_network(p_set=-0.3)
    n.pf()
    sn = n.snapshots[0]
    assert n.c.buses.dynamic.v_mag_pu.at[sn, "B"] == pytest.approx(1.0)
    q = n.c.synchronous_condensers.dynamic.q.at[sn, "sc"]
    assert q > 10.0
    assert n.c.synchronous_condensers.dynamic.p.at[sn, "sc"] == pytest.approx(-0.3)
    # reactive power balance at bus B
    q_line = n.c.lines.dynamic.q1.at[sn, "L"]
    assert q - 10.0 - q_line == pytest.approx(0.0, abs=1e-6)


def test_pf_pq_control():
    n = _pf_network(control="PQ", q_set=5.0)
    n.pf()
    sn = n.snapshots[0]
    assert n.c.synchronous_condensers.dynamic.q.at[sn, "sc"] == pytest.approx(5.0)
    assert n.c.buses.dynamic.v_mag_pu.at[sn, "B"] < 1.0


def test_pf_shares_reactive_power_with_generator():
    n = _pf_network()
    n.add("Generator", "gen B", bus="B", control="PV", p_nom=10.0, p_set=5.0)
    n.pf()
    sn = n.snapshots[0]
    q_gen = n.c.generators.dynamic.q.at[sn, "gen B"]
    q_sc = n.c.synchronous_condensers.dynamic.q.at[sn, "sc"]
    # shared in the ratio of the ratings 10:30
    assert q_sc == pytest.approx(3 * q_gen)
    assert n.c.buses.dynamic.v_mag_pu.at[sn, "B"] == pytest.approx(1.0)


def test_fix_optimal_dispatch_feeds_power_flow():
    n = _network()
    n.add("Generator", "slack", bus="A", control="Slack")
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()
    n.optimize.fix_optimal_dispatch()
    assert_allclose(n.c.synchronous_condensers.dynamic.p_set["sc"], -0.5)

    n.lpf()
    assert_allclose(n.c.synchronous_condensers.dynamic.p["sc"], -0.5)
    assert_allclose(n.c.lines.dynamic.p1["L"], [-100.5, -60.5])


# Network handling
# ----------------


def test_clustering_keeps_synchronous_condensers():
    n = _network(s_nom_extendable=False, s_nom=30.0)
    busmap = pd.Series({"A": "X", "B": "X"})
    clustered = n.cluster.spatial.get_clustering_from_busmap(busmap).n
    assert clustered.c.synchronous_condensers.static.at["sc", "bus"] == "X"


def test_slice_network():
    n = _network()
    assert n.slice_network(buses=["A"]).c.synchronous_condensers.static.empty
    sliced = n.slice_network(buses=["B"])
    assert list(sliced.c.synchronous_condensers.static.index) == ["sc"]


def test_netcdf_roundtrip(tmp_path):
    n = _network()
    n.add("GlobalConstraint", "H", type="inertia", sense=">=", constant=100.0)
    n.optimize()
    fn = tmp_path / "syncon.nc"
    n.export_to_netcdf(fn)
    m = pypsa.Network(fn)
    assert_allclose(
        m.c.synchronous_condensers.dynamic.s, n.c.synchronous_condensers.dynamic.s
    )
    assert m.c.synchronous_condensers.static.at["sc", "s_nom_opt"] == pytest.approx(
        50.0
    )
    assert np.isclose(m.c.synchronous_condensers.static.at["sc", "loss"], 0.01)
