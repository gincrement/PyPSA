# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for 3-winding transformers modelled by their star equivalent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import pypsa
from pypsa.components._types.transformers3w import delta_to_star

TYPE = "63/25/38 MVA 110/20/10 kV"
OVERRIDE_MSG = "overridden by the standard type"


def _network(**kwargs) -> pypsa.Network:
    """HV/MV/LV network fed from HV with loads at MV and LV."""
    n = pypsa.Network()
    n.add("Carrier", "AC")
    n.add("Bus", ["HV", "MV", "LV"], v_nom=[110.0, 20.0, 10.0], carrier="AC")
    n.c.buses.static.loc["HV", ["x", "y"]] = [1.0, 2.0]
    n.add("Generator", "Grid", bus="HV", control="Slack", p_nom=100.0)
    n.add("Load", "Load MV", bus="MV", p_set=10.0, q_set=3.0)
    n.add("Load", "Load LV", bus="LV", p_set=15.0, q_set=5.0)
    n.add("Transformer3W", "T", bus0="HV", bus1="MV", bus2="LV", **kwargs)
    return n


def _manual_params(**overrides) -> dict:
    params = {
        "s_nom0": 100.0,
        "s_nom1": 50.0,
        "s_nom2": 40.0,
        "r01": 0.004,
        "x01": 0.10,
        "r12": 0.006,
        "x12": 0.20,
        "r02": 0.005,
        "x02": 0.16,
    }
    params.update(overrides)
    return params


# Star equivalent
# ---------------


def test_delta_to_star_recovers_pairwise_impedances():
    z01, z12, z02 = 0.1 + 0.4j, 0.2 + 0.3j, 0.15 + 0.5j
    z0, z1, z2 = delta_to_star(z01, z12, z02)
    assert z0 + z1 == pytest.approx(z01)
    assert z1 + z2 == pytest.approx(z12)
    assert z0 + z2 == pytest.approx(z02)


def test_star_equivalent_manual_parameters():
    n = _network(
        **_manual_params(),
        g=0.001,
        b=-0.002,
        tap_side=1,
        tap_ratio=1.05,
        phase_shift1=30.0,
        phase_shift2=-150.0,
    )
    legs = n.c.transformers3w.star_equivalent()

    assert list(legs.index) == ["T leg0", "T leg1", "T leg2"]
    assert list(legs.bus0) == ["HV", "T star", "T star"]
    assert list(legs.bus1) == ["T star", "MV", "LV"]
    assert_allclose(legs.s_nom, [100.0, 50.0, 40.0])

    # star impedances on base s_nom0, rescaled to each leg's own rating
    z0, z1, z2 = delta_to_star(0.004 + 0.1j, 0.006 + 0.2j, 0.005 + 0.16j)
    expected = np.array([z0, z1 * 50.0 / 100.0, z2 * 40.0 / 100.0])
    assert_allclose(legs.r, expected.real)
    assert_allclose(legs.x, expected.imag)

    # magnetising shunt on leg 0 only
    assert_allclose(legs.g, [0.001, 0.0, 0.0])
    assert_allclose(legs.b, [-0.002, 0.0, 0.0])

    # tap changer on winding 1, at the outer bus of leg 1
    assert_allclose(legs.tap_ratio, [1.0, 1.05, 1.0])
    assert list(legs.tap_side) == [0, 1, 1]

    assert_allclose(legs.phase_shift, [0.0, 30.0, -150.0])


def test_star_equivalent_pairwise_impedances_seen_between_buses():
    """The series impedance between two outer buses equals the pairwise input."""
    n = _network(**_manual_params())
    legs = n.c.transformers3w.star_equivalent()
    # back to common base s_nom0 = 100
    z = (legs.r + 1j * legs.x) * 100.0 / legs.s_nom
    z0, z1, z2 = z.to_numpy()
    assert z0 + z1 == pytest.approx(0.004 + 0.1j)
    assert z1 + z2 == pytest.approx(0.006 + 0.2j)
    assert z0 + z2 == pytest.approx(0.005 + 0.16j)


# Standard types
# --------------


def test_default_standard_types_available():
    n = pypsa.Network()
    assert TYPE in n.c.transformer3w_types.static.index
    assert "63/25/38 MVA 110/10/10 kV" in n.c.transformer3w_types.static.index


def test_type_derives_parameters(caplog):
    n = _network(type=TYPE, tap_position=2)
    n.c.transformers3w.star_equivalent()
    t = n.c.transformers3w.static.loc["T"]

    assert OVERRIDE_MSG not in caplog.text
    assert_allclose(t[["s_nom0", "s_nom1", "s_nom2"]].astype(float), [63, 25, 38])

    # vsc01 = 10.4 % referred to min(63, 25) = 25 MVA, rescaled to base 63 MVA
    z01 = 0.104 * 63 / 25
    r01 = 0.0028 * 63 / 25
    assert t.r01 == pytest.approx(r01)
    assert t.x01 == pytest.approx(np.sqrt(z01**2 - r01**2))
    # vsc12 referred to min(25, 38) = 25 MVA
    assert t.r12 == pytest.approx(0.0032 * 63 / 25)
    # vsc02 referred to min(63, 38) = 38 MVA
    assert t.r02 == pytest.approx(0.0035 * 63 / 38)

    g = 35 / (1000 * 63)
    assert t.g == pytest.approx(g)
    assert t.b == pytest.approx(-np.sqrt(0.0089**2 - g**2))

    assert t.tap_side == 0
    assert t.tap_ratio == pytest.approx(1 + 2 * 0.012)


def test_manual_impedance_override_warns(caplog):
    n = _network(type=TYPE)
    n.c.transformers3w.star_equivalent()

    n.c.transformers3w.static.loc["T", ["r01", "x01"]] = 0.01
    n.c.transformers3w.star_equivalent()
    assert OVERRIDE_MSG in caplog.text
    assert n.c.transformers3w.static.loc["T", "r01"] != 0.01

    caplog.clear()
    n.c.transformers3w.star_equivalent()
    assert OVERRIDE_MSG not in caplog.text


def test_missing_type_raises():
    n = _network(type="does not exist")
    with pytest.raises(ValueError, match="does not exist"):
        n.c.transformers3w.expand()


# Expansion
# ---------


def test_expand_adds_star_bus_and_legs():
    n = _network(type=TYPE)
    legs = n.c.transformers3w.expand()

    assert list(legs) == ["T leg0", "T leg1", "T leg2"]
    assert n.c.transformers3w.static.loc["T", "star_bus"] == "T star"

    star = n.c.buses.static.loc["T star"]
    assert star.v_nom == 110.0
    assert star.carrier == "AC"
    assert (star.x, star.y) == (1.0, 2.0)

    trafos = n.c.transformers.static.loc[legs]
    assert list(trafos.bus0) == ["HV", "T star", "T star"]
    assert list(trafos.bus1) == ["T star", "MV", "LV"]
    # legs carry the derived parameters, not the 2-winding type
    assert (trafos.type == "").all()


def test_expand_is_idempotent_and_updates_legs():
    n = _network(**_manual_params())
    n.c.transformers3w.expand()
    n.c.transformers3w.expand()
    assert len(n.c.buses.static) == 4
    assert len(n.c.transformers.static) == 3

    n.c.transformers3w.static.loc["T", "x01"] = 0.3
    n.c.transformers3w.expand()
    expected = n.c.transformers3w.star_equivalent()
    assert_allclose(n.c.transformers.static.loc[expected.index, "x"], expected.x)


def test_expand_empty():
    n = pypsa.Network()
    assert n.c.transformers3w.expand().empty
    assert n.c.transformers.static.empty


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"s_nom1": 0.0}, "positive ratings"),
        ({"tap_side": 3}, "tap_side"),
    ],
)
def test_expand_invalid_parameters_raise(overrides, match):
    n = _network(**_manual_params(**overrides))
    with pytest.raises(ValueError, match=match):
        n.c.transformers3w.expand()


def test_expand_unknown_bus_raises():
    n = _network(**_manual_params())
    n.c.transformers3w.static.loc["T", "bus2"] = "missing"
    with pytest.raises(ValueError, match="missing"):
        n.c.transformers3w.expand()


def test_unexpanded_transformer_warns(caplog):
    n = _network(**_manual_params())
    n.add("Line", "L", bus0="HV", bus1="MV", x=0.1, r=0.01)
    n.lpf()
    assert "transformers3w.expand()" in caplog.text

    caplog.clear()
    n.c.transformers3w.expand()
    n.lpf()
    assert "transformers3w.expand()" not in caplog.text


# Power flow
# ----------

# Reference results from pandapower 3.5.6 for the same network
# (`pp.create_transformer3w(..., std_type=TYPE, tap_pos=...)`, ext_grid at HV)
PANDAPOWER_REFERENCE = {
    0: {
        "v_mag_pu": [1.0, 0.9762851070260677, 0.9779883572193976, 0.9860255852667055],
        "v_ang": [0.0, -3.6192179165187177, -3.1739580580278868, -1.9712400358980489],
        "p0": 25.084761051153436,
        "q0": 10.223251752095393,
    },
    3: {
        "v_mag_pu": [1.0, 0.9405198976058514, 0.9423044028460129, 0.9506632374271852],
        "v_ang": [0.0, -3.8925187505655727, -3.4129133119599206, -2.1182023262338388],
        "p0": 25.086237012965487,
        "q0": 10.313642715604828,
    },
}


@pytest.mark.parametrize("tap_position", [0, 3])
def test_pf_against_pandapower_reference(tap_position):
    n = _network(type=TYPE, tap_position=tap_position)
    n.c.transformers3w.expand()
    n.lpf()
    n.pf(use_seed=True)

    ref = PANDAPOWER_REFERENCE[tap_position]
    buses = ["HV", "MV", "LV", "T star"]
    sn = n.snapshots[0]
    assert_allclose(
        n.c.buses.dynamic.v_mag_pu.loc[sn, buses], ref["v_mag_pu"], rtol=1e-6
    )
    assert_allclose(
        np.rad2deg(n.c.buses.dynamic.v_ang.loc[sn, buses]),
        ref["v_ang"],
        atol=1e-6,
    )

    dynamic = n.c.transformers3w.dynamic
    assert dynamic.p0.loc[sn, "T"] == pytest.approx(ref["p0"], rel=1e-6)
    assert dynamic.q0.loc[sn, "T"] == pytest.approx(ref["q0"], rel=1e-6)
    assert dynamic.p1.loc[sn, "T"] == pytest.approx(-10.0)
    assert dynamic.q1.loc[sn, "T"] == pytest.approx(-3.0)
    assert dynamic.p2.loc[sn, "T"] == pytest.approx(-15.0)
    assert dynamic.q2.loc[sn, "T"] == pytest.approx(-5.0)


def test_lpf_maps_flows_and_balances():
    n = _network(**_manual_params())
    n.c.transformers3w.expand()
    n.lpf()

    dynamic = n.c.transformers3w.dynamic
    sn = n.snapshots[0]
    assert dynamic.p0.loc[sn, "T"] == pytest.approx(25.0)
    assert dynamic.p1.loc[sn, "T"] == pytest.approx(-10.0)
    assert dynamic.p2.loc[sn, "T"] == pytest.approx(-15.0)


def test_lpf_phase_shift():
    """A phase shift on winding 1 rotates the MV angle relative to the star bus."""
    angles = {}
    for shift in [0.0, 30.0]:
        n = _network(**_manual_params(), phase_shift1=shift)
        n.c.transformers3w.expand()
        n.lpf()
        v_ang = n.c.buses.dynamic.v_ang.iloc[0]
        angles[shift] = v_ang["MV"] - v_ang["T star"]
    assert angles[30.0] - angles[0.0] == pytest.approx(-np.deg2rad(30.0))


def test_inactive_transformer_creates_inactive_legs():
    n = _network(**_manual_params(), active=False)
    legs = n.c.transformers3w.expand()
    assert not n.c.transformers.static.loc[legs, "active"].any()


# Optimisation
# ------------


def test_optimize_respects_winding_ratings():
    n = _network(**_manual_params(s_nom1=8.0))
    n.c.generators.static.loc["Grid", "marginal_cost"] = 10.0
    n.add("Generator", "MV gen", bus="MV", p_nom=50.0, marginal_cost=100.0)
    n.c.transformers3w.expand()

    status, _ = n.optimize()
    assert status == "ok"

    dynamic = n.c.transformers3w.dynamic
    sn = n.snapshots[0]
    # winding 1 is limited to 8 MW, the rest of the MV load is served locally
    assert dynamic.p1.loc[sn, "T"] == pytest.approx(-8.0)
    assert dynamic.p2.loc[sn, "T"] == pytest.approx(-15.0)
    assert dynamic.p0.loc[sn, "T"] == pytest.approx(23.0)
    assert n.c.generators.dynamic.p.loc[sn, "MV gen"] == pytest.approx(2.0)


# Network handling
# ----------------


def test_netcdf_roundtrip(tmp_path):
    n = _network(type=TYPE, tap_position=1)
    n.c.transformers3w.expand()
    n.lpf()

    fn = tmp_path / "trafo3w.nc"
    n.export_to_netcdf(fn)
    m = pypsa.Network(fn)

    cols = ["bus0", "bus1", "bus2", "type", "tap_position", "star_bus"]
    pd.testing.assert_frame_equal(
        m.c.transformers3w.static[cols],
        n.c.transformers3w.static[cols],
        check_dtype=False,
    )
    assert_allclose(m.c.transformers3w.dynamic.p1, n.c.transformers3w.dynamic.p1)
    # standard types are not exported by default but still available
    assert TYPE in m.c.transformer3w_types.static.index


def test_slice_network_keeps_only_fully_contained():
    n = _network(**_manual_params())
    n.c.transformers3w.expand()

    sliced = n.slice_network(buses=["HV", "MV", "LV", "T star"])
    assert list(sliced.c.transformers3w.static.index) == ["T"]

    sliced = n.slice_network(buses=["HV", "MV"])
    assert sliced.c.transformers3w.static.empty


def test_not_supported_for_stochastic_networks():
    n = _network(**_manual_params())
    n.set_scenarios({"a": 0.5, "b": 0.5})
    with pytest.raises(NotImplementedError, match="stochastic"):
        n.c.transformers3w.expand()


# Capacity expansion
# ------------------


def _expansion_network(**kwargs) -> pypsa.Network:
    """Cheap supply at HV, expensive at MV; loads of 40 MW at MV and 20 MW at LV."""
    n = pypsa.Network()
    n.add("Bus", ["HV", "MV", "LV"], v_nom=[110.0, 20.0, 10.0])
    n.add("Generator", "grid", bus="HV", p_nom=1000.0, marginal_cost=10.0)
    n.add("Generator", "MV gen", bus="MV", p_nom=1000.0, marginal_cost=100.0)
    n.add("Load", "MV load", bus="MV", p_set=40.0)
    n.add("Load", "LV load", bus="LV", p_set=20.0)
    params = {
        **_manual_params(),
        "s_nom_extendable": True,
        "capital_cost": 1.0,
        **kwargs,
    }
    n.add("Transformer3W", "T", bus0="HV", bus1="MV", bus2="LV", **params)
    n.c.transformers3w.expand()
    return n


def test_star_equivalent_expansion_parameters():
    n = _network(
        **_manual_params(),
        s_nom_extendable=True,
        s_nom_min=10.0,
        s_nom_max=200.0,
        s_nom_mod=20.0,
        capital_cost=5.0,
        fom_cost=1.0,
        build_year=2030,
        lifetime=40.0,
    )
    legs = n.c.transformers3w.star_equivalent()

    assert legs.s_nom_extendable.all()
    # bounds scale with the winding ratings 100:50:40
    assert_allclose(legs.s_nom_min, [10.0, 5.0, 4.0])
    assert_allclose(legs.s_nom_max, [200.0, 100.0, 80.0])
    # costs and modular size sit on leg 0 only
    assert_allclose(legs.capital_cost, [5.0, 0.0, 0.0])
    assert_allclose(legs.fom_cost, [1.0, 0.0, 0.0])
    assert_allclose(legs.s_nom_mod, [20.0, 0.0, 0.0])
    # all legs share the build year and lifetime
    assert list(legs.build_year) == [2030] * 3
    assert_allclose(legs.lifetime, [40.0] * 3)


def test_optimize_expands_as_one_unit():
    n = _expansion_network()
    status, _ = n.optimize()
    assert status == "ok"

    assert "Transformer3W-s_nom1-coupling" in n.model.constraints
    assert "Transformer3W-s_nom2-coupling" in n.model.constraints

    # winding 1 must carry 40 MW, which fixes the size of the unit
    t = n.c.transformers3w.static.loc["T"]
    assert t.s_nom1_opt == pytest.approx(40.0)
    assert t.s_nom0_opt == pytest.approx(80.0)
    assert t.s_nom2_opt == pytest.approx(32.0)
    # capital costs refer to winding 0 only
    total = n.objective + n.objective_constant
    assert total == pytest.approx(80.0 * 1.0 + 60.0 * 10.0)
    assert n.c.transformers3w.dynamic.p1.iloc[0]["T"] == pytest.approx(-40.0)


def test_optimize_respects_s_nom_max():
    n = _expansion_network(s_nom_max=60.0)
    n.optimize()

    t = n.c.transformers3w.static.loc["T"]
    assert t.s_nom0_opt == pytest.approx(60.0)
    assert t.s_nom1_opt == pytest.approx(30.0)
    assert t.s_nom2_opt == pytest.approx(24.0)
    # the remaining 10 MW of the MV load is served locally
    assert n.c.generators.dynamic.p.iloc[0]["MV gen"] == pytest.approx(10.0)


def test_non_extendable_reports_ratings():
    n = _expansion_network(s_nom_extendable=False)
    n.c.loads.static.loc["MV load", "p_set"] = 10.0
    n.optimize()

    assert "Transformer3W-s_nom1-coupling" not in n.model.constraints
    t = n.c.transformers3w.static.loc["T"]
    opt = t[["s_nom0_opt", "s_nom1_opt", "s_nom2_opt"]].astype(float)
    assert_allclose(opt, [100, 50, 40])


def test_manually_extendable_legs_are_not_coupled():
    """Legs made extendable by hand keep expanding independently."""
    n = _expansion_network(s_nom_extendable=False)
    n.c.transformers.static["s_nom_extendable"] = True
    n.optimize()

    assert "Transformer3W-s_nom1-coupling" not in n.model.constraints
    t = n.c.transformers3w.static.loc["T"]
    assert t.s_nom1_opt == pytest.approx(40.0)
    assert t.s_nom2_opt == pytest.approx(20.0)


def test_fix_optimal_capacities():
    n = _expansion_network()
    n.optimize()
    n.optimize.fix_optimal_capacities()

    t = n.c.transformers3w.static.loc["T"]
    assert not t.s_nom_extendable
    assert_allclose(t[["s_nom0", "s_nom1", "s_nom2"]].astype(float), [80, 40, 32])
    legs = n.c.transformers.static
    assert not legs.s_nom_extendable.any()

    # re-expanding keeps the fixed ratings
    n.c.transformers3w.expand()
    assert_allclose(legs.loc[["T leg0", "T leg1", "T leg2"], "s_nom"], [80, 40, 32])


# Clustering
# ----------


def test_spatial_clustering_drops_transformers3w(caplog):
    n = _network(**_manual_params())
    n.add("Bus", "HV2", v_nom=110.0)
    n.add("Line", "L", bus0="HV", bus1="HV2", x=0.1, s_nom=100.0)
    n.c.transformers3w.expand()

    busmap = pd.Series(
        {"HV": "A", "HV2": "A", "MV": "B", "LV": "C", "T star": "A"}, name="busmap"
    )
    clustering = n.cluster.spatial.get_clustering_from_busmap(busmap)
    clustered = clustering.n

    assert "3-winding transformer(s) T and their star equivalents are dropped" in (
        caplog.text
    )
    assert clustered.c.transformers3w.static.empty
    assert clustered.c.transformers.static.empty
    assert sorted(clustered.c.buses.static.index) == ["A", "B", "C"]
    # the star bus does not shift the aggregated coordinates of cluster A
    assert clustered.c.buses.static.loc["A", "x"] == pytest.approx(
        n.c.buses.static.loc[["HV", "HV2"], "x"].mean()
    )
    pd.testing.assert_series_equal(clustering.busmap, busmap)


def test_spatial_clustering_star_bus_in_own_cluster():
    n = _network(**_manual_params())
    n.c.transformers3w.expand()

    busmap = pd.Series({"HV": "A", "MV": "B", "LV": "C", "T star": "S"})
    clustered = n.cluster.spatial.get_clustering_from_busmap(busmap).n
    assert "S" not in clustered.c.buses.static.index


def test_temporal_clustering_keeps_transformers3w():
    n = _network(type=TYPE)
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    n.c.transformers3w.expand()

    m = n.cluster.temporal.resample("2h")
    assert list(m.c.transformers3w.static.index) == ["T"]
    assert m.c.transformers3w.static.loc["T", "star_bus"] == "T star"
    m.lpf()
    assert_allclose(m.c.transformers3w.dynamic.p1["T"], -10.0)
