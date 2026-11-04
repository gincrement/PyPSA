# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa
from pypsa.constants import DEFAULT_TIMESTAMP

PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
)
@pytest.mark.skipif(PANDAS_3, reason="pandapower does not yet support pandas 3")
@pytest.mark.parametrize("use_pandapower_index", [True, False])
@pytest.mark.parametrize("extra_line_data", [True, False])
def test_pandapower_custom_case(
    pandapower_custom_network, use_pandapower_index, extra_line_data
):
    import pandapower as pp

    net = pandapower_custom_network
    # because of phase angles, need to init with DC
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    n = pypsa.Network()
    n.import_from_pandapower_net(
        net, use_pandapower_index=use_pandapower_index, extra_line_data=extra_line_data
    )

    # seed PF with LPF solution because of phase angle jumps
    n.lpf()
    n.pf(use_seed=True)

    # use same index for everything
    net.res_bus.index = net.bus.name.values
    net.res_line.index = net.line.name.values

    # compare bus angles
    equal(
        n.c.buses.dynamic.v_ang.loc[DEFAULT_TIMESTAMP] * 180 / np.pi,
        net.res_bus.va_degree,
    )

    # compare bus voltage magnitudes
    equal(n.c.buses.dynamic.v_mag_pu.loc[DEFAULT_TIMESTAMP], net.res_bus.vm_pu)

    # pandapower folds shunt impedance power into the bus totals, while PyPSA
    # reports it on the shunt component (p positive if load, q positive if
    # generation), so add it back in the bus injection convention before comparing
    buses_i = n.c.buses.static.index
    shunt = n.c.shunt_impedances
    by_bus = shunt.static.bus
    p_per_shunt = shunt.dynamic.p.loc[DEFAULT_TIMESTAMP]
    q_per_shunt = shunt.dynamic.q.loc[DEFAULT_TIMESTAMP]
    shunt_p = p_per_shunt.groupby(by_bus).sum().reindex(buses_i, fill_value=0.0)
    shunt_q = q_per_shunt.groupby(by_bus).sum().reindex(buses_i, fill_value=0.0)
    bus_p = n.c.buses.dynamic.p.loc[DEFAULT_TIMESTAMP] - shunt_p
    bus_q = n.c.buses.dynamic.q.loc[DEFAULT_TIMESTAMP] + shunt_q

    # compare bus active power (NB: pandapower uses load signs)
    equal(bus_p, -net.res_bus.p_mw)

    # compare bus reactive power (NB: pandapower uses load signs)
    equal(bus_q, -net.res_bus.q_mvar)

    # compare branch flows
    equal(n.c.lines.dynamic.p0.loc[DEFAULT_TIMESTAMP], net.res_line.p_from_mw)
    equal(n.c.lines.dynamic.p1.loc[DEFAULT_TIMESTAMP], net.res_line.p_to_mw)
    equal(n.c.lines.dynamic.q0.loc[DEFAULT_TIMESTAMP], net.res_line.q_from_mvar)
    equal(n.c.lines.dynamic.q1.loc[DEFAULT_TIMESTAMP], net.res_line.q_to_mvar)

    equal(n.c.transformers.dynamic.p0.loc[DEFAULT_TIMESTAMP], net.res_trafo.p_hv_mw)
    equal(n.c.transformers.dynamic.p1.loc[DEFAULT_TIMESTAMP], net.res_trafo.p_lv_mw)
    equal(n.c.transformers.dynamic.q0.loc[DEFAULT_TIMESTAMP], net.res_trafo.q_hv_mvar)
    equal(n.c.transformers.dynamic.q1.loc[DEFAULT_TIMESTAMP], net.res_trafo.q_lv_mvar)


def _pandapower_trafo3w_network(std_type: bool):
    pp = pytest.importorskip("pandapower", reason="pandapower not installed")
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.0, name="HV")
    mv = pp.create_bus(net, vn_kv=20.0, name="MV")
    lv = pp.create_bus(net, vn_kv=10.0, name="LV")
    lv2 = pp.create_bus(net, vn_kv=10.0, name="LV 2")
    pp.create_ext_grid(net, bus=hv, vm_pu=1.02, name="Grid")
    pp.create_load(net, bus=mv, p_mw=10.0, q_mvar=3.0, name="Load MV")
    pp.create_load(net, bus=lv2, p_mw=15.0, q_mvar=5.0, name="Load LV")
    pp.create_line(
        net, lv, lv2, length_km=0.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="L"
    )
    if std_type:
        pp.create_transformer3w(
            net, hv, mv, lv, std_type="63/25/38 MVA 110/20/10 kV", tap_pos=2, name="T"
        )
    else:
        pp.create_transformer3w_from_parameters(
            net,
            hv,
            mv,
            lv,
            vn_hv_kv=110.0,
            vn_mv_kv=20.0,
            vn_lv_kv=10.0,
            sn_hv_mva=40.0,
            sn_mv_mva=20.0,
            sn_lv_mva=30.0,
            vk_hv_percent=11.0,
            vk_mv_percent=9.0,
            vk_lv_percent=12.0,
            vkr_hv_percent=0.3,
            vkr_mv_percent=0.35,
            vkr_lv_percent=0.4,
            pfe_kw=30.0,
            i0_percent=0.5,
            tap_side="mv",
            tap_neutral=0,
            tap_min=-5,
            tap_max=5,
            tap_step_percent=1.5,
            tap_pos=-2,
            tap_changer_type="Ratio",
            name="T",
        )
    return net


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
)
@pytest.mark.skipif(PANDAS_3, reason="pandapower does not yet support pandas 3")
@pytest.mark.parametrize("std_type", [True, False])
def test_pandapower_trafo3w(std_type):
    import pandapower as pp

    net = _pandapower_trafo3w_network(std_type)
    pp.runpp(net, calculate_voltage_angles=True)

    n = pypsa.Network()
    n.import_from_pandapower_net(net)
    assert list(n.c.transformers3w.static.index) == ["T"]

    n.lpf()
    n.pf(use_seed=True)

    buses = net.bus.name.values
    equal(
        n.c.buses.dynamic.v_mag_pu.loc[DEFAULT_TIMESTAMP, buses],
        net.res_bus.vm_pu.values,
    )
    equal(
        n.c.buses.dynamic.v_ang.loc[DEFAULT_TIMESTAMP, buses] * 180 / np.pi,
        net.res_bus.va_degree.values,
    )

    res = net.res_trafo3w.iloc[0]
    dynamic = n.c.transformers3w.dynamic
    for attr, col in {
        "p0": "p_hv_mw",
        "q0": "q_hv_mvar",
        "p1": "p_mv_mw",
        "q1": "q_mv_mvar",
        "p2": "p_lv_mw",
        "q2": "q_lv_mvar",
    }.items():
        equal(dynamic[attr].loc[DEFAULT_TIMESTAMP, "T"], res[col])


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
)
@pytest.mark.skipif(PANDAS_3, reason="pandapower does not yet support pandas 3")
@pytest.mark.parametrize("std_type", [True, False])
def test_pandapower_trafo(std_type):
    """2-winding transformers imported by standard type or from parameters."""
    pp = pytest.importorskip("pandapower", reason="pandapower not installed")

    type_name = "0.4 MVA 20/0.4 kV"
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=20.0, name="HV")
    lv = pp.create_bus(net, vn_kv=0.4, name="LV")
    pp.create_ext_grid(net, bus=hv, vm_pu=1.0, name="Grid")
    pp.create_load(net, bus=lv, p_mw=0.3, q_mvar=0.1, name="Load")
    if std_type:
        pp.create_transformer(net, hv, lv, std_type=type_name, name="T")
    else:
        params = net.std_types["trafo"][type_name]
        pp.create_transformer_from_parameters(
            net,
            hv,
            lv,
            sn_mva=params["sn_mva"],
            vn_hv_kv=20.0,
            vn_lv_kv=0.4,
            vk_percent=params["vk_percent"],
            vkr_percent=params["vkr_percent"],
            pfe_kw=params["pfe_kw"],
            i0_percent=params["i0_percent"],
            shift_degree=params["shift_degree"],
            name="T",
        )
    pp.runpp(net, calculate_voltage_angles=True)

    n = pypsa.Network()
    n.import_from_pandapower_net(net)
    n.lpf()
    n.pf(use_seed=True)

    buses = net.bus.name.values
    equal(
        n.c.buses.dynamic.v_mag_pu.loc[DEFAULT_TIMESTAMP, buses],
        net.res_bus.vm_pu.values,
    )
    equal(
        n.c.buses.dynamic.v_ang.loc[DEFAULT_TIMESTAMP, buses] * 180 / np.pi,
        net.res_bus.va_degree.values,
    )
    equal(n.c.transformers.dynamic.p0.loc[DEFAULT_TIMESTAMP], net.res_trafo.p_hv_mw)
    equal(n.c.transformers.dynamic.q0.loc[DEFAULT_TIMESTAMP], net.res_trafo.q_hv_mvar)
