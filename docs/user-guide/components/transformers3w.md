<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# 3-Winding Transformer

The [`Transformer3W`][pypsa.components.Transformers3W] components represent 3-winding transformers that connect three voltage levels. Winding 0 is attached to `bus0` (typically high voltage), winding 1 to `bus1` (typically medium voltage) and winding 2 to `bus2` (typically low voltage). The electrical parameters are either given directly or taken from a [3-winding transformer type](transformer3w-types.md).

## Star equivalent

A 3-winding transformer is modelled by its star equivalent, as in [pandapower](https://pandapower.readthedocs.io/en/latest/elements/trafo3w.html) and [DIgSILENT PowerFactory](https://www.digsilent.de/en/powerfactory.html). The pairwise short circuit impedances $z_{01}$, $z_{12}$ and $z_{02}$ (per unit on `s_nom0`), each measured with the third winding open, are converted to the star impedances

$$z_0 = \frac{z_{01} + z_{02} - z_{12}}{2}, \qquad z_1 = \frac{z_{01} + z_{12} - z_{02}}{2}, \qquad z_2 = \frac{z_{02} + z_{12} - z_{01}}{2}.$$

One of the star impedances can be negative, which is physical for this equivalent circuit.

[`n.c.transformers3w.expand()`][pypsa.components.Transformers3W.expand] adds the star equivalent to the network:

- an auxiliary [`Bus`](buses.md) `<name> star` with `v_nom`, `carrier` and coordinates of `bus0`, and
- three 2-winding [`Transformer`](transformers.md) legs: `<name> leg0` from `bus0` to the star bus, `<name> leg1` from the star bus to `bus1` and `<name> leg2` from the star bus to `bus2`. Each leg is rated at the rating of its winding, `s_nom0`, `s_nom1` or `s_nom2`, and its star impedance is converted to that base.

The magnetising shunt `g`, `b` is placed on leg 0. The tap changer is placed on the leg of the winding given by `tap_side`, at the side of the outer bus. The phase shifts `phase_shift1` and `phase_shift2` are applied to legs 1 and 2.

After expansion, [`n.pf()`][pypsa.Network.pf], [`n.lpf()`][pypsa.Network.lpf] and [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] treat the legs like any other transformer, so the flow through each winding is limited by its rating. Afterwards the flows of the legs are mapped back to `p0`, `p1`, `p2` (and `q0`, `q1`, `q2` for the non-linear power flow) of the 3-winding transformer. Call `expand()` again after changing the parameters of a 3-winding transformer to update its legs. A 3-winding transformer which is not expanded is ignored in power flow and optimisation, and a warning is logged.

``` py
>>> n = pypsa.Network()
>>> n.add("Bus", ["HV", "MV", "LV"], v_nom=[110.0, 20.0, 10.0])
>>> n.add("Generator", "grid", bus="HV", control="Slack")
>>> n.add("Load", "load MV", bus="MV", p_set=10.0, q_set=3.0)
>>> n.add("Load", "load LV", bus="LV", p_set=15.0, q_set=5.0)
>>> n.add(
...     "Transformer3W",
...     "T",
...     bus0="HV",
...     bus1="MV",
...     bus2="LV",
...     type="63/25/38 MVA 110/20/10 kV",
... )
>>> list(n.c.transformers3w.expand())
['T leg0', 'T leg1', 'T leg2']
>>> n.pf()  # doctest: +SKIP
>>> n.c.transformers3w.dynamic.p1  # doctest: +SKIP
name         T
snapshot
now      -10.0

```

## Capacity expansion

With `s_nom_extendable=True`, a 3-winding transformer is expanded as one unit, like adding parallel units of the same transformer. All three legs become extendable and the optimisation keeps their ratings in the ratio of the nominal ratings:

$$s_{nom,1}^{opt} = \frac{s_{nom,1}}{s_{nom,0}} s_{nom,0}^{opt}, \qquad s_{nom,2}^{opt} = \frac{s_{nom,2}}{s_{nom,0}} s_{nom,0}^{opt}$$

The bounds `s_nom_min`, `s_nom_max` and `s_nom_set` as well as the costs `capital_cost`, `overnight_cost` and `fom_cost` and the modular unit size `s_nom_mod` refer to winding 0. On expansion, the bounds of legs 1 and 2 are scaled by their rating ratio, while costs and modular size are put on leg 0 only. All legs share `build_year` and `lifetime`. After the optimisation, the optimised ratings are written to `s_nom0_opt`, `s_nom1_opt` and `s_nom2_opt`. [`n.optimize.fix_optimal_capacities()`][pypsa.optimization.OptimizationAccessor.fix_optimal_capacities] also fixes `s_nom0`, `s_nom1` and `s_nom2` to the optimised ratings. As for 2-winding transformers, the impedances stay per unit of the rating, so they decrease in absolute terms when the transformer is expanded.

Legs made extendable by hand, without `s_nom_extendable` on the 3-winding transformer, are expanded independently.

## Clustering

Spatial clustering does not carry over transformers. As for 2-winding transformers, 3-winding transformers and their star equivalents are dropped by [`get_clustering_from_busmap`][pypsa.clustering.SpatialClusteringAccessor.get_clustering_from_busmap], with a logged warning, and their star buses are left out of the bus aggregation. Temporal clustering keeps them.

!!! note

    3-winding transformers are not yet supported in stochastic networks.

{{ read_csv('../../../pypsa/data/component_attrs/transformers3w.csv') }}
