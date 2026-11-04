<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Synchronous Condenser

The [`SynchronousCondenser`][pypsa.components.SynchronousCondensers] components represent synchronous machines without prime mover. They provide reactive power for voltage control, inertia and short-circuit power, and draw active power for their losses.

## Optimisation

The operational variable of a synchronous condenser is its online capacity $s_{k,t}$ (output `s`), bounded by `s_min_pu` and `s_max_pu` times the nominal capacity `s_nom`, which can be extended with `s_nom_extendable` at `capital_cost` like for other components. The online capacity draws the losses `loss` $\cdot\, s_{k,t}$ from the bus (output `p`, negative) and costs `marginal_cost` per MVA and hour.

Since the linear optimisation has no reactive power, nothing makes synchronous condensers go online or be built by themselves. Their dispatch and expansion are driven by [system strength requirements](../optimization/global-constraints.md#system-strength): global constraints of type `inertia` and `short_circuit_power`, to which the online capacity contributes `inertia_constant` $\cdot\, s_{k,t}$ (in MWs) and `short_circuit_factor` $\cdot\, s_{k,t}$ (in MVA).

## Power flow

In [`n.pf()`][pypsa.Network.pf], a synchronous condenser with `control="PV"` (the default) controls the voltage magnitude of its bus to the bus attribute `v_mag_pu_set`; its reactive power `q` follows from the power flow. At a bus with a voltage controlling generator, the reactive power is shared in proportion to `p_nom` of the generator and `s_nom` of the condensers. With `control="PQ"`, the condenser injects `q_set`. In both cases it draws the active power `p_set`, which [`n.optimize.fix_optimal_dispatch()`][pypsa.optimization.OptimizationAccessor.fix_optimal_dispatch] sets to the optimised losses.

!!! note

    Reactive power limits are not enforced in the power flow.

{{ read_csv('../../../pypsa/data/component_attrs/synchronous_condensers.csv') }}
