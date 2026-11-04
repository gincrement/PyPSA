<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Transformer

The [`Transformer`][pypsa.components.Transformers] components represent 2-winding transformers that convert AC power from one voltage level to another. They connect a `bus0` (typically at higher voltage) to a `bus1` (typically at lower voltage). Power flow through transformers is not directly controllable, but is determined passively by their impedances and the nodal power imbalances. To see how the impedances are used in the power flow, see the [transformer model](../power-flow.md#transformer-model).

## Phase-shifting transformers

A transformer with `phase_shift_min < phase_shift_max` is a controllable phase-shifting transformer (PST): the optimisation chooses its phase shift in each snapshot within these bounds (output `phase_shift_opt`), at a cost of `phase_shift_cost` per degree and hour. PSTs are expanded with `s_nom_extendable` like other transformers. A time-varying `phase_shift` is respected by [`n.lpf()`][pypsa.Network.lpf] and [`n.pf()`][pypsa.Network.pf], so that the optimised shift can be fed into the power flow with [`n.optimize.fix_optimal_dispatch()`][pypsa.optimization.OptimizationAccessor.fix_optimal_dispatch]. See [phase-shifting transformers](../optimization/power-flow.md#phase-shifting-transformers-psts) for the formulation.

A transformer can be switched by a [circuit breaker](circuit-breakers.md).

{{ read_csv('../../../pypsa/data/component_attrs/transformers.csv') }}
