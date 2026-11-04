<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Circuit Breaker

The [`CircuitBreaker`][pypsa.components.CircuitBreakers] components open and close a [`Line`](lines.md) or [`Transformer`](transformers.md), given by `component` and `branch`. At most one active breaker can be attached to a branch.

- **Fixed status**: the static or time-varying `status` (1 closed, 0 open) takes the branch out of [`n.lpf()`][pypsa.Network.lpf], [`n.pf()`][pypsa.Network.pf] and [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] in the snapshots where it is open. The power flows solve each group of snapshots with the same breaker status with its own topology.
- **Optimal switching**: with `switchable=True`, the optimisation decides the status in each snapshot (output `status_opt`), e.g. to relieve congestion by opening a branch (Braess paradox). Switching operations (output `switch`) cost `switching_cost`, starting from `status_initial`.
- **Expansion**: with `extendable=True`, the optimisation also decides whether to install the breaker at `capital_cost` (output `installed_opt`). The branch can only be opened by an installed breaker.

See [optimal transmission switching](../optimization/power-flow.md#optimal-transmission-switching) for the formulation. [`n.optimize.fix_optimal_dispatch()`][pypsa.optimization.OptimizationAccessor.fix_optimal_dispatch] copies `status_opt` to `status` for a subsequent power flow.

``` py
>>> n = pypsa.Network()
>>> n.add("Bus", ["A", "B", "C"], v_nom=380)
>>> n.add("Generator", "cheap", bus="A", p_nom=500, marginal_cost=10)
>>> n.add("Generator", "dear", bus="C", p_nom=500, marginal_cost=100)
>>> n.add("Load", "load", bus="C", p_set=100)
>>> n.add("Line", "AC", bus0="A", bus1="C", x=1, s_nom=100)
>>> n.add("Line", "AB", bus0="A", bus1="B", x=1, s_nom=100)
>>> n.add("Line", "BC", bus0="B", bus1="C", x=1, s_nom=30)
>>> n.add("CircuitBreaker", "breaker BC", branch="BC", switchable=True)
>>> n.optimize()  # doctest: +ELLIPSIS
('ok', 'optimal')
>>> n.c.circuit_breakers.dynamic.status_opt
name      breaker BC
snapshot
now              0.0

```

Opening line B-C lets the direct line A-C carry its full capacity, instead of the 90 MW that the loop flow over A-B-C allows.

{{ read_csv('../../../pypsa/data/component_attrs/circuit_breakers.csv') }}
