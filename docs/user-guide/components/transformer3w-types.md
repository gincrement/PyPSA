<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# 3-Winding Transformer Types

The `Transformer3WType` components describe standard 3-winding transformer types.
If for a [`Transformer3W`][pypsa.components.Transformers3W] the attribute `type` is non-empty, the electrical
parameters of the `Transformer3WType` are mapped to the [`Transformer3W`][pypsa.components.Transformers3W].

The short circuit voltages `vsc01`, `vsc12` and `vsc02` (and their real parts `vscr01`, `vscr12` and `vscr02`) of each winding pair are given relative to the smaller rating of the two windings, as on transformer nameplates. They are converted to the common base `s_nom0`:

$$z_{ij} = \frac{v_{sc,ij}}{100} \frac{s_{nom,0}}{\min(s_{nom,i}, s_{nom,j})}, \qquad r_{ij} = \frac{v_{scr,ij}}{100} \frac{s_{nom,0}}{\min(s_{nom,i}, s_{nom,j})}, \qquad x_{ij} = \sqrt{z_{ij}^2 - r_{ij}^2}$$

The magnetising shunt follows from the iron losses `pfe` and the no load current `i0`, both relative to `s_nom0`, as for 2-winding [transformer types](transformer-types.md). The tap ratio is $1 + (\text{tap_position} - \text{tap_neutral}) \cdot \text{tap_step} / 100$ at the winding given by `tap_side`.

{{ read_csv('../../../pypsa/data/component_attrs/transformer3w_types.csv') }}

The following standard 3-winding transformer types are available, which are based on [pandapower's standard types](https://pandapower.readthedocs.io/en/latest/std_types/basic.html).

{{ read_csv('../../../pypsa/data/standard_types/transformer3w_types.csv') }}
