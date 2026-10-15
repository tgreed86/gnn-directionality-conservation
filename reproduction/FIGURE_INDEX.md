# Figure recipe index

All 23 active PNG assets in the local 10-13 manuscript have runnable recipes. The Burgers examples use author-confirmed index 1; the main traffic curve uses the author-confirmed 48-test-trajectory selection. The architecture schematic is separate LaTeX source: `reference/architecture_schematic.tex`. Scientific settings are in `figures.json`. Commands work from the release root.

| Asset / recipe ID | Input selection | Status |
|---|---|---|
| `vanilla_meshgraphnet_linear_adv_spacetime_compare_id26` | 1D_advection/meshgraphnet_no_advection; trajectory 26; 100 transitions | ready |
| `vanilla_meshgraphnet_burgers_spacetime_compare_id300` | Burgers/meshgraphnet_no_advection; length-three bank; confirmed trajectory 1, t=0; regenerated predictions | ready |
| `meshgraphnet-w-adv_linear_adv_spacetime_compare_id26` | 1D_advection/meshgraphnet_with_advection; trajectory 26; 100 transitions | ready |
| `meshgraphnet-w-adv_burgers_spacetime_compare_id300` | Burgers/meshgraphnet_with_advection; length-three bank; confirmed trajectory 1, t=0; regenerated predictions | ready |
| `linear_advection_error_curves_compare_all` | 10 held-out trajectories; 7 plotted curves; smoothing 1 | ready |
| `burgers_error_curves_compare_all` | 10 held-out trajectories; 7 plotted curves; smoothing 9 | ready |
| `traffic_error_curves_compare_all_metrics` | 48 test trajectories, equal split/topology weighting | ready |
| `cell-migration_error_curves_compare_all_metrics` | 15 test trajectories; saved local-mesh SSIM and CFL-subcycled baselines; smoothing 9 | ready |
| `1D_heat_rollout_sample` | heat_equation/graphsage_1d; saved pair 2470 (trajectory 19); steps 1, 25, 75, 130 | ready |
| `vanilla_sageconv_linear_adv_spacetime_compare_id26` | 1D_advection/graphsage_no_advection; trajectory 26; 100 transitions | ready |
| `vanilla_sageconv_burgers_spacetime_compare_id300` | Burgers/graphsage_no_advection; length-three bank; confirmed trajectory 1, t=0; regenerated predictions | ready |
| `sageconv-w-adv_linear_adv_spacetime_compare_id26` | 1D_advection/graphsage_with_advection; trajectory 26; 100 transitions | ready |
| `sageconv-w-adv_burgers_spacetime_compare_id300` | Burgers/graphsage_with_advection; length-three bank; confirmed trajectory 1, t=0; regenerated predictions | ready |
| `meshgraphnet-abs_linear_adv_spacetime_compare_id26` | 1D_advection/meshgraphnet-abs_no_advection; trajectory 26; 100 transitions | ready |
| `meshgraphnet-abs_burgers_spacetime_compare_id300` | Burgers/meshgraphnet-abs_no_advection; length-three bank; confirmed trajectory 1, t=0; regenerated predictions | ready |
| `fluxgraphnet-w-adv_linear_adv_spacetime_compare_id26` | 1D_advection/fluxgraphnet_with_advection; trajectory 26; 100 transitions | ready |
| `fluxgraphnet-w-adv_burgers_spacetime_compare_id300` | Burgers/fluxgraphnet_with_advection; length-three bank; confirmed trajectory 1, t=0; regenerated predictions | ready |
| `linear_advection_error_curves_compare` | 10 held-out trajectories; 5 plotted curves; smoothing 1 | ready |
| `burgers_error_curves_compare` | 10 held-out trajectories; 5 plotted curves; smoothing 9 | ready |
| `traffic_topology-26_trajectory_0000_network` | fluxgraphnet_with_advection; topology 26, trajectory 0; steps 0, 85, 170, 256 | ready |
| `cell-migr_trajectory_048_rollout` | fluxgraphnet_with_advection; trajectory 48; steps 0, 43, 85, 128 | ready |
| `traffic_error_curves_compare` | 48 test trajectories, equal split/topology weighting; author-confirmed | ready |
| `cell-migration_error_curves_compare` | 15 test trajectories; saved local-mesh SSIM and CFL-subcycled baselines; smoothing 9 | ready |

Run any ready recipe by its exact ID:

```sh
python reproduction/reproduce.py figures --only RECIPE_ID
```

Outputs use the manuscript PNG filename in `outputs/figures/`. Most also have PDF exports. The traffic curve script writes separate split plots, a combined-test plot, and an averaging/SSIM summary. The runner copies the selected figure to its manuscript filename. Reconstructed rendering settings are identified in the JSON; original PNGs are retained for exact appearance.

For the schematic, include `reproduction/reference/architecture_schematic.tex` in a LaTeX document with `graphicx`, `xcolor`, `tikz`, and the TikZ `arrows.meta,calc,positioning` libraries. It is the original figure source, not a raster approximation.
