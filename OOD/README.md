# Figure 4 OOD Robustness Data Package

This folder processes the available full512 Method 2b COMSOL/diffusion OOD outputs for the Advanced Science Figure 4 plan.

## Source data
- Manifest rows: 512
- Run root: `H:/My Drive/MN_simulation/outputs/rspi_joint_052026/method2b_512_five_seed_ood_full512_20260611`
- Cache dir: `H:/My Drive/MN_simulation/outputs/rspi_joint_052026/method2b_512_targetmap_diffusion_full512_20260611`
- Complete seeds used: [1, 2, 3, 4, 5]
- Missing/incomplete planned seeds: []
- Existing OOD families: morphology/shape, depth range, layer modulus, and operator noise.
- Size, compression-mismatch, and full missing-channel metric sets were not present in the current full512 run; missing-channel is included only as a controlled representative-case perturbation.

## Outputs
- `metrics/id_ood_split_summary.csv`: ID/OOD split counts.
- `metrics/fig4_fold_level_metrics.csv`: seed x model x scenario metrics copied from the completed posterior evaluator.
- `metrics/fig4_performance_by_method_scenario.csv`: cross-seed summaries.
- `metrics/fig4_performance_retention.csv`: OOD/ID retention ratios.
- `metrics/fig4_uncertainty_shift.csv`: group-level uncertainty shift tests and BH-adjusted q values.
- `metrics/fig4_failure_detection_auroc.csv`: existing pixel-error failure AUROC summaries.
- `metrics/fig4_risk_coverage.csv`: group-level risk-coverage curves.
- `selected_cases/*.npz` and `arrays/*.npy`: Figure 4b representative case arrays.
- `panels/*.png` and `panels/*.svg`: Figure 4a-e preview panels.

## Conditional diffusion summary
| scenario_label | ssim_norm_mean | mae_kpa_mean | posterior_std_norm_mean_mean | failure_auroc_top10_abs_error_mean |
| --- | --- | --- | --- | --- |
| ID held-out | 0.2639 | 2.9316 | 0.0658 | 0.7946 |
| Depth OOD | 0.2426 | 3.2413 | 0.0690 | 0.7967 |
| Mechanical-background OOD | 0.2431 | 3.2079 | 0.0733 | 0.8118 |
| Noise/drift OOD | 0.2097 | 2.8135 | 0.0781 | 0.7751 |
| Morphology OOD | 0.1824 | 3.5619 | 0.0785 | 0.7978 |

## Real-anchor audit
| anchor_type | found_candidate_file_count | status |
| --- | --- | --- |
| phantom | 0 | missing |
| animal | 20 | candidate_files_found_needs_manual_schema_check |
| ex_vivo | 0 | missing |
| clinical_complex | 0 | missing |

Candidate real-anchor files are not automatically treated as usable unless they match the B8/reference-map schema. No synthetic real-world anchor data were fabricated.

## Representative cases
| case_key | status | scenario_label | condition_id | synthetic_note |
| --- | --- | --- | --- | --- |
| fig4b_irregular_morphology | generated | Morphology OOD | F3NCS_0374_ood_unseen_shape_lobed_zb07p3_E27p3 |  |
| fig4b_deep_lesion | generated | Depth OOD | F3NCS_0390_ood_unseen_depth_range_two_foci_zb08p6_E39p9 |  |
| fig4b_noisy_input | generated | Noise/drift OOD | F3NCS_0485_ood_unseen_operator_noise_two_foci_zb07p5_E25p6 |  |
| fig4b_missing_channels | generated | ID held-out | F3NCS_0139_iid_support_two_foci_zb03p2_E25p1 | Controlled missing-channel perturbation applied to an ID held-out case. |

## Limitations
- The completed posterior evaluator stores fold/test-group metrics, not per-sample posterior arrays for all 512 cases. Therefore uncertainty shift and risk-coverage are reported at seed-test-group granularity.
- Figure 4b arrays are generated from the selected seed checkpoint only and are intended as representative visual material, while quantitative panels use all complete seeds.
- Current full512 outputs use 64x64 target maps. The pipeline keeps native resolution instead of upsampling to 256x256.
