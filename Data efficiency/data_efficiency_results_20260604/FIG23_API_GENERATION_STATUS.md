# Figure 2 / Figure 3 API Visualization Status

Generated: 2026-06-04.

## API Outputs

- OpenAI image output: `api_generated_visualizations/openai_gpt-image-1.5_fig23_data_efficiency.png`
- NanoBanana/Gemini image output: `api_generated_visualizations/nanobanana_gemini-2.5-flash-image_fig23_data_efficiency.png`
- API prompt: `api_generated_visualizations/fig23_api_visualization_prompt.txt`
- API log: `api_generated_visualizations/fig23_api_generation_log.json`

## QC Decision

- OpenAI `gpt-image-1.5`: generated successfully. Layout is usable as a style reference, but labels/colors are not guaranteed to preserve the exact local CSV values.
- NanoBanana/Gemini `gemini-2.5-flash-image`: generated successfully. It contains visible text corruption and numeric/label distortion.

## Manuscript Recommendation

Use the local programmatic figures as the authoritative quantitative visualizations:

- `visualizations/figure2_data_efficiency_MAE_by_device.png`
- `visualizations/figure3_data_efficiency_MAE_by_device.png`
- `visualizations/fig23_N90_DER_summary.png`

Use the API-generated images only for visual style exploration. For any manuscript figure, overlay exact labels and values from `fig2_metrics_by_N.csv`, `fig3_metrics_by_N.csv`, and `fig23_N90_AULC_DER_summary.csv`.
