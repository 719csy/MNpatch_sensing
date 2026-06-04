# API generation status

Generated: 2026-06-04 12:33

## Status

- Local exact visualizations: completed and remain authoritative for data values.
- OpenAI API key: visible and working.
- OpenAI image generation: success.
- NanoBanana/Gemini API key: visible and working.
- NanoBanana/Gemini image generation: success.

## Generated files

- OpenAI: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\openai_gpt-image-1.5_ncs_data_efficiency.png`
- NanoBanana/Gemini: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\nanobanana_gemini-2.5-flash-image_ncs_data_efficiency.png`
- API prompt: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\api_visualization_prompt.txt`
- Detailed log: `H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\ncs_audit_20260604\api_generated_visualizations\api_generation_log.json`

## Quality control

| Provider | Model | API status | Visual QC | Manuscript use |
| --- | --- | --- | --- | --- |
| OpenAI | `gpt-image-1.5` | Success | Usable as a presentation concept, but generated text/numeric labels are not guaranteed exact. | Do not use as the authoritative quantitative figure without local text/table overlay. |
| NanoBanana/Gemini | `gemini-2.5-flash-image` | Success | Contains visible text corruption and distorted labels. | Not publication-ready for a quantitative data figure. |

## Recommendation

For a Nature Computational Science submission, use the local programmatic visualizations as the authoritative figures because they are rendered directly from CSV/JSON:

- `visualizations\ncs_readiness_scorecard.png`
- `visualizations\domain_randomization_class_coverage.png`
- `visualizations\figure2_figure3_data_efficiency_dashboard.png`

Use API-generated images only as style exploration or backgrounds after removing generated text and overlaying exact labels from the local data tables.

## Rerun command

```powershell
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User')
$env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY','User')
if (-not $env:GEMINI_API_KEY) { $env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable('GOOGLE_API_KEY','User') }
& 'C:\Users\Songyue Chen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'H:\My Drive\Manuscript\Sparse reconstruction\Data\Data efficiency\fig23_data_efficiency_outputs\scripts\generate_api_visualizations.py'
```
