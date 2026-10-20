# MMM Recovery Checkpoint

Checkpoint date: 2026-10-10. This file is the portable handoff; chat archives are supporting history, not executable specifications.

## Start Here

Read README.md and MMM_Implementation_Blueprint.pdf, then review outputs_kalman_profit_validated/run_config.json, metrics.json, and run_report.md. Do not refit or overwrite results just to resume.

Current objective: an understandable synthetic MMM workflow for forecasting, attribution, expected-profit allocation, and risk-adjusted allocation, with a blueprint for later real-data adaptation. No real-data ingestion pipeline has been implemented yet.

## Current Model

- 156 weekly periods, four products, three tiers; 130 training weeks and 26 held out. Seed 20261007.
- Bayesian product/tier baseline pooling and positive product-level media shrinkage. Display, Video, Social, CTV use adstock and saturation; paid search has no adstock. Social uses an S-shaped Hill curve.
- CTV affects sales directly and through a search/click funnel. Mediator likelihoods use unique training product-weeks.
- A Kalman filter updates a stochastic latent logit brand state from training survey scores. Media, sponsorship, and weighted distribution drive its target. Holdout scores do not update the state.
- Previous-week brand score contributes to baseline log sales. Media has separate sales and brand coefficients; sponsorship has no immediate-sales term. There is no separately filtered dynamic sales baseline.
- Sales are censored at supplied inventory caps. Nonlinear sales are not Gaussian Kalman observations; brand-to-sales uses a logistic-normal moment approximation. This is not an exact fully joint latent-state sales likelihood.
- Planning carries adstock and filtered brand state forward. Fixed weekly shares for 13 weeks, fixed product proportions, six spend lines including sponsorship, fixed future coverage and inventory.
- Three allocations: expected units, expected net contribution, and 50% predictive mean profit plus 50% lower-tail 5% CVaR over 512 scenarios.
- Margins are illustrative product-tier rates, not measured COGS. Taxes, returns, fulfillment/fixed costs, uncertain media prices, replenishment, substitution, and benefits beyond 13 weeks are excluded.

## Verified Reference

outputs_kalman_profit_validated is the current saved reference. The recorded fit passed all gates: zero divergences, max R-hat 1.0055; brand RMSE 1.26 versus 3.64 last-score benchmark. Expected synthetic net contribution: about $948.5m baseline and $953.6m optimized. Predictive profit CVaR: about $944.3m baseline and $952.8m risk allocation. These are synthetic dollars, not business forecasts. Nineteen regression tests passed in the previous validation; they have not been rerun merely for this checkpoint.

Earlier output folders and PDFs remain historical references, including the deterministic-brand fit. Source has evolved since those fits: old commands need the matching historic commit to reproduce the old specification. Preserve Read MMM.py, SimpleAdd.py, and user edits.

## Amazon Extension Status (2026-10-19)

Source now has an opt-in `--amazon` synthetic extension; the default five-channel generator/model and saved validated folders remain unchanged. Amazon Display is a separate impression channel at an illustrative $14 CPM. Amazon Search uses synthetic on-Amazon query opportunities and a spend-dependent Binomial click mediator; realized synthetic spend is paid clicks times an illustrative $2 CPC, so no clicks means no click charges. Planned Search budgets are caps/activity inputs; expected CPC charges vary with clicks and can leave budget unused. Amazon Search clicks are not individually linked to purchases. All 20 regression tests pass on the final source, including generator reconstruction, finite model gradients, holdout mediator exclusion, planning equivalence, and zero-click billing. No full Amazon MCMC fit or passing Amazon recovery run has been completed. The rebuilt six-page blueprint explicitly identifies the saved result as pre-Amazon.

Generate the optional dataset with `--amazon --generate-only`; the intended full extension run is documented in README.md and must use a new output folder. Once the run passes all gates, generate a separate PDF from its saved results. Do not replace the historical validated outputs or claim Amazon recovery until that fit is complete. Do not commit or push without user confirmation.

## Commands

From the workspace in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest test_mmm -v
& '.\.venv\Scripts\python.exe' '.\build_summary_pdf.py' --results-dir '.\outputs_kalman_profit_validated' --output '.\MMM_Implementation_Blueprint.pdf'
```

Only when intentionally making a new fit, use a new output folder:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --funnel --stockouts --hierarchical-media --brand-state --target-accept 0.99 --output '.\outputs_new_run'
```

The last full fit took about 104 minutes. Use saved posterior_draws.npz to retain exact fitted results; a fresh MCMC run is not guaranteed bit-identical across environments. Python 3.14 was verified; requirements.txt records model dependencies. Backup metadata additionally records the installed package list and Git commit.

## Recovery And Chat

Git ignores outputs*/ and .venv/. A code push is not a results backup. Run backup_checkpoint.ps1 to create a dated sibling recovery folder containing project.zip, raw VS Code workspace chat records, environment details, and SHA-256 checksums. The project ZIP includes all output folders and .git, but excludes .venv and __pycache__. Recreate the environment rather than copying a moved venv.

Raw chat records are VS Code implementation details: retain originals but do not assume copying them restores a supported session or a model's exact working context. To make an official portable export, open each desired chat and use Ctrl+Shift+P > Chat: Export Chat..., then preserve a readable Markdown companion with Copy All. Review history for sensitive content before sharing. Local backup is not off-machine storage: copy the entire recovery folder to a protected cloud/external destination.

## Resume Prompt

> Read CHECKPOINT.md, README.md, and the implementation blueprint. Review the current saved validated results before changing anything. Summarize the model, assumptions, validation status, and unresolved decisions. Do not refit yet. Preserve previous outputs and unrelated user edits.

Next development priorities: validate real-data definitions and economics, build an explicit ingestion/data-validation adapter, time-aware baselines and rolling-origin checks, experimental calibration, and sensitivity of allocations to margins, risk settings, and state-model assumptions. Nothing about synthetic recovery proves real-world causality.