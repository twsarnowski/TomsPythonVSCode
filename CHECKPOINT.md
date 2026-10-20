# MMM Recovery Checkpoint

Checkpoint date: 2026-10-20. This file is the portable handoff; chat archives are supporting history, not executable specifications.

## Start Here

Read README.md and MMM_Implementation_Blueprint_Amazon.pdf, then review outputs_amazon_full/run_config.json, metrics.json, and run_report.md. The Amazon fit completed and passed all recorded gates; do not refit or overwrite it just to resume. The original five-channel reference remains on branch `update-simple-math` and in its preserved output folders.

Current objective: an understandable synthetic MMM workflow for forecasting, attribution, expected-profit allocation, and risk-adjusted allocation, with a blueprint for later real-data adaptation. No real-data ingestion pipeline has been implemented yet.

## Current Model

- 156 weekly periods, four products, three tiers; 130 training weeks and 26 held out. Seed 20261007.
- Bayesian product/tier baseline pooling and positive product-level media shrinkage over seven channels: Display, Video, Social, CTV, Paid Search, Amazon Display, and Amazon Search. Display, Video, Social, CTV, and Amazon Display use adstock and saturation; paid clicks have no adstock. Social uses an S-shaped Hill curve.
- CTV affects sales directly and through a Google branded-search/click funnel. Amazon Search has a separate on-Amazon query/click mediator. Mediator likelihoods use unique training product-weeks; holdout mediators are forecast targets only.
- Amazon Display is an impression channel priced at illustrative $14 CPM. Amazon Search uses query opportunities and spend-dependent Binomial paid clicks at illustrative $2 CPC; realized CPC costs are click-dependent, with zero clicks producing zero charges. These are synthetic assumptions, not Amazon billing or attribution data.
- A Kalman filter updates a stochastic latent logit brand state from training survey scores. Media, sponsorship, and weighted distribution drive its target. Holdout scores do not update the state.
- Previous-week brand score contributes to baseline log sales. Media has separate sales and brand coefficients; sponsorship has no immediate-sales term. There is no separately filtered dynamic sales baseline.
- Sales are censored at supplied inventory caps. Nonlinear sales are not Gaussian Kalman observations; brand-to-sales uses a logistic-normal moment approximation. This is not an exact fully joint latent-state sales likelihood.
- Planning carries adstock and filtered brand state forward. Fixed weekly shares for 13 weeks, fixed product proportions, seven media channels plus sponsorship in brand mode, and fixed future coverage and inventory. Amazon Search budget is a cap/activity input; scenario click charges are sampled and reported separately.
- Three allocations: expected units, expected net contribution, and 50% predictive mean profit plus 50% lower-tail 5% CVaR over 512 scenarios.
- Margins are illustrative product-tier rates, not measured COGS. Taxes, returns, fulfillment/fixed costs, uncertain media prices, replenishment, substitution, and benefits beyond 13 weeks are excluded.

## Amazon Verified Run

The current full workflow is saved in `outputs_amazon_full/` on branch `amazon-full-workflow`. Seed 20261007; 156 weeks; 130 training and 26 holdout weeks; four chains; 1,500 tune and 1,000 draws; target acceptance 0.99. The complete output folder, including posterior draws and derived artifacts, is committed with the branch so the fit can be inspected without refitting.

All recorded gates passed: zero divergences, max R-hat 1.0048, minimum bulk/tail ESS 1468/1538; holdout sales coverage 92.9% and RMSE improvement 89.7%; 7/7 channel truths and 28/28 product-channel truths covered; maximum product media coefficient error 4.8%; Amazon query/click holdout coverage 91.3%/93.3%; exact CPC spend reconciliation; brand score RMSE 0.96 versus 2.61 last-score benchmark with 88.5% coverage; contribution, allocation, and budget gates passed. The 13-week unit allocation lift is 1.87% with known-truth regret 0.00042%. Expected synthetic contribution is about $1.860B baseline/$1.891B optimized; risk allocation CVaR(5%) is about $1.884B versus $1.846B baseline. Synthetic dollars and one matched seed do not establish real-world causality.

The original non-Amazon reference remains preserved in `outputs_kalman_profit_validated/`, with its pre-Amazon blueprint `MMM_Implementation_Blueprint.pdf` and prior report. Do not replace that reference when rerunning either specification.

## GitHub Handoff

The full Amazon workflow is maintained on `amazon-full-workflow`; the prior model remains on `update-simple-math`. Both branches should be pushed. Result folders are Git-ignored by default, so `outputs_amazon_full/` must be force-added deliberately to keep the posterior and outputs with the code. The unrelated Feature Store files in the workspace are not part of this project change.

Earlier output folders and PDFs remain historical references, including the deterministic-brand fit. Source has evolved since those fits: old commands need the matching historic commit to reproduce the old specification. Preserve Read MMM.py, SimpleAdd.py, and user edits.

## Source And Test Status

The `--amazon` workflow completed and all run-level gates passed. The full sampler, Amazon mediator, holdout, brand-state, attribution, and budget results are in `outputs_amazon_full/`; the detailed metrics and caveats are recorded in its `run_report.md` and the new blueprint. The run took about 68 minutes. The original five-channel specification and its prior outputs remain intact on `update-simple-math`.

## Commands

From the workspace in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest test_mmm -v
& '.\.venv\Scripts\python.exe' '.\build_summary_pdf.py' --results-dir '.\outputs_amazon_full' --output '.\MMM_Implementation_Blueprint_Amazon.pdf'
```

Only when intentionally making a new fit, use a new output folder:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --amazon --funnel --stockouts --hierarchical-media --brand-state --target-accept 0.99 --output '.\outputs_new_run'
```

The current saved full fit took about 68 minutes. Review its `posterior_draws.npz` to retain exact fitted results; a fresh MCMC run is not guaranteed bit-identical across environments. Python 3.14 was verified; requirements.txt records model dependencies. Backup metadata additionally records the installed package list and Git commit.

## Recovery And Chat

Git ignores outputs*/ and .venv/. A code push is not a results backup. Run backup_checkpoint.ps1 to create a dated sibling recovery folder containing project.zip, raw VS Code workspace chat records, environment details, and SHA-256 checksums. The project ZIP includes all output folders and .git, but excludes .venv and __pycache__. Recreate the environment rather than copying a moved venv.

Raw chat records are VS Code implementation details: retain originals but do not assume copying them restores a supported session or a model's exact working context. To make an official portable export, open each desired chat and use Ctrl+Shift+P > Chat: Export Chat..., then preserve a readable Markdown companion with Copy All. Review history for sensitive content before sharing. Local backup is not off-machine storage: copy the entire recovery folder to a protected cloud/external destination.

## Resume Prompt

> Read CHECKPOINT.md, README.md, and MMM_Implementation_Blueprint_Amazon.pdf. Review outputs_amazon_full/run_config.json, metrics.json, and run_report.md before changing anything. Summarize the model, assumptions, validation status, and unresolved decisions. Do not refit or overwrite saved results. Preserve the pre-Amazon branch and unrelated user edits.

Next development priorities: validate real-data definitions and economics, build an explicit ingestion/data-validation adapter, time-aware baselines and rolling-origin checks, experimental calibration, and sensitivity of allocations to margins, risk settings, and state-model assumptions. Nothing about synthetic recovery proves real-world causality.