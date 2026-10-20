# Synthetic MMM Reference Prototype

A reproducible example for understanding sales forecasting, channel attribution,
and budget allocation before adapting the workflow to observational business data.
The existing example scripts and legacy ZIP archive are not changed.

## Run

In PowerShell, from this folder:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py'
```

The existing environment already contains the dependencies. To reproduce them in
another environment, install the versions in [requirements.txt](requirements.txt).
The verified environment is Python 3.14; PyMC and ArviZ APIs differ across versions.

Run the small deterministic tests separately:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest test_mmm -v
```

Generate data without fitting:

```powershell
& '.\.venv\Scripts\python.exe' '.\run_mmm.py' --generate-only
```

An independent later run can use a different seed and a separate output folder:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --seed 20261008 --output '.\outputs_second_seed'
```

The default output folder is `outputs`. Rerunning there replaces that run's artifacts;
choose another folder when preserving a previous run. No automatic dataset tuning
or specification search occurs. Sampling defaults to four chains, 1,500 tuning
steps, and 1,000 retained draws per chain, with target acceptance 0.95 and depth 12.

## Model And Data

[mmm_data.py](mmm_data.py) defines the synthetic generating process and shared
NumPy adstock/Hill functions. [run_mmm.py](run_mmm.py) implements the corresponding
PyMC model, checks, forecasts, attribution, planning, plots, and report.

The panel has 156 weeks, four product groups, three price tiers, and five channels.
The last 26 weeks are excluded from the likelihood and retained for conditional
forecast validation. This initial specification uses a seasonal baseline, not a
latent random walk. All synthetic effects are specified before fitting.

- Display, Video, Social, and CTV have their own estimated geometric adstock decay.
- Paid-search clicks have no carryover. In the funnel variant, spend-dependent clicks are modeled as
  a capture of branded-search opportunities; sales use expected clicks rather than observed future clicks.
- Every channel has estimated diminishing-return saturation. Social uses a fixed
  Hill power of 2 for an S-shaped curve; the others use power 1.
- Adstock uses the current week plus 12 prior weeks, with normalized lag weights
  and explicit normalization at the beginning of the historical series.
- Media coefficients are nonnegative maximum log-sales effects. Positivity is an
  explicit modeling assumption, not proof that all media is beneficial in reality.
- Product, tier, and product-tier intercept effects use zero-sum contrast bases.
  Estimated group scales provide partial pooling without redundant intercepts.
- Controls include relative list price, macro demand, base-price growth, current
  promotion, a nonnegative post-promotion dip, PrimeDays, and annual sine/cosine.
- Log sales have Normal noise, so sales have a log-Normal predictive distribution.
  Rounded synthetic unit counts are modeled as continuous log sales.
- There are no stockouts, funnel mediation, or stochastic cost/reach likelihoods.

## Optional Amazon Channels

The opt-in `--amazon` extension adds distinct `amazon_display` and `amazon_search` channels; the
five-channel reference remains the default and its saved outputs are preserved. Amazon Display is
synthetic impression exposure priced at $14 CPM. Amazon Search generates on-Amazon query
opportunities from independent organic demand and controls; spend-dependent click capture generates
paid clicks, which are the sales feature. Its illustrative CPC is $2, and realized synthetic spend
is exactly paid clicks times CPC, so zero clicks produce zero click charges. Query and click
likelihoods use training product-weeks only; held-out mediators are prediction targets, not model
inputs. Clicks are not individually linked to purchases, and this matched synthetic demonstration
does not establish Amazon incrementality or platform billing behavior. Optimized Search budgets are
spend caps/activity inputs; expected paid-click charges are scenario-dependent and can be below the
allocated cap. Profit allocation exports planned budget and expected CPC charges separately.

Generate the Amazon panel without fitting:

```powershell
& '.\.venv\Scripts\python.exe' '.\run_mmm.py' --amazon --generate-only --output '.\outputs_amazon_data'
```

Run the combined Amazon, funnel, stockout, hierarchical-media, and brand-state specification into
a new folder:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --amazon --funnel --stockouts --hierarchical-media --brand-state --target-accept 0.99 --output '.\outputs_amazon_full'
```

## Completed Amazon Reference Run

The full specification completed in `outputs_amazon_full` on branch `amazon-full-workflow` using
seed 20261007, 156 weeks, 130 training weeks, 26 holdout weeks, four chains, 1,500 tuning steps,
1,000 draws per chain, and target acceptance 0.99. The complete output folder, including posterior
draws, generated data/truth, metrics, diagnostics, forecasts, attribution, optimizer tables, and
plots, is committed with the source. The original five-channel run and its outputs remain available
on `update-simple-math`; this branch adds the Amazon terms and preserves both workflows.

All recorded gates passed: zero divergences, maximum R-hat 1.0048, minimum bulk/tail ESS 1468/1538;
sales holdout coverage 92.9% and RMSE improvement 89.7%; all 7 media truths and 28/28 product media
truths covered by their 90% intervals; Amazon Search opportunity/click coverage 91.3%/93.3%; exact
CPC spend reconciliation; brand holdout coverage 88.5%; and unit/profit/CVaR budget gates passed.
The fitted unit allocation predicts +1.87% units. Synthetic expected contribution is about $1.860B
at baseline and $1.891B optimized; these values rely on illustrative margins and are not business
forecasts or real-world causal validation.

The new [MMM_Implementation_Blueprint_Amazon.pdf](MMM_Implementation_Blueprint_Amazon.pdf) and
`outputs_amazon_full/run_report.md` document this run. The older
[MMM_Implementation_Blueprint.pdf](MMM_Implementation_Blueprint.pdf) remains the pre-Amazon
reference. Rebuild the Amazon blueprint without fitting again:

```powershell
& '.\.venv\Scripts\python.exe' '.\build_summary_pdf.py' --results-dir '.\outputs_amazon_full' --output '.\MMM_Implementation_Blueprint_Amazon.pdf'
```

Resume by reviewing `CHECKPOINT.md`, the Amazon blueprint, and the saved config/metrics/report.
The full fit took about 68 minutes; a fresh MCMC run is not required to inspect the committed results.

Channel activity is independently scheduled across broad low, medium, high, and
off periods. It is deliberately informative, rather than representative of every
real advertising calendar. Sales noise is modest but nonzero. Prices and discounts
also vary exogenously. Signal is designed into the dataset; it is not discovered
by repeatedly changing the generator after viewing the holdout.

## Learning Brief and Implementation Blueprint

[MMM_Implementation_Blueprint.pdf](MMM_Implementation_Blueprint.pdf) keeps the
standalone executive summary on page one and the Bayesian-versus-Ridge comparison on
page two, followed by four practical appendices: data contract and identification;
model blocks, priors, and likelihoods; reproducible workflow and validation; and the
decision layer plus a staged real-data roadmap.

Regenerate from the saved passing run without another model fit:

```powershell
& '.\.venv\Scripts\python.exe' '.\build_summary_pdf.py' --results-dir '.\outputs_kalman_profit_validated' --output '.\MMM_Implementation_Blueprint.pdf'
```

The ReportLab builder reads saved metrics/config and enforces the six-page structure.
Previous one- and two-page PDFs are retained as historical versions. This blueprint is
a summary and implementation starting point, not a certificate of real-data causal
validity; replace synthetic assumptions and margins with approved definitions before use.

## Read The Workflow

1. **Generate:** export weekly observations and `truth_parameters.json`. Truth
   sales columns and parameters exist for evaluation, not as sales-model predictors.
2. **Estimate:** use MCMC to estimate effects from only 130 training weeks. The
   fitted priors express sign, scale, and pooling assumptions, not planted values.
3. **Forecast:** pass held-out media, price, promotions, events, and macro inputs
   through the estimated model. Predictive intervals include parameter uncertainty
   and new sales noise. This is conditional prediction, not forecasting media or
   macro inputs themselves.
4. **Recover effects:** compare fitted coefficients, decay, and saturation with
   known truth. A single simulated panel is an initial check, not a repeated-seed
   proof of interval calibration or causal validity.
5. **Attribute:** calculate baseline expected sales and total media lift. Allocate
   that lift in proportion to channel log effects, so contributions sum exactly.
   This is an explicit accounting convention. Removing channels one at a time in
   a multiplicative model would produce increments that do not generally add up.
6. **Plan:** translate channel spend to exposures using known CPM/CPC mappings,
   retain the final 12 training weeks of carryover, and optimize expected units
   across 13 future weeks under a fixed budget and channel share bounds.
7. **Check the allocation:** apply the fitted recommendation to the known generating
   model and compare it with the best known-truth allocation found under identical
   constraints. The known-truth optimizer is a validation tool, not available on
   real business data.

## Stockout-Only Variant

Run the paired extension separately, leaving `outputs` intact:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --stockouts
```

It writes to `outputs_stockouts` by default. Underlying demand and existing predictors
are unchanged at the same seed; inventory is added using an independent random stream,
product/tier-specific demand quantiles, and temporary supply reductions. Inventory is
set before realized demand shocks are observed. Sales equal the smaller of demand and cap.

The likelihood is right-censored in log units. The workflow exports both latent demand
and fulfilled sales predictions, stockout probabilities, and event calibration bins.
The optimizer and contribution tables use expected fulfilled units under supplied,
fixed inventory caps. Expected capped demand is calculated analytically, not by clipping
the mean. No inventory replenishment, inter-week stock transfer, supply forecasting, or
funnel mediation is fitted. Latent-demand truth and cap-generation parameters are never
sales predictors. This intentionally simple matched extension tests censoring, not the
harder problem of jointly forecasting an unknown supply process.

The stockout holdout reuses the baseline dates and demand draws for a paired comparison;
it is not an independent new real-world validation. Review its generated report and gates
before proceeding to the separate funnel extension.

## Funnel-Only Variant

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --funnel
```

This variant writes to `outputs_funnel`, preserving baseline and stockout artifacts.
Use both flags for the separately validated combined mode described below.

The matched funnel adds CTV-to-search intensity, exogenous observed organic search
demand, and diminishing paid-spend capture of query opportunities. Search volume is
rounded log-Normal around the latent intensity. Click counts are Binomial conditional
on observed searches, and are zero at zero paid spend. The sales driver is latent
intensity times capture, not noisy realized click counts. There is no additional
organic-search-to-sales term in this initial funnel.

Search and click likelihoods use unique product-weeks from training only. Held-out
mediator observations are used only for evaluation. Sales forecasts condition on
future CTV, other media, paid spend, organic index, and nonmedia controls supplied as
planning inputs; the mediators themselves are predicted by the model.

CTV attribution includes direct log lift plus the additional click-related log lift
it induces. Paid search receives the click-related log lift with CTV removed. This
ordered attribution convention avoids double counting but is not the only possible
interaction allocation. Response curves and the optimizer propagate spending changes
through search and clicks rather than assuming a fixed independent click supply.

The nominal $1.50 CPC now defines a paid-activity spend reference; realized effective
CPC varies with search opportunity and capture. Stockout generation, the baseline
independent-click design, and their saved outputs remain available unchanged.

The new report includes mediator predictions, funnel-parameter recovery, direct versus
mediated CTV log lift, and the same forecast/attribution/optimization gates. As before,
the informative exogenous synthetic design demonstrates a workflow; it does not
establish real-data causal mediation. The combined variant is assessed separately.

## Combined Funnel And Stockouts

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --funnel --stockouts
```

This fourth variant writes to `outputs_combined`; the baseline, stockout-only, and
funnel-only result folders are protected from being overwritten by a combined run.
Its generator applies the funnel first, then inventory censoring. Underlying demand
and mediator observations match the uncapped funnel example at the same seed.

The fit jointly uses training-only search and click likelihoods and a censored sales
likelihood. Holdout exports include searches, clicks, latent demand, fulfilled sales,
and stockout probabilities. Attribution retains the direct/mediated CTV convention,
but allocates fulfilled-sales lift under inventory caps. Optimization propagates both
CTV-to-click mediation and capacity limits with fixed supplied future inventory.

The same gates apply; the combined model is not assumed to pass just because each
extension passed separately. Review `outputs_combined/run_report.md` and its
`metrics.json` for the completed run's outcome. This is a paired synthetic extension,
not independent validation of real-world mediation, inventory forecasting, or causality.

## Planning Assumptions

Display/Video/Social/CTV CPMs are $4/$12/$9/$22; paid-search CPC is $1.50. Future
weekly channel shares are constant across the 13-week horizon. Product proportions
are fixed rather than optimized. Baseline spending is the final 26 training weeks'
average, and channel shares may range from 40% to 180% of baseline shares.
Future nonmedia controls are treated as supplied planning assumptions. They are
the same for the fitted and known-truth objectives; future sales are never used
to choose the allocation. The original allocation objective is units; the current workflow also
reports expected-profit and risk-adjusted profit allocations under explicit synthetic margin assumptions.

Each unit, expected-profit, and risk-adjusted allocation uses six SLSQP starts. This is a
best-found constrained solution, not a guarantee of a global optimum, especially with
Social's S-shaped response.
Response curves vary one channel's future spend while holding the others fixed;
their comparison point is zero *future* spend with existing carryover retained.

## Hierarchical Media Responses

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --funnel --stockouts --hierarchical-media
```

This optional extension writes to `outputs_combined_hierarchical`, preserving all
four shared-media reference folders. Other flag combinations also get a distinct
`_hierarchical` output folder. Without this flag, all earlier specifications remain
unchanged.

For each channel, a population coefficient and a learned between-product deviation
scale partially pool four positive product coefficients. Coefficients are population
response times exp(log deviation). Zero-sum log contrasts make the population response
the geometric center of the product responses; noncentered standard-Normal contrasts
avoid a redundant product mean. Deviation scales have HalfNormal(0.3) priors.

The generator uses predeclared product deviations, while exposing the same independent
media schedules, noise, controls, and optional mechanisms. Truth is not fed into priors.
Product responses drive predictions, attribution, mediation, response curves, and
optimization; product spending proportions remain fixed. Decay, saturation, control
coefficients, and the search/capture model remain shared. Price-tier-specific media
slopes are not introduced in this step.

Recovery outputs include 20 product coefficients and five learned hierarchy scales.
Successful recovery is not a demonstration of improved predictive performance versus
no pooling: that requires a matched comparison and prior sensitivity tests, especially
with only four product groups. The original one-page PDF retains the earlier shared
media combined run. [MMM_Hierarchical_Media_One_Page.pdf](MMM_Hierarchical_Media_One_Page.pdf)
documents this optional extension; its detailed report is in
`outputs_combined_hierarchical/run_report.md`.

Regenerate the updated handout without another fit:

```powershell
& '.\.venv\Scripts\python.exe' '.\build_summary_pdf.py' --results-dir '.\outputs_combined_hierarchical' --output '.\MMM_Hierarchical_Media_One_Page.pdf'
```

## Brand Equity: Optional Separate Outcome

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_brand_equity.py'
```

This initial stage adds a synthetic continuous survey-style brand-equity index on
0-100, once per product-week. It writes to `outputs_brand_equity` and reads the
saved `outputs_combined_hierarchical` run without altering its sales model, data,
posterior, attribution, or allocation. The index is a chosen illustrative composite,
not a universally defined brand-equity metric or respondent success proportion.

The conditional Beta outcome model has a logistic mean, separate nonnegative brand
media coefficients, partially pooled product intercepts, macro, and annual seasonality.
It uses the saved MMM adstock/saturation/latent-click features averaged over posterior
draws. This plug-in stage does not propagate feature uncertainty into its intervals
and is not a fully joint multi-response model. Brand response coefficients are shared
across products; the existing sales media-response hierarchy remains unchanged.

Brand holdouts, coefficient recovery, bounded forecasts, posterior draws, and a chart
are exported with `brand_report.md`. Equity does not feed back into sales, does not
have a distinct persistent state, and does not enter the budget objective. Those roles
require an explicit definition of the measurement construct and additional validation.
No dollar value, lifetime-value effect, or long-term causal interpretation is assumed.
The earlier one-page PDFs describe the sales prototypes, not this optional brand stage.

## Persistent Brand Equity With Sales Feedback

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --funnel --stockouts --hierarchical-media --brand-state --target-accept 0.99 --output '.\outputs_brand_state_validated'
```

The successful joint reference run is in `outputs_brand_state_validated`. The first
attempt remains in `outputs_combined_hierarchical_brand_state`: it had 25 divergences
and correctly skipped optimization. The successful rerun used an equivalent centered
brand-intercept parameterization and target acceptance 0.99; generating data and
substantive priors were unchanged. Without an explicit output argument the brand mode
uses the first folder name, so use the command above to reproduce the passing run.
Both runs protect the existing sales variants and the earlier separate
`outputs_brand_equity` experiment.
The separate-outcome script does not implement this sales-feedback specification.

Brand equity is a noisy measured 0-100 score of a persistent latent logit state.
Media response features, sponsorship impressions, and weekly product-level distribution
coverage determine its target. A learned persistence parameter blends target and prior
state. Previous-week score relative to 50 affects baseline log sales, per 10 score points.
All parameters are jointly estimated; holdout score measurements are excluded and are
never substituted into held-out sales predictions.

Sponsorship has its own impression and spend columns with a known synthetic $18 CPM.
It builds brand rather than entering the immediate-sales equation. Distribution is
volume-weighted percentage coverage: the fraction of available store/market volume
whose stores carry the product. The synthetic generator explicitly weights covered
stores by market volume. This operational measure captures where the product is
currently assorted, not SKU breadth; no separate assortment variable was invented.

The logit brand state follows a linear-Gaussian autoregression with process shocks and
logistic-normal score measurement noise. A Kalman filter updates state estimates from
training brand scores and forecasts states without holdout score observations. Sales depend
nonlinearly on the latent brand score and are not treated as Gaussian Kalman measurements;
the fitted sales effect uses a logistic-normal moment approximation. Future distribution,
media, organic demand, and inventory are supplied planning scenarios. This matched synthetic
specification is not evidence that the measurement or state model fits real brand data.

The optimizer includes sponsorship as a sixth funded line, retaining its distinct cost.
It carries filtered brand state and uncertainty forward and simulates how proposed spend
affects future baseline and fulfilled sales. Distribution is fixed because no cost/reach
mapping was specified. Alongside the original 13-week unit objective, the workflow compares
expected net contribution profit and a risk-adjusted objective: 50% expected profit plus
50% lower-tail CVaR at 5%, over 512 predictive scenarios. Posterior parameter draws,
log-Normal sales noise, inventory caps, and brand process shocks contribute to those scenarios.

Profit is a synthetic proxy: fulfilled units times list price after discount times the
assumed product-tier contribution-margin table in `mmm_data.py`, minus media and sponsorship
spend. The assumed margin rates are `[18%, 20%, 16%]`, `[22%, 28%, 19%]`,
`[25%, 30%, 22%]`, and `[20%, 26%, 18%]` for products in the declared order and Core,
Premium, Value tiers. They are not observed costs. Taxes, returns, fulfilment/fixed costs,
uncertain media prices, and value beyond the 13-week horizon are excluded. Replace these
assumptions with validated unit economics before interpreting results as business profit.

The updated stochastic-state and profit run is preserved separately from the earlier
deterministic brand reference:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_mmm.py' --funnel --stockouts --hierarchical-media --brand-state --target-accept 0.99 --output '.\outputs_kalman_profit_validated'
```

This fit passed the recorded gates (zero divergences; R-hat 1.0055). Brand holdout RMSE was
1.26 score points versus 3.64 for last-score carry-forward. Under the synthetic margin table,
expected 13-week net contribution was $948.5m at baseline and $953.6m at the expected-profit
allocation; 5% predictive profit CVaR was $944.3m at baseline and $952.8m at the risk-adjusted
allocation. These are illustrative synthetic dollars, not business forecasts. See
`outputs_kalman_profit_validated/run_report.md` for full results and caveats.

Current brand contribution is booked separately within baseline sales, while the existing
short-run/funnel channel table remains additive. Long-run brand lift is not retrospectively
credited to specific historic media funding without an agreed counterfactual convention.
Exports include `brand_drivers.csv`, `brand_state_predictions.csv`, `brand_state_recovery.csv`,
and a score forecast chart. Earlier PDFs remain preserved.

[MMM_Implementation_Blueprint.pdf](MMM_Implementation_Blueprint.pdf) documents the current
Kalman-state/profit run in a six-page format. Its appendices carry the workflow and
implementation detail above the two-page summary and alternatives section. Regenerate
without refitting:

```powershell
& '.\.venv\Scripts\python.exe' '.\build_summary_pdf.py' --results-dir '.\outputs_kalman_profit_validated' --output '.\MMM_Implementation_Blueprint.pdf'
```

## Gates And Limits

The generated run report records predefined, lightweight gates for sampling,
holdout prediction, parameter/contribution recovery, and known-truth allocation
regret. Failure is reported rather than hidden by automatic data or prior changes.
If sampling fails, the workflow saves diagnostics and forecasts but skips allocation.
If an attribution gate fails, any generated allocation remains an educational
scenario, not a decision-ready recommendation.

Success here demonstrates a coherent reference workflow under matching assumptions.
Real-data use additionally requires dealing with endogenous spend, confounding,
measurement errors, model mismatch, unknown costs, and changing behavior. It may
require experiments or external evidence before channel effects are causal.

After this baseline works, keep it intact as a reference. Add new features in
separate variants and validate them before replacing it.