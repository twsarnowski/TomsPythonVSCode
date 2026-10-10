import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.linalg import helmert
from scipy.special import expit

from mmm_data import CHANNELS, PRODUCTS, HILL_POWER, lagged_exposures, prepare_data
from run_mmm import funnel_predictions, sampler_diagnostics, select_samples, truth_samples


ROOT = Path(__file__).resolve().parent


def brand_features(samples, data):
    lagged, valid = lagged_exposures(data["ratio"])
    weights = samples["decay"][:, None, :] ** np.arange(lagged.shape[2])[None, :, None]
    adstock = np.einsum("ptlc,dlc->dptc", lagged, weights)
    adstock /= np.einsum("tl,dlc->dtc", valid, weights)[:, None, :, :]
    powered = adstock ** HILL_POWER
    feature = powered / (powered + samples["half_saturation"][:, None, None, :] ** HILL_POWER)
    if data["funnel"]:
        _, clicks = funnel_predictions(samples, data, feature[:, :, :, 3])
        feature[:, :, :, 4] = clicks / (clicks + samples["half_saturation"][:, None, None, 4])
    return feature


def generate_brand_panel(data, sales_truth, seed):
    features = brand_features(truth_samples(sales_truth), data)[0]
    rows = data["controls"].reshape(len(data["weeks"]), len(PRODUCTS), 3, -1)
    controls = rows[:, :, 0, [1, 6, 7]].transpose(1, 0, 2)
    truth = {
        "intercept": -0.35,
        "product_effect": [-0.20, 0.10, -0.05, 0.15],
        "media_beta": [0.35, 0.50, 0.60, 0.45, 0.20],
        "control_beta": [0.30, 0.10, -0.06],
        "precision": 1000.0,
        "seed": seed,
        "measure": "synthetic continuous survey-style composite index on 0-100; not a respondent proportion",
        "role": "separate outcome; no brand-to-sales feedback or long-run brand state",
    }
    logit = (
        truth["intercept"] + np.asarray(truth["product_effect"])[:, None]
        + features @ truth["media_beta"] + controls @ truth["control_beta"]
    )
    mean = expit(logit)
    rng = np.random.default_rng(seed)
    observed = rng.beta(mean * truth["precision"], (1 - mean) * truth["precision"])
    table = pd.DataFrame({
        "product": np.repeat(PRODUCTS, len(data["weeks"])),
        "week": np.tile(data["weeks"], len(PRODUCTS)),
        "brand_equity_index": observed.reshape(-1) * 100,
        "truth_mean_index": mean.reshape(-1) * 100,
    })
    return table, truth, controls


def build_brand_model(features, controls, observed, train_weeks):
    contrast = helmert(len(PRODUCTS), full=False).T
    with pm.Model(coords={"product": PRODUCTS, "channel": CHANNELS}) as model:
        intercept = pm.Normal("brand_intercept", 0, 1)
        product_sd = pm.HalfNormal("brand_product_sd", 0.5)
        raw = pm.Normal("brand_product_raw", 0, 1, shape=len(PRODUCTS) - 1)
        product_effect = pm.Deterministic("brand_product_effect", pt.dot(contrast, raw) * product_sd, dims="product")
        media_beta = pm.HalfNormal("brand_media_beta", 0.8, dims="channel")
        control_beta = pm.Normal("brand_control_beta", 0, 0.4, shape=3)
        precision = pm.LogNormal("brand_precision", np.log(500), 1)
        mean = pm.Deterministic("brand_mean", pm.math.sigmoid(
            intercept + product_effect[:, None]
            + pt.sum(features * media_beta, axis=2)
            + pt.sum(controls * control_beta, axis=2)
        ))
        pm.Beta(
            "brand_observed", alpha=mean[:, :train_weeks] * precision,
            beta=(1 - mean[:, :train_weeks]) * precision,
            observed=observed[:, :train_weeks],
        )
    return model


def write_report(metrics, output, source):
    sampling, forecast, recovery = metrics["sampling"], metrics["forecast"], metrics["recovery"]
    text = f"""# Brand Equity: Separate Outcome Prototype

Status: {'PASS' if metrics['overall_pass'] else 'NOT ALL GATES PASSED'}.

This optional extension defines a synthetic continuous 0-100 survey-style composite.
Brand equity is not a universally standardized metric: a real application must specify
awareness, consideration, preference, or another measured construct and its survey design.
The index is modeled with a Beta likelihood on its 0-1 fraction and a logistic mean.
There is one brand observation per product-week, not three repeated tier observations.

Media inputs use saved adstock, saturation, and latent-click funnel transformations from
{source.name}. These are posterior-mean feature estimates: this is a conditional plug-in
outcome model, not a fully joint sales/brand model. Feature uncertainty is not propagated
into brand intervals in this first version. Brand media coefficients are separate from
sales coefficients and shared across products; product intercepts use partial pooling.
Macro and annual seasonality are included. All channel signs are constrained positive.

The synthetic brand index is generated with known coefficients on the matching transforms
using a separate random stream. Its likelihood sees only the first 130 weeks by default;
the final 26 are held out. Future media, organic-demand, and control inputs are supplied.
Brand measurements never enter the sales predictor or optimizer, and do not alter its saved
posterior. There is no persistent brand state or equity-to-sales pathway in this stage.

## Verified Results

- MCMC maximum R-hat: {sampling['max_rhat']}; divergences: {sampling['divergences']}.
- Holdout RMSE: {forecast['rmse_index_points']:.3f} index points versus product-mean benchmark
  {forecast['benchmark_rmse_index_points']:.3f}.
- Nominal 90% predictive interval coverage: {forecast['coverage_90']:.1%}.
- Media truths covered: {recovery['truths_covered']}/5; max relative coefficient error:
  {recovery['max_relative_error']:.1%}.

Gates declared before fitting: existing sampler gates; lower RMSE than the product-mean
benchmark; coverage 85%-97%; at least four of five coefficient truths in 90% intervals
and maximum relative error 25%. One controlled synthetic seed is not real-data calibration.

## Interpretation And Next Decisions

Brand coefficients are log-odds effects on this bounded index, not sales coefficients,
currency, lifetime value, or additive percentage-point contributions. Brand outcome can
respond to existing media carryover but this is not a separately persistent brand stock.
The optimizer still maximizes fulfilled units only: brand equity does not enter the
objective, constraints, or revenue model. Joint estimation, measurement-error propagation,
brand-specific carryover, product-level brand-response shrinkage, equity-to-sales mediation,
and multi-objective budget tradeoffs require explicit decisions and further validation.
Do not assign a dollar value or claim a long-term causal brand effect from this example.

![Brand equity holdout forecasts](brand_equity_forecast.png)
"""
    (output / "brand_report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Add a separate synthetic brand-equity outcome using saved MMM transforms.")
    parser.add_argument("--source", type=Path, default=ROOT / "outputs_combined_hierarchical")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_brand_equity")
    parser.add_argument("--seed", type=int, default=20262107)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    protected = {ROOT / name for name in ["outputs", "outputs_stockouts", "outputs_funnel", "outputs_combined", "outputs_combined_hierarchical"]}
    if output == source or output in protected:
        parser.error("Brand outputs must not replace any existing MMM reference run.")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "run_config.json").read_text())
    sales_truth = json.loads((source / "truth_parameters.json").read_text())
    frame = pd.read_csv(source / "synthetic_weekly.csv", parse_dates=["week"])
    data = prepare_data(frame)
    saved = np.load(source / "posterior_draws.npz")
    samples = select_samples({name: saved[name] for name in saved.files}, 512)
    features = brand_features(samples, data).mean(axis=0)
    panel, truth, controls = generate_brand_panel(data, sales_truth, args.seed)
    train_weeks = config["train_weeks"]
    observed = panel.brand_equity_index.to_numpy().reshape(len(PRODUCTS), -1) / 100
    panel.to_csv(output / "brand_equity_weekly.csv", index=False)
    (output / "brand_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    model = build_brand_model(features, controls, observed, train_weeks)
    with model:
        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=args.chains,
            random_seed=args.seed + 1,
            step=pm.NUTS(target_accept=0.95, max_treedepth=12),
        )
    sampling = sampler_diagnostics(model, idata, output)
    mean = idata.posterior["brand_mean"].values.reshape(-1, len(PRODUCTS), len(data["weeks"]))
    precision = idata.posterior["brand_precision"].values.reshape(-1, 1, 1)
    beta_draws = idata.posterior["brand_media_beta"].values.reshape(-1, len(CHANNELS))
    np.savez_compressed(output / "brand_posterior_draws.npz", mean=mean, precision=precision, media_beta=beta_draws)
    rng = np.random.default_rng(args.seed + 2)
    predictive = rng.beta(mean * precision, (1 - mean) * precision) * 100
    lower, upper = np.quantile(predictive, [0.05, 0.95], axis=0)
    expected = mean.mean(axis=0) * 100
    test = slice(train_weeks, len(data["weeks"]))
    error = expected[:, test] - observed[:, test] * 100
    benchmark = observed[:, :train_weeks].mean(axis=1, keepdims=True) * 100
    benchmark_error = benchmark - observed[:, test] * 100
    rmse = float(np.sqrt(np.mean(error**2)))
    benchmark_rmse = float(np.sqrt(np.mean(benchmark_error**2)))
    coverage = float(np.mean((observed[:, test] * 100 >= lower[:, test]) & (observed[:, test] * 100 <= upper[:, test])))
    forecast = {"rmse_index_points": rmse, "benchmark_rmse_index_points": benchmark_rmse, "coverage_90": coverage,
                "pass": bool(rmse < benchmark_rmse and 0.85 <= coverage <= 0.97)}
    table = panel.copy()
    table["split"] = np.tile(np.where(np.arange(len(data["weeks"])) < train_weeks, "train", "holdout"), len(PRODUCTS))
    table["predicted_mean_index"] = expected.reshape(-1)
    table["p05_index"] = lower.reshape(-1)
    table["p95_index"] = upper.reshape(-1)
    table.to_csv(output / "brand_equity_predictions.csv", index=False)
    recovery_rows = []
    for channel, name in enumerate(CHANNELS):
        low, high = np.quantile(beta_draws[:, channel], [0.05, 0.95])
        known = truth["media_beta"][channel]
        recovery_rows.append({
            "channel": name, "truth": known, "estimate": float(beta_draws[:, channel].mean()),
            "p05": float(low), "p95": float(high), "truth_in_90_interval": bool(low <= known <= high),
            "relative_error": float(abs(beta_draws[:, channel].mean() - known) / known),
        })
    recovery_table = pd.DataFrame(recovery_rows)
    recovery_table.to_csv(output / "brand_media_recovery.csv", index=False)
    recovery = {"truths_covered": int(recovery_table.truth_in_90_interval.sum()),
                "max_relative_error": float(recovery_table.relative_error.max())}
    recovery["pass"] = recovery["truths_covered"] >= 4 and recovery["max_relative_error"] <= 0.25
    metrics = {"sampling": sampling, "forecast": forecast, "recovery": recovery,
               "role": "separate conditional outcome; no sales feedback or brand optimization"}
    metrics["overall_pass"] = sampling["pass"] and forecast["pass"] and recovery["pass"]
    (output / "brand_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for product, axis in enumerate(axes.flat):
        axis.plot(data["weeks"], observed[product] * 100, color="#444444", linewidth=1, label="Synthetic index")
        axis.plot(data["weeks"], expected[product], color="#007f73", label="Conditional mean")
        axis.fill_between(data["weeks"], lower[product], upper[product], color="#007f73", alpha=0.18)
        axis.axvline(data["weeks"][train_weeks], color="#b94d28", linestyle="--", label="Holdout begins")
        axis.set(title=PRODUCTS[product], ylabel="Brand index (0-100)", xlabel="Week", ylim=(0, 100))
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "brand_equity_forecast.png", dpi=160)
    plt.close(fig)
    write_report(metrics, output, source)
    print(json.dumps(metrics, indent=2))
    print(f"Brand-equity results saved in {output}; source MMM files were not modified.")


if __name__ == "__main__":
    main()