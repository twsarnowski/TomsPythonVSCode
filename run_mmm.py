import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pymc as pm
from pytensor import scan
import pytensor.tensor as pt
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import ndtr, expit
from scipy.stats import binom

from mmm_data import (
    CHANNELS, PRODUCTS, TIERS, CONTROL_NAMES, MAX_LAG,
    CONTRIBUTION_MARGIN_RATE,
    generate_data, lagged_exposures, media_features, prepare_data, save_data,
    expected_fulfilled_units,
    funnel_search_log_mean, funnel_click_probability,
    persistent_brand_state, kalman_filter_brand_state,
)


ROOT = Path(__file__).resolve().parent
PROFIT_RISK_WEIGHT = 0.5
PROFIT_CVAR_ALPHA = 0.05
PROFIT_SCENARIOS = 512


def build_model(data, train_weeks):
    data["train_weeks"] = train_weeks
    train = np.flatnonzero(data["week"] < train_weeks)
    channels = data["channels"]
    carry_indices = data["carry_indices"]
    product_contrast = helmert(len(PRODUCTS), full=False).T
    tier_contrast = helmert(len(TIERS), full=False).T
    lagged, valid = lagged_exposures(data["ratio"])
    coords = {
        "channel": channels, "carry_channel": [channels[index] for index in carry_indices],
        "product": PRODUCTS, "tier": TIERS, "control": CONTROL_NAMES,
        "observation": np.arange(len(data["frame"])),
        "train_observation": train,
    }
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", np.log(3000), 1.0)
        product_sd = pm.HalfNormal("product_sd", 0.8)
        tier_sd = pm.HalfNormal("tier_sd", 0.6)
        interaction_sd = pm.HalfNormal("interaction_sd", 0.15)
        product_raw = pm.Normal("product_contrast", 0, product_sd, shape=len(PRODUCTS) - 1)
        tier_raw = pm.Normal("tier_contrast", 0, tier_sd, shape=len(TIERS) - 1)
        interaction_raw = pm.Normal(
            "interaction_contrast", 0, interaction_sd,
            shape=(len(PRODUCTS) - 1, len(TIERS) - 1),
        )
        product_effect = pm.Deterministic(
            "product_effect", pt.dot(product_contrast, product_raw), dims="product"
        )
        tier_effect = pm.Deterministic(
            "tier_effect", pt.dot(tier_contrast, tier_raw), dims="tier"
        )
        interaction = pm.Deterministic(
            "interaction", pt.dot(pt.dot(product_contrast, interaction_raw), tier_contrast.T),
            dims=("product", "tier"),
        )
        beta_price = pm.Normal("beta_price", -1.0, 0.7)
        beta_macro = pm.Normal("beta_macro", 0, 1.0)
        beta_base_price = pm.Normal("beta_base_price", 0, 1.0)
        beta_promotion = pm.Normal("beta_promotion", 0, 0.3)
        dip = pm.HalfNormal("post_promotion_effect", 0.15)
        beta_event = pm.Normal("prime_days_beta", 0, 0.4)
        beta_sin = pm.Normal("seasonal_sin_beta", 0, 0.3)
        beta_cos = pm.Normal("seasonal_cos_beta", 0, 0.3)
        control_beta = pm.Deterministic(
            "control_beta",
            pt.stack([beta_price, beta_macro, beta_base_price, beta_promotion, -dip, beta_event, beta_sin, beta_cos]),
            dims="control",
        )
        carry_decay = pm.Beta("carry_decay", 2, 2, dims="carry_channel")
        decay = pm.Deterministic(
            "decay", pt.set_subtensor(pt.zeros(len(channels))[carry_indices], carry_decay),
            dims="channel",
        )
        half = pm.LogNormal("half_saturation", 0, 0.6, dims="channel")
        media_beta = pm.HalfNormal("media_beta", 0.8, dims="channel")
        if data["hierarchical_media"]:
            media_product_sd = pm.HalfNormal("media_product_sd", 0.3, dims="channel")
            media_product_raw = pm.Normal(
                "media_product_raw", 0, 1,
                shape=(len(PRODUCTS) - 1, len(channels)),
            )
            product_media_beta = pm.Deterministic(
                "product_media_beta",
                media_beta[None, :] * pt.exp(pt.dot(product_contrast, media_product_raw) * media_product_sd),
                dims=("product", "channel"),
            )
            observation_media_beta = product_media_beta[data["product"]]
        else:
            observation_media_beta = media_beta
        weights = carry_decay[None, :] ** np.arange(MAX_LAG)[:, None]
        carried = pt.sum(lagged[:, :, :, carry_indices] * weights[None, None, :, :], axis=2)
        carried = carried / pt.dot(valid, weights)[None, :, :]
        adstock = pt.as_tensor_variable(data["ratio"])
        adstock = pt.set_subtensor(adstock[:, :, carry_indices], carried)
        hill_power = data["hill_power"]
        powered = adstock ** hill_power
        features = powered / (powered + half ** hill_power)
        if data["funnel"]:
            search_intercept = pm.Normal(
                "search_intercept", np.log(data["search_counts"][:, :train_weeks]).mean(axis=1),
                0.8, dims="product",
            )
            search_ctv_beta = pm.HalfNormal("search_ctv_beta", 1.0)
            search_control_beta = pm.Normal("search_control_beta", 0, 1, shape=4)
            search_sigma = pm.HalfNormal("search_sigma", 0.2)
            click_max_rate = pm.Beta("click_max_rate", 2, 5)
            click_budget_half = pm.LogNormal("click_budget_half", 0, 0.6)
            search_log_mean = (
                search_intercept[:, None] + search_ctv_beta * features[:, :, 3]
                + pt.sum(data["search_controls"] * search_control_beta, axis=2)
            )
            pm.Normal(
                "log_search", mu=search_log_mean[:, :train_weeks], sigma=search_sigma,
                observed=np.log(data["search_counts"][:, :train_weeks]),
            )
            capture = click_max_rate * data["paid_budget_ratio"] / (
                data["paid_budget_ratio"] + click_budget_half
            )
            active = data["paid_budget_ratio"][:, :train_weeks] > 0
            pm.Binomial(
                "paid_clicks", n=data["search_counts"][:, :train_weeks][active],
                p=capture[:, :train_weeks][active],
                observed=data["click_counts"][:, :train_weeks][active],
            )
            click_ratio = pt.exp(search_log_mean) * capture / data["reference"][:, 4:5]
            click_feature = click_ratio / (click_ratio + half[4])
            features = pt.concatenate([features[:, :, :4], click_feature[:, :, None]], axis=2)
        if data["amazon"]:
            amazon_search_intercept = pm.Normal(
                "amazon_search_intercept",
                np.log(data["amazon_search_counts"][:, :train_weeks]).mean(axis=1),
                0.8, dims="product",
            )
            amazon_search_control_beta = pm.Normal(
                "amazon_search_control_beta", 0, 1, shape=4,
            )
            amazon_search_sigma = pm.HalfNormal("amazon_search_sigma", 0.2)
            amazon_click_max_rate = pm.Beta("amazon_click_max_rate", 2, 5)
            amazon_click_budget_half = pm.LogNormal("amazon_click_budget_half", 0, 0.6)
            amazon_search_log_mean = (
                amazon_search_intercept[:, None]
                + pt.sum(data["amazon_search_controls"] * amazon_search_control_beta, axis=2)
            )
            pm.Normal(
                "amazon_log_search", mu=amazon_search_log_mean[:, :train_weeks],
                sigma=amazon_search_sigma,
                observed=np.log(data["amazon_search_counts"][:, :train_weeks]),
            )
            amazon_capture = (
                amazon_click_max_rate * data["amazon_budget_ratio"]
                / (data["amazon_budget_ratio"] + amazon_click_budget_half)
            )
            amazon_active = data["amazon_budget_ratio"][:, :train_weeks] > 0
            pm.Binomial(
                "amazon_paid_clicks",
                n=data["amazon_search_counts"][:, :train_weeks][amazon_active],
                p=amazon_capture[:, :train_weeks][amazon_active],
                observed=data["amazon_click_counts"][:, :train_weeks][amazon_active],
            )
            amazon_search_index = channels.index("amazon_search")
            amazon_click_ratio = (
                pt.exp(amazon_search_log_mean) * amazon_capture
                / data["reference"][:, amazon_search_index:amazon_search_index + 1]
            )
            amazon_click_feature = amazon_click_ratio / (
                amazon_click_ratio + half[amazon_search_index]
            )
            features = pt.concatenate([
                features[:, :, :amazon_search_index], amazon_click_feature[:, :, None],
                features[:, :, amazon_search_index + 1:],
            ], axis=2)
        base = (
            intercept + product_effect[data["product"]] + tier_effect[data["tier"]]
            + interaction[data["product"], data["tier"]]
            + pt.dot(data["controls"], control_beta)
        )
        if data["brand_state"]:
            brand_intercept = pm.Normal("brand_intercept", 0, 1)
            brand_product_sd = pm.HalfNormal("brand_product_sd", 0.5)
            brand_product_contrast = pm.Normal(
                "brand_product_contrast", 0, brand_product_sd, shape=len(PRODUCTS) - 1
            )
            brand_product_effect = pm.Deterministic(
                "brand_product_effect", pt.dot(product_contrast, brand_product_contrast),
                dims="product",
            )
            brand_initial = pm.Normal("brand_initial", 0, 0.5, dims="product")
            brand_media_beta = pm.HalfNormal("brand_media_beta", 0.8, dims="channel")
            brand_sponsorship_beta = pm.HalfNormal("brand_sponsorship_beta", 0.8)
            brand_distribution_beta = pm.HalfNormal("brand_distribution_beta", 1.0)
            brand_persistence = pm.Beta("brand_persistence", 8, 2)
            brand_process_sigma = pm.HalfNormal("brand_process_sigma", 0.05)
            brand_measurement_sigma = pm.HalfNormal("brand_measurement_sigma", 0.05)
            brand_sales_beta = pm.HalfNormal("brand_sales_beta", 0.2)
            sponsor_feature = data["sponsorship_ratio"] / (1 + data["sponsorship_ratio"])
            target = (
                brand_intercept + brand_product_effect[:, None]
                + pt.sum(features * brand_media_beta, axis=2)
                + brand_sponsorship_beta * sponsor_feature
                + brand_distribution_beta * (data["distribution_fraction"] - 0.5)
            )
            def filter_step(target_week, observed_week, previous_mean, previous_variance,
                            persistence, process_sigma, measurement_sigma):
                predicted_mean = persistence * previous_mean + (1 - persistence) * target_week
                predicted_variance = persistence**2 * previous_variance + process_sigma**2
                innovation_variance = predicted_variance + measurement_sigma**2
                gain = predicted_variance / innovation_variance
                innovation = observed_week - predicted_mean
                filtered_mean = predicted_mean + gain * innovation
                filtered_variance = (1 - gain) * predicted_variance
                return filtered_mean, filtered_variance, predicted_mean, predicted_variance

            filtered_mean, filtered_variance, predicted_mean, predicted_variance = scan(
                filter_step,
                sequences=[
                    pt.as_tensor_variable(target[:, :train_weeks].T),
                    pt.as_tensor_variable(data["brand_observed_logit"][:, :train_weeks].T),
                ],
                outputs_info=[brand_initial, pt.zeros(len(PRODUCTS)), None, None],
                non_sequences=[brand_persistence, brand_process_sigma, brand_measurement_sigma],
                n_steps=train_weeks,
                return_updates=False,
                strict=True,
            )
            pm.Normal(
                "brand_score_measurement", mu=predicted_mean.T,
                sigma=pt.sqrt(predicted_variance.T + brand_measurement_sigma**2),
                observed=data["brand_observed_logit"][:, :train_weeks],
            )

            def forecast_step(target_week, previous_mean, previous_variance, persistence, process_sigma):
                predicted_mean = persistence * previous_mean + (1 - persistence) * target_week
                predicted_variance = persistence**2 * previous_variance + process_sigma**2
                return predicted_mean, predicted_variance

            forecast_mean, forecast_variance = scan(
                forecast_step,
                sequences=[pt.as_tensor_variable(target[:, train_weeks:].T)],
                outputs_info=[filtered_mean[-1], filtered_variance[-1]],
                non_sequences=[brand_persistence, brand_process_sigma],
                n_steps=len(data["weeks"]) - train_weeks,
                return_updates=False,
                strict=True,
            )
            state = pm.Deterministic(
                "brand_state_logit", pt.concatenate([filtered_mean.T, forecast_mean.T], axis=1)
            )
            state_variance = pm.Deterministic(
                "brand_state_variance", pt.concatenate([filtered_variance.T, forecast_variance.T], axis=1)
            )
            previous_state_mean = pt.concatenate([brand_initial[:, None], state[:, :-1]], axis=1)
            previous_state_variance = pt.concatenate(
                [pt.zeros((len(PRODUCTS), 1)), state_variance[:, :-1]], axis=1
            )
            expected_previous_score = pm.math.sigmoid(
                previous_state_mean / pt.sqrt(1 + np.pi * previous_state_variance / 8)
            )
            brand_effect = brand_sales_beta * (100 * expected_previous_score - 50) / 10
            base = base + brand_effect[data["product"], data["week"]]
        mu = pm.Deterministic(
            "log_sales_mean",
            base + pt.sum(features[data["product"], data["week"]] * observation_media_beta, axis=1),
            dims="observation",
        )
        sigma = pm.HalfNormal("sigma", 0.3)
        if data["inventory"] is None:
            pm.Normal(
                "log_sales", mu=mu[train], sigma=sigma,
                observed=data["log_units"][train], dims="train_observation",
            )
        else:
            pm.Censored(
                "log_sales", pm.Normal.dist(mu=mu[train], sigma=sigma),
                lower=None, upper=np.log(data["inventory"][train]),
                observed=data["log_units"][train], dims="train_observation",
            )
    return model


def extract_samples(dataset):
    names = [
        "intercept", "product_effect", "tier_effect", "interaction", "control_beta",
        "media_beta", "decay", "half_saturation", "sigma",
    ]
    result = {}
    if "product_media_beta" in dataset:
        names.extend(["product_media_beta", "media_product_sd"])
    if "brand_persistence" in dataset:
        names.extend([
            "brand_intercept", "brand_product_effect", "brand_initial", "brand_media_beta",
            "brand_sponsorship_beta", "brand_distribution_beta", "brand_persistence",
            "brand_process_sigma", "brand_measurement_sigma", "brand_sales_beta",
        ])
    if "search_intercept" in dataset:
        names.extend([
            "search_intercept", "search_ctv_beta", "search_control_beta", "search_sigma",
            "click_max_rate", "click_budget_half",
        ])
    if "amazon_search_intercept" in dataset:
        names.extend([
            "amazon_search_intercept", "amazon_search_control_beta", "amazon_search_sigma",
            "amazon_click_max_rate", "amazon_click_budget_half",
        ])
    for name in names:
        variable = dataset[name].transpose("chain", "draw", ...)
        result[name] = variable.values.reshape((-1,) + variable.shape[2:])
    return result


def select_samples(samples, count=512):
    indices = np.linspace(0, len(samples["sigma"]) - 1, min(count, len(samples["sigma"])), dtype=int)
    return {name: values[indices] for name, values in samples.items()}


def base_predictions(samples, data, include_brand=True):
    result = (
        samples["intercept"][:, None]
        + samples["product_effect"][:, data["product"]]
        + samples["tier_effect"][:, data["tier"]]
        + samples["interaction"][:, data["product"], data["tier"]]
        + np.einsum("dk,nk->dn", samples["control_beta"], data["controls"])
    )
    if include_brand and data["brand_state"]:
        _, _, brand_effect = brand_predictions(samples, data)
        result += brand_effect[:, data["product"], data["week"]]
    return result


def response_features(samples, data):
    lagged, valid = lagged_exposures(data["ratio"])
    weights = samples["decay"][:, None, :] ** np.arange(MAX_LAG)[None, :, None]
    adstock = np.einsum("ptlc,dlc->dptc", lagged, weights)
    adstock /= np.einsum("tl,dlc->dtc", valid, weights)[:, None, :, :]
    hill_power = data["hill_power"]
    powered = adstock ** hill_power
    feature = powered / (powered + samples["half_saturation"][:, None, None, :] ** hill_power)
    if data["funnel"]:
        _, click_ratio = funnel_predictions(samples, data, feature[:, :, :, 3])
        feature[:, :, :, 4] = click_ratio / (click_ratio + samples["half_saturation"][:, None, None, 4])
    if data["amazon"]:
        _, click_ratio = amazon_search_predictions(samples, data)
        amazon_search_index = data["channels"].index("amazon_search")
        feature[:, :, :, amazon_search_index] = click_ratio / (
            click_ratio + samples["half_saturation"][:, None, None, amazon_search_index]
        )
    return feature


def brand_target(samples, features, sponsor_ratio, distribution):
    return (
        samples["brand_intercept"][:, None, None]
        + samples["brand_product_effect"][:, :, None]
        + np.einsum("dptc,dc->dpt", features, samples["brand_media_beta"])
        + samples["brand_sponsorship_beta"][:, None, None] * sponsor_ratio / (1 + sponsor_ratio)
        + samples["brand_distribution_beta"][:, None, None] * (distribution - 0.5)
    )


def brand_predictions(samples, data, include_variance=False):
    features = response_features(samples, data)
    target = brand_target(samples, features, data["sponsorship_ratio"], data["distribution_fraction"])
    state, variance = kalman_filter_brand_state(
        target, samples["brand_persistence"], samples["brand_initial"],
        samples["brand_process_sigma"], samples["brand_measurement_sigma"],
        data["brand_observed_logit"], data.get("train_weeks", 130),
    )
    score_probability = expit(state / np.sqrt(1 + np.pi * variance / 8))
    previous_mean = np.concatenate([samples["brand_initial"][:, :, None], state[:, :, :-1]], axis=2)
    previous_variance = np.concatenate([np.zeros_like(variance[:, :, :1]), variance[:, :, :-1]], axis=2)
    previous_probability = expit(previous_mean / np.sqrt(1 + np.pi * previous_variance / 8))
    effect = samples["brand_sales_beta"][:, None, None] * (100 * previous_probability - 50) / 10
    result = (state, 100 * score_probability, effect)
    return (*result, variance) if include_variance else result


def funnel_predictions(samples, data, ctv_feature):
    log_search = funnel_search_log_mean(
        ctv_feature, data["search_controls"], samples["search_intercept"],
        samples["search_ctv_beta"], samples["search_control_beta"],
    )
    capture = funnel_click_probability(
        data["paid_budget_ratio"], samples["click_max_rate"], samples["click_budget_half"]
    )
    return log_search, np.exp(log_search) * capture / data["reference"][None, :, 4:5]


def amazon_search_predictions(samples, data):
    log_search = (
        samples["amazon_search_intercept"][:, :, None]
        + np.einsum(
            "dk,ptk->dpt", samples["amazon_search_control_beta"],
            data["amazon_search_controls"],
        )
    )
    capture = funnel_click_probability(
        data["amazon_budget_ratio"], samples["amazon_click_max_rate"],
        samples["amazon_click_budget_half"],
    )
    index = data["channels"].index("amazon_search")
    click_ratio = np.exp(log_search) * capture / data["reference"][None, :, index:index + 1]
    return log_search, click_ratio


def product_media_coefficients(samples):
    if "product_media_beta" in samples:
        return samples["product_media_beta"]
    return np.broadcast_to(
        samples["media_beta"][:, None, :],
        (len(samples["sigma"]), len(PRODUCTS), samples["media_beta"].shape[-1]),
    )


def predictions(samples, data):
    lagged, valid = lagged_exposures(data["ratio"])
    weights = samples["decay"][:, None, :] ** np.arange(MAX_LAG)[None, :, None]
    adstock = np.einsum("ptlc,dlc->dptc", lagged, weights)
    adstock /= np.einsum("tl,dlc->dtc", valid, weights)[:, None, :, :]
    hill_power = data["hill_power"]
    powered = adstock ** hill_power
    feature = powered / (powered + samples["half_saturation"][:, None, None, :] ** hill_power)
    if data["funnel"]:
        _, click_ratio = funnel_predictions(samples, data, feature[:, :, :, 3])
        _, no_ctv_click_ratio = funnel_predictions(samples, data, np.zeros_like(feature[:, :, :, 3]))
        click_feature = click_ratio / (click_ratio + samples["half_saturation"][:, None, None, 4])
        no_ctv_click_feature = no_ctv_click_ratio / (no_ctv_click_ratio + samples["half_saturation"][:, None, None, 4])
        feature[:, :, :, 4] = click_feature
    if data["amazon"]:
        _, amazon_click_ratio = amazon_search_predictions(samples, data)
        amazon_search_index = data["channels"].index("amazon_search")
        feature[:, :, :, amazon_search_index] = amazon_click_ratio / (
            amazon_click_ratio + samples["half_saturation"][:, None, None, amazon_search_index]
        )
    coefficients = product_media_coefficients(samples)[:, data["product"], :]
    effects = feature[:, data["product"], data["week"], :] * coefficients
    if data["funnel"]:
        no_ctv_paid = no_ctv_click_feature[:, data["product"], data["week"]] * coefficients[:, :, 4]
        effects[:, :, 3] += effects[:, :, 4] - no_ctv_paid
        effects[:, :, 4] = no_ctv_paid
    base = base_predictions(samples, data)
    return base + effects.sum(axis=2), base, effects


def sampler_diagnostics(model, idata, output):
    table = az.summary(idata, var_names=[variable.name for variable in model.free_RVs], kind="diagnostics")
    table.to_csv(output / "sampler_diagnostics.csv")
    rhat = table.r_hat.to_numpy(float)
    result = {
        "max_rhat": float(np.max(rhat)) if np.isfinite(rhat).all() else None,
        "min_bulk_ess": float(table.ess_bulk.min()),
        "min_tail_ess": float(table.ess_tail.min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "max_tree_depth": int(idata.sample_stats["tree_depth"].max()),
    }
    result["pass"] = bool(
        result["max_rhat"] is not None and result["max_rhat"] < 1.01
        and result["min_bulk_ess"] >= 400 and result["min_tail_ess"] >= 200
        and result["divergences"] == 0 and result["max_tree_depth"] < 12
    )
    return result


def forecast_outputs(samples, data, train_weeks, output, seed):
    mu, base, effects = predictions(samples, data)
    inventory = data["inventory"]
    latent_expected = np.exp(mu + samples["sigma"][:, None]**2 / 2)
    expected = expected_fulfilled_units(mu, samples["sigma"][:, None], inventory)
    rng = np.random.default_rng(seed)
    latent_predictive = np.exp(rng.normal(mu, samples["sigma"][:, None]))
    predictive = latent_predictive if inventory is None else np.minimum(latent_predictive, inventory)
    lower, upper = np.quantile(predictive, [0.05, 0.95], axis=0)
    frame = data["frame"]
    table = frame[["week", "product", "tier", "units"]].copy()
    table["split"] = np.where(data["week"] < train_weeks, "train", "holdout")
    table["predicted_mean_units"] = expected.mean(axis=0)
    table["predicted_p05_units"] = lower
    table["predicted_p95_units"] = upper
    if inventory is not None:
        latent_lower, latent_upper = np.quantile(latent_predictive, [0.05, 0.95], axis=0)
        table["inventory_available"] = inventory
        table["stockout"] = frame.stockout.to_numpy()
        table["latent_demand_units_truth"] = frame.latent_demand_units.to_numpy()
        table["latent_mean_units"] = latent_expected.mean(axis=0)
        table["latent_p05_units"] = latent_lower
        table["latent_p95_units"] = latent_upper
        table["predicted_stockout_probability"] = ndtr(
            (mu - np.log(inventory)) / samples["sigma"][:, None]
        ).mean(axis=0)
    table.to_csv(output / "sales_predictions.csv", index=False)
    test = data["week"] >= train_weeks
    train = ~test
    benchmark = {}
    for product in range(len(PRODUCTS)):
        for tier in range(len(TIERS)):
            selected = train & (data["product"] == product) & (data["tier"] == tier)
            benchmark[product, tier] = data["log_units"][selected].mean()
    benchmark_test = np.array([benchmark[product, tier] for product, tier in zip(data["product"][test], data["tier"][test])])
    log_prediction = (
        mu.mean(axis=0)
        if inventory is None
        else np.log(expected.mean(axis=0))
    )
    error = log_prediction[test] - data["log_units"][test]
    rmse = float(np.sqrt(np.mean(error**2)))
    benchmark_rmse = float(np.sqrt(np.mean((benchmark_test - data["log_units"][test])**2)))
    coverage = float(np.mean((frame.units.to_numpy()[test] >= lower[test]) & (frame.units.to_numpy()[test] <= upper[test])))
    metrics = {
        "test_rows": int(test.sum()), "test_weeks": len(data["weeks"]) - train_weeks,
        "rmse_log_sales": rmse, "benchmark_rmse_log_sales": benchmark_rmse,
        "rmse_improvement": 1 - rmse / benchmark_rmse,
        "coverage_90": coverage, "bias_log_sales": float(error.mean()),
        "pass": bool(rmse < benchmark_rmse and 0.85 <= coverage <= 0.97),
    }
    if inventory is not None:
        latent_truth = frame.latent_demand_units.to_numpy()
        latent_error = mu[:, test].mean(axis=0) - np.log(latent_truth[test])
        latent_coverage = float(np.mean(
            (latent_truth[test] >= latent_lower[test]) & (latent_truth[test] <= latent_upper[test])
        ))
        event = frame.stockout.to_numpy(dtype=bool)
        probabilities = table.predicted_stockout_probability.to_numpy()
        metrics.update({
            "latent_rmse_log_demand": float(np.sqrt(np.mean(latent_error**2))),
            "latent_coverage_90": latent_coverage,
            "train_stockout_rate": float(event[train].mean()),
            "holdout_stockout_rate": float(event[test].mean()),
            "holdout_predicted_stockout_rate": float(probabilities[test].mean()),
            "stockout_brier": float(np.mean((probabilities[test] - event[test])**2)),
            "stockout_brier_training_rate_benchmark": float(np.mean((event[train].mean() - event[test])**2)),
        })
        metrics["pass"] = bool(
            metrics["pass"] and metrics["latent_rmse_log_demand"] < 0.10
            and 0.85 <= latent_coverage <= 0.97
        )
        calibration = []
        for lower_bin, upper_bin in [(0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]:
            selected = test & (probabilities >= lower_bin) & (probabilities < upper_bin)
            if upper_bin == 1.0:
                selected = test & (probabilities >= lower_bin) & (probabilities <= upper_bin)
            if selected.any():
                calibration.append({
                    "probability_lower": lower_bin, "probability_upper": upper_bin,
                    "n": int(selected.sum()), "mean_predicted": float(probabilities[selected].mean()),
                    "observed_rate": float(event[selected].mean()),
                })
        pd.DataFrame(calibration).to_csv(output / "stockout_calibration.csv", index=False)
    weekly = []
    for week, indices in frame.groupby("week").groups.items():
        indices = np.array(list(indices))
        totals = predictive[:, indices].sum(axis=1)
        weekly.append({
            "week": week, "actual_units": int(frame.iloc[indices].units.sum()),
            "predicted_mean_units": float(expected[:, indices].sum(axis=1).mean()),
            "p05_units": float(np.quantile(totals, 0.05)),
            "p95_units": float(np.quantile(totals, 0.95)),
        })
    pd.DataFrame(weekly).to_csv(output / "weekly_predictions.csv", index=False)
    return metrics, expected, base, effects


def parameter_recovery(samples, truth, output):
    channels = truth.get("channels", CHANNELS)
    rows = []
    for name in ["media_beta", "decay", "half_saturation", "control_beta"]:
        labels = channels if name != "control_beta" else CONTROL_NAMES
        for index, label in enumerate(labels):
            if name == "decay" and label in {"paid_search", "amazon_search"}:
                continue
            values = samples[name][:, index]
            lower, upper = np.quantile(values, [0.05, 0.95])
            planted = truth[name][index]
            rows.append({
                "parameter": name, "label": label, "truth": planted,
                "estimate": float(values.mean()), "p05": float(lower), "p95": float(upper),
                "truth_in_90_interval": bool(lower <= planted <= upper),
                "absolute_error": float(abs(values.mean() - planted)),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "parameter_recovery.csv", index=False)
    media = table[table.parameter == "media_beta"]
    decay = table[table.parameter == "decay"]
    metrics = {
        "media_truths_covered": int(media.truth_in_90_interval.sum()),
        "media_total": len(media),
        "max_media_relative_error": float(np.max(media.absolute_error / media.truth)),
        "max_adstock_absolute_error": float(decay.absolute_error.max()),
    }
    metrics["pass"] = bool(
        metrics["media_truths_covered"] >= 4
        and metrics["max_media_relative_error"] <= 0.25
        and metrics["max_adstock_absolute_error"] <= 0.10
    )
    if "product_media_beta" in samples:
        product_rows = []
        for product, product_name in enumerate(PRODUCTS):
            for channel, channel_name in enumerate(channels):
                draws = samples["product_media_beta"][:, product, channel]
                lower, upper = np.quantile(draws, [0.05, 0.95])
                known = truth["product_media_beta"][product][channel]
                center = samples["media_beta"][:, channel]
                deviation = np.log(draws / center)
                product_rows.append({
                    "product": product_name, "channel": channel_name, "truth": known,
                    "estimate": float(draws.mean()), "p05": float(lower), "p95": float(upper),
                    "population_estimate": float(center.mean()),
                    "log_deviation_mean": float(deviation.mean()),
                    "relative_error": float(abs(draws.mean() - known) / known),
                    "truth_in_90_interval": bool(lower <= known <= upper),
                })
        product_table = pd.DataFrame(product_rows)
        product_table.to_csv(output / "product_media_recovery.csv", index=False)
        scale_rows = []
        for channel, name in enumerate(channels):
            values = samples["media_product_sd"][:, channel]
            lower, upper = np.quantile(values, [0.05, 0.95])
            scale_rows.append({
                "channel": name, "log_deviation_scale_mean": float(values.mean()),
                "p05": float(lower), "p95": float(upper),
            })
        pd.DataFrame(scale_rows).to_csv(output / "media_hierarchy_scales.csv", index=False)
        metrics["product_media_truths_covered"] = int(product_table.truth_in_90_interval.sum())
        metrics["product_media_total"] = len(product_table)
        metrics["max_product_media_relative_error"] = float(product_table.relative_error.max())
        metrics["pass"] = bool(
            metrics["pass"]
            and metrics["product_media_truths_covered"] >= int(np.ceil(0.8 * len(product_table)))
            and metrics["max_product_media_relative_error"] <= 0.25
        )
    return metrics


def funnel_outputs(samples, data, truth, train_weeks, output, seed):
    lagged, valid = lagged_exposures(data["ratio"])
    weights = samples["decay"][:, None, 3] ** np.arange(MAX_LAG)[None, :]
    adstock = np.einsum("ptl,dl->dpt", lagged[:, :, :, 3], weights)
    adstock /= np.einsum("tl,dl->dt", valid, weights)[:, None, :]
    ctv_feature = adstock / (adstock + samples["half_saturation"][:, None, None, 3])
    log_search, _ = funnel_predictions(samples, data, ctv_feature)
    capture = funnel_click_probability(
        data["paid_budget_ratio"], samples["click_max_rate"], samples["click_budget_half"]
    )
    rng = np.random.default_rng(seed)
    search_draws = np.maximum(1, np.rint(
        np.exp(rng.normal(log_search, samples["search_sigma"][:, None, None]))
    )).astype(np.int64)
    click_draws = rng.binomial(search_draws, capture)
    test = slice(train_weeks, len(data["weeks"]))
    rows = []
    metrics = {}
    for name, observed, draws in [
        ("search", data["search_counts"], search_draws),
        ("clicks", data["click_counts"], click_draws),
    ]:
        lower, upper = np.quantile(draws, [0.05, 0.95], axis=0)
        mean = draws.mean(axis=0)
        error = np.log1p(mean[:, test]) - np.log1p(observed[:, test])
        metrics[f"{name}_holdout_log_rmse"] = float(np.sqrt(np.mean(error**2)))
        metrics[f"{name}_holdout_coverage_90"] = float(np.mean(
            (observed[:, test] >= lower[:, test]) & (observed[:, test] <= upper[:, test])
        ))
        for product, product_name in enumerate(PRODUCTS):
            for week, date in enumerate(data["weeks"]):
                rows.append({
                    "mediator": name, "product": product_name, "week": date,
                    "split": "train" if week < train_weeks else "holdout",
                    "observed": int(observed[product, week]), "predicted_mean": float(mean[product, week]),
                    "p05": float(lower[product, week]), "p95": float(upper[product, week]),
                })
    pd.DataFrame(rows).to_csv(output / "mediator_predictions.csv", index=False)
    _, _, effects = predictions(samples, data)
    direct = ctv_feature[:, data["product"], data["week"]] * product_media_coefficients(samples)[:, data["product"], 3]
    mediated = effects[:, :, 3] - direct
    components = data["frame"][["week", "product", "tier"]].copy()
    components["ctv_direct_log_lift"] = direct.mean(axis=0)
    components["ctv_mediated_log_lift"] = mediated.mean(axis=0)
    components["truth_ctv_direct_log_lift"] = data["frame"].truth_ctv_direct_log_effect.to_numpy()
    components["truth_ctv_mediated_log_lift"] = data["frame"].truth_ctv_mediated_log_effect.to_numpy()
    components.to_csv(output / "ctv_mediation.csv", index=False)
    known_mediated = float(components.truth_ctv_mediated_log_lift.sum())
    metrics["ctv_mediated_log_lift_relative_error"] = abs(float(mediated.mean(axis=0).sum()) - known_mediated) / known_mediated
    recovery = []
    for name in ["search_ctv_beta", "click_max_rate", "click_budget_half", "search_sigma"]:
        draws = samples[name]
        lower, upper = np.quantile(draws, [0.05, 0.95])
        known = truth[name]
        error = abs(float(draws.mean()) - known) / known
        metrics[f"{name}_relative_error"] = error
        recovery.append({
            "parameter": name, "truth": known, "estimate": float(draws.mean()),
            "p05": float(lower), "p95": float(upper), "relative_error": error,
            "truth_in_90_interval": bool(lower <= known <= upper),
        })
    pd.DataFrame(recovery).to_csv(output / "funnel_recovery.csv", index=False)
    metrics["pass"] = bool(
        metrics["search_holdout_log_rmse"] < 0.20 and metrics["clicks_holdout_log_rmse"] < 0.20
        and 0.85 <= metrics["search_holdout_coverage_90"] <= 0.97
        and 0.85 <= metrics["clicks_holdout_coverage_90"] <= 0.99
        and metrics["ctv_mediated_log_lift_relative_error"] < 0.25
        and all(metrics[f"{name}_relative_error"] < 0.20 for name in ["search_ctv_beta", "click_max_rate", "click_budget_half"])
    )
    return metrics


def amazon_search_outputs(samples, data, truth, train_weeks, output, seed):
    log_search, _ = amazon_search_predictions(samples, data)
    capture = funnel_click_probability(
        data["amazon_budget_ratio"], samples["amazon_click_max_rate"],
        samples["amazon_click_budget_half"],
    )
    rng = np.random.default_rng(seed)
    search_draws = np.maximum(1, np.rint(
        np.exp(rng.normal(log_search, samples["amazon_search_sigma"][:, None, None]))
    )).astype(np.int64)
    click_draws = rng.binomial(search_draws, capture)
    test = slice(train_weeks, len(data["weeks"]))
    rows = []
    metrics = {}
    for name, observed, draws in [
        ("amazon_search_opportunities", data["amazon_search_counts"], search_draws),
        ("amazon_paid_clicks", data["amazon_click_counts"], click_draws),
    ]:
        lower, upper = np.quantile(draws, [0.05, 0.95], axis=0)
        mean = draws.mean(axis=0)
        error = np.log1p(mean[:, test]) - np.log1p(observed[:, test])
        metrics[f"{name}_holdout_log_rmse"] = float(np.sqrt(np.mean(error**2)))
        metrics[f"{name}_holdout_coverage_90"] = float(np.mean(
            (observed[:, test] >= lower[:, test]) & (observed[:, test] <= upper[:, test])
        ))
        for product, product_name in enumerate(PRODUCTS):
            for week, date in enumerate(data["weeks"]):
                rows.append({
                    "mediator": name, "product": product_name, "week": date,
                    "split": "train" if week < train_weeks else "holdout",
                    "observed": int(observed[product, week]),
                    "predicted_mean": float(mean[product, week]),
                    "p05": float(lower[product, week]), "p95": float(upper[product, week]),
                })
    pd.DataFrame(rows).to_csv(output / "amazon_mediator_predictions.csv", index=False)
    recovery = []
    for name in ["amazon_click_max_rate", "amazon_click_budget_half", "amazon_search_sigma"]:
        draws = samples[name]
        lower, upper = np.quantile(draws, [0.05, 0.95])
        known = truth[name]
        error = abs(float(draws.mean()) - known) / known
        metrics[f"{name}_relative_error"] = error
        recovery.append({
            "parameter": name, "truth": known, "estimate": float(draws.mean()),
            "p05": float(lower), "p95": float(upper), "relative_error": error,
            "truth_in_90_interval": bool(lower <= known <= upper),
        })
    pd.DataFrame(recovery).to_csv(output / "amazon_search_recovery.csv", index=False)
    spend_error = np.max(np.abs(
        data["frame"].amazon_search_spend.to_numpy()
        - data["frame"].amazon_search_exposure.to_numpy() * data["unit_cost"][data["channels"].index("amazon_search")]
    ))
    metrics["cpc_spend_reconciliation_error"] = float(spend_error)
    metrics["pass"] = bool(
        metrics["amazon_search_opportunities_holdout_log_rmse"] < 0.20
        and metrics["amazon_paid_clicks_holdout_log_rmse"] < 0.20
        and 0.85 <= metrics["amazon_search_opportunities_holdout_coverage_90"] <= 0.97
        and 0.85 <= metrics["amazon_paid_clicks_holdout_coverage_90"] <= 0.99
        and all(metrics[f"{name}_relative_error"] < 0.20 for name in [
            "amazon_click_max_rate", "amazon_click_budget_half", "amazon_search_sigma",
        ])
        and spend_error < 1e-8
    )
    return metrics


def brand_state_outputs(samples, data, truth, train_weeks, output, seed):
    state, latent_score, effect, state_variance = brand_predictions(
        samples, data, include_variance=True
    )
    rng = np.random.default_rng(seed)
    latent_draw = rng.normal(state, np.sqrt(state_variance))
    predictive = 100 * expit(rng.normal(
        latent_draw, samples["brand_measurement_sigma"][:, None, None]
    ))
    observed = 100 * expit(data["brand_observed_logit"])
    lower, upper = np.quantile(predictive, [0.05, 0.95], axis=0)
    expected = predictive.mean(axis=0)
    error = expected[:, train_weeks:] - observed[:, train_weeks:]
    benchmark_error = observed[:, train_weeks - 1:train_weeks] - observed[:, train_weeks:]
    rmse = float(np.sqrt(np.mean(error**2)))
    benchmark_rmse = float(np.sqrt(np.mean(benchmark_error**2)))
    coverage = float(np.mean(
        (observed[:, train_weeks:] >= lower[:, train_weeks:]) & (observed[:, train_weeks:] <= upper[:, train_weeks:])
    ))
    rows = []
    for product, name in enumerate(PRODUCTS):
        for week, date in enumerate(data["weeks"]):
            rows.append({
                "product": name, "week": date, "split": "train" if week < train_weeks else "holdout",
                "observed_brand_equity_score": float(observed[product, week]),
                "predicted_score": float(expected[product, week]),
                "p05_score": float(lower[product, week]), "p95_score": float(upper[product, week]),
                "latent_score_mean": float(latent_score[:, product, week].mean()),
                "lagged_brand_sales_log_effect": float(effect[:, product, week].mean()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "brand_state_predictions.csv", index=False)
    drivers = data["frame"][[
        "week", "product", "brand_equity_score", "sponsorship_impressions",
        "sponsorship_spend", "distribution_volume_weighted_pct",
    ]].drop_duplicates(["week", "product"])
    drivers.to_csv(output / "brand_drivers.csv", index=False)
    recovery = []
    for name in ["brand_media_beta", "brand_sponsorship_beta", "brand_distribution_beta", "brand_persistence", "brand_process_sigma", "brand_sales_beta"]:
        values = samples[name]
        if values.ndim == 1:
            values = values[:, None]
        known_values = np.atleast_1d(truth[name])
        labels = truth.get("channels", CHANNELS) if name == "brand_media_beta" else [name]
        for index, known in enumerate(known_values):
            draws = values[:, index]
            low, high = np.quantile(draws, [0.05, 0.95])
            recovery.append({
                "parameter": name, "label": labels[index], "truth": float(known),
                "estimate": float(draws.mean()), "p05": float(low), "p95": float(high),
                "relative_error": float(abs(draws.mean() - known) / known),
                "truth_in_90_interval": bool(low <= known <= high),
            })
    recovery_table = pd.DataFrame(recovery)
    recovery_table.to_csv(output / "brand_state_recovery.csv", index=False)
    persistence_error = abs(float(samples["brand_persistence"].mean()) - truth["brand_persistence"])
    maximum_error = float(recovery_table.relative_error.max())
    result = {
        "holdout_rmse_score_points": rmse,
        "last_score_benchmark_rmse": benchmark_rmse,
        "holdout_coverage_90": coverage,
        "max_response_relative_error": maximum_error,
        "persistence_absolute_error": persistence_error,
        "estimated_persistence": float(samples["brand_persistence"].mean()),
        "estimated_sales_log_effect_per_10_points": float(samples["brand_sales_beta"].mean()),
        "pass": bool(rmse < benchmark_rmse and 0.85 <= coverage <= 0.97 and maximum_error <= 0.25 and persistence_error < 0.10),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for product, axis in enumerate(axes.flat):
        axis.plot(data["weeks"], observed[product], color="#777777", label="Observed score", linewidth=1)
        axis.plot(data["weeks"], expected[product], color="#007f73", label="Predicted score")
        axis.fill_between(data["weeks"], lower[product], upper[product], alpha=0.2, color="#007f73")
        axis.axvline(data["weeks"][train_weeks], color="#b94d28", linestyle="--")
        axis.set(title=PRODUCTS[product], ylabel="Brand equity score (0-100)")
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", labelsize=8)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "brand_state_forecast.png", dpi=160)
    plt.close(fig)
    return result


def truth_samples(truth):
    result = {
        name: np.asarray([truth[name]], dtype=float)
        for name in [
            "intercept", "product_effect", "tier_effect", "interaction", "control_beta",
            "media_beta", "decay", "half_saturation", "sigma",
        ]
    }
    if "search_ctv_beta" in truth:
        for name in ["search_intercept", "search_ctv_beta", "search_control_beta", "search_sigma", "click_max_rate", "click_budget_half"]:
            result[name] = np.asarray([truth[name]], dtype=float)
    if "amazon_search_intercept" in truth:
        for name in [
            "amazon_search_intercept", "amazon_search_control_beta", "amazon_search_sigma",
            "amazon_click_max_rate", "amazon_click_budget_half",
        ]:
            result[name] = np.asarray([truth[name]], dtype=float)
    if "product_media_beta" in truth:
        result["product_media_beta"] = np.asarray([truth["product_media_beta"]], dtype=float)
    if "brand_persistence" in truth:
        for name in [
            "brand_intercept", "brand_product_effect", "brand_initial", "brand_media_beta",
            "brand_sponsorship_beta", "brand_distribution_beta", "brand_persistence",
            "brand_process_sigma", "brand_measurement_sigma", "brand_sales_beta",
        ]:
            result[name] = np.asarray([truth[name]], dtype=float)
    return result


def attribution_outputs(samples, data, truth, expected, base, effects, train_weeks, output):
    inventory = data["inventory"]
    baseline = expected_fulfilled_units(base, samples["sigma"][:, None], inventory)
    increment = expected - baseline
    weights = np.divide(
        effects, effects.sum(axis=2, keepdims=True),
        out=np.zeros_like(effects), where=effects.sum(axis=2, keepdims=True) > 1e-12,
    )
    allocated = increment[:, :, None] * weights
    np.testing.assert_allclose(baseline + allocated.sum(axis=2), expected, atol=1e-7)
    frame = data["frame"]
    truth_baseline = expected_fulfilled_units(frame.truth_base_log_sales.to_numpy(), truth["sigma"], inventory)
    truth_total = (
        frame.truth_expected_units.to_numpy()
        if inventory is None
        else frame.truth_expected_fulfilled_units.to_numpy()
    )
    channels = data["channels"]
    truth_effects = frame[[f"truth_{name}_log_effect" for name in channels]].to_numpy()
    truth_weights = np.divide(
        truth_effects, truth_effects.sum(axis=1, keepdims=True),
        out=np.zeros_like(truth_effects), where=truth_effects.sum(axis=1, keepdims=True) > 1e-12,
    )
    truth_allocated = (truth_total - truth_baseline)[:, None] * truth_weights
    table = frame[["week", "product", "tier"]].copy()
    table["split"] = np.where(data["week"] < train_weeks, "train", "holdout")
    table["baseline_units"] = baseline.mean(axis=0)
    table["modeled_units"] = expected.mean(axis=0)
    if data["brand_state"]:
        without_brand = expected_fulfilled_units(
            base_predictions(samples, data, include_brand=False), samples["sigma"][:, None], inventory
        )
        table["baseline_without_brand_units"] = without_brand.mean(axis=0)
        table["brand_baseline_incremental_units"] = (baseline - without_brand).mean(axis=0)
    for channel, name in enumerate(channels):
        table[f"{name}_allocated_units"] = allocated[:, :, channel].mean(axis=0)
        table[f"{name}_truth_allocated_units"] = truth_allocated[:, channel]
    table.to_csv(output / "unit_decomposition.csv", index=False)
    rows = []
    for split in ["train", "holdout"]:
        selected = table.split.to_numpy() == split
        for channel, name in enumerate(channels):
            draw_totals = allocated[:, selected, channel].sum(axis=1)
            planted = float(truth_allocated[selected, channel].sum())
            mean = float(draw_totals.mean())
            lower, upper = np.quantile(draw_totals, [0.05, 0.95])
            rows.append({
                "split": split, "channel": name, "allocated_units_mean": mean,
                "p05": float(lower), "p95": float(upper), "truth_allocated_units": planted,
                "relative_error": abs(mean - planted) / planted,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "channel_contributions.csv", index=False)
    maximum_error = float(summary.relative_error.max())
    return {"max_channel_contribution_relative_error": maximum_error, "pass": maximum_error <= 0.25}


def planning_context(samples, data, train_weeks, horizon=13,
                     sample_state_uncertainty=False, seed=601):
    data["train_weeks"] = train_weeks
    rng = np.random.default_rng(seed)
    selected = (data["week"] >= train_weeks) & (data["week"] < train_weeks + horizon)
    history = data["ratio"][:, train_weeks - MAX_LAG + 1:train_weeks, :]
    padded = np.concatenate([history, np.zeros((len(PRODUCTS), horizon, len(data["channels"])))], axis=1)
    lagged, valid = lagged_exposures(padded)
    lagged = lagged[:, -horizon:, :, :]
    weights = samples["decay"][:, None, :] ** np.arange(MAX_LAG)[None, :, None]
    weights /= weights.sum(axis=1, keepdims=True)
    historical = np.einsum("ptlc,dlc->dptc", lagged, weights)
    future_fraction = np.cumsum(weights, axis=1)[:, :horizon, :]
    result = {
        "base": base_predictions(samples, data, include_brand=False)[:, selected],
        "historical": historical, "future_fraction": future_fraction,
        "product": data["product"][selected], "week": data["week"][selected] - train_weeks,
        "samples": samples, "horizon": horizon,
        "contribution_per_unit": data["contribution_per_unit"][selected],
        "demand_standard_normal": rng.normal(size=(len(samples["sigma"]), int(selected.sum()))),
        "inventory": data["inventory"][selected] if data["inventory"] is not None else None,
        "funnel": data["funnel"],
        "search_controls": data["search_controls"][:, train_weeks:train_weeks + horizon, :] if data["funnel"] else None,
        "amazon": data["amazon"],
        "amazon_search_controls": data["amazon_search_controls"][:, train_weeks:train_weeks + horizon, :] if data["amazon"] else None,
        "reference": data["reference"],
        "unit_cost": data["unit_cost"],
        "channels": data["channels"],
        "hill_power": data["hill_power"],
        "brand_state": data["brand_state"],
    }
    if data["amazon"]:
        result["amazon_search_standard_normal"] = rng.normal(
            size=(len(samples["sigma"]), len(PRODUCTS), horizon)
        )
        result["amazon_click_uniform"] = rng.uniform(
            size=(len(samples["sigma"]), len(PRODUCTS), horizon)
        )
    if data["brand_state"]:
        state, _, _, state_variance = brand_predictions(
            samples, data, include_variance=True
        )
        result["brand_initial"] = state[:, :, train_weeks - 1]
        result["brand_initial_variance"] = state_variance[:, :, train_weeks - 1]
        result["distribution_fraction"] = data["distribution_fraction"][:, train_weeks:train_weeks + horizon]
        if sample_state_uncertainty:
            result["brand_initial"] += np.sqrt(result["brand_initial_variance"]) * rng.normal(
                size=result["brand_initial"].shape
            )
            result["brand_process_innovations"] = rng.normal(
                0, samples["brand_process_sigma"][:, None, None],
                size=(len(samples["sigma"]), len(PRODUCTS), horizon),
            )
    return result


def scenario_log_means(context, exposure_ratio):
    samples = context["samples"]
    channel_count = len(context["channels"])
    adstock = context["historical"] + context["future_fraction"][:, None, :, :] * exposure_ratio[None, None, None, :channel_count]
    hill_power = context["hill_power"]
    powered = adstock ** hill_power
    features = powered / (powered + samples["half_saturation"][:, None, None, :] ** hill_power)
    if context["funnel"]:
        log_search = funnel_search_log_mean(
            features[:, :, :, 3], context["search_controls"], samples["search_intercept"],
            samples["search_ctv_beta"], samples["search_control_beta"],
        )
        capture = funnel_click_probability(
            np.full(features.shape[1:3], exposure_ratio[4]),
            samples["click_max_rate"], samples["click_budget_half"],
        )
        click_ratio = np.exp(log_search) * capture / context["reference"][None, :, 4:5]
        features[:, :, :, 4] = click_ratio / (click_ratio + samples["half_saturation"][:, None, None, 4])
    if context["amazon"]:
        amazon_search_index = context["channels"].index("amazon_search")
        log_search = (
            samples["amazon_search_intercept"][:, :, None]
            + np.einsum(
                "dk,ptk->dpt", samples["amazon_search_control_beta"],
                context["amazon_search_controls"],
            )
        )
        capture = funnel_click_probability(
            np.full(features.shape[1:3], exposure_ratio[amazon_search_index]),
            samples["amazon_click_max_rate"], samples["amazon_click_budget_half"],
        )
        click_ratio = np.exp(log_search) * capture / context["reference"][
            None, :, amazon_search_index:amazon_search_index + 1
        ]
        features[:, :, :, amazon_search_index] = click_ratio / (
            click_ratio + samples["half_saturation"][:, None, None, amazon_search_index]
        )
    media = (features * product_media_coefficients(samples)[:, :, None, :]).sum(axis=3)
    mu = context["base"] + media[:, context["product"], context["week"]]
    if context["brand_state"]:
        sponsor_ratio = np.full(features.shape[1:3], exposure_ratio[channel_count])
        target = brand_target(samples, features, sponsor_ratio, context["distribution_fraction"])
        innovations = context.get("brand_process_innovations")
        state = persistent_brand_state(
            target, samples["brand_persistence"], context["brand_initial"], innovations
        )
        previous_mean = np.concatenate([context["brand_initial"][:, :, None], state[:, :, :-1]], axis=2)
        if innovations is None:
            state_variance = np.empty_like(state)
            previous_variance = context["brand_initial_variance"]
            for week in range(context["horizon"]):
                previous_variance = (
                    samples["brand_persistence"][:, None] ** 2 * previous_variance
                    + samples["brand_process_sigma"][:, None] ** 2
                )
                state_variance[:, :, week] = previous_variance
            previous_state_variance = np.concatenate(
                [context["brand_initial_variance"][:, :, None], state_variance[:, :, :-1]], axis=2
            )
            previous_probability = expit(
                previous_mean / np.sqrt(1 + np.pi * previous_state_variance / 8)
            )
        else:
            previous_probability = expit(previous_mean)
        effect = samples["brand_sales_beta"][:, None, None] * (100 * previous_probability - 50) / 10
        mu += effect[:, context["product"], context["week"]]
    return mu


def scenario_units(context, exposure_ratio):
    return scenario_fulfilled_units(context, exposure_ratio).sum(axis=1)


def scenario_fulfilled_units(context, exposure_ratio):
    mu = scenario_log_means(context, exposure_ratio)
    samples = context["samples"]
    return expected_fulfilled_units(mu, samples["sigma"][:, None], context["inventory"])


def scenario_amazon_search_spend(context, exposure_ratio):
    samples = context["samples"]
    index = context["channels"].index("amazon_search")
    search_log_mean = (
        samples["amazon_search_intercept"][:, :, None]
        + np.einsum(
            "dk,ptk->dpt", samples["amazon_search_control_beta"],
            context["amazon_search_controls"],
        )
    )
    capture = funnel_click_probability(
        np.full(search_log_mean.shape[1:], exposure_ratio[index]),
        samples["amazon_click_max_rate"], samples["amazon_click_budget_half"],
    )
    search_counts = np.maximum(1, np.rint(np.exp(
        search_log_mean
        + samples["amazon_search_sigma"][:, None, None]
        * context["amazon_search_standard_normal"]
    ))).astype(np.int64)
    paid_clicks = binom.ppf(
        context["amazon_click_uniform"], search_counts, capture
    ).astype(np.int64)
    return paid_clicks.sum(axis=(1, 2)) * context["unit_cost"][index]


def scenario_media_spend(context, exposure_ratio, cost_reference):
    samples = context["samples"]
    planned_spend = float(np.sum(exposure_ratio * cost_reference) * context["horizon"])
    if context["amazon"]:
        index = context["channels"].index("amazon_search")
        planned_search_spend = exposure_ratio[index] * cost_reference[index] * context["horizon"]
        planned_spend += scenario_amazon_search_spend(context, exposure_ratio) - planned_search_spend
        return planned_spend
    return np.full(len(samples["sigma"]), planned_spend)


def scenario_expected_profit_draws(context, exposure_ratio, budget_week, cost_reference):
    fulfilled = scenario_fulfilled_units(context, exposure_ratio)
    contribution = (fulfilled * context["contribution_per_unit"][None, :]).sum(axis=1)
    spend = scenario_media_spend(context, exposure_ratio, cost_reference)
    return contribution - spend


def scenario_profit_draws(context, exposure_ratio, budget_week, cost_reference):
    samples = context["samples"]
    mu = scenario_log_means(context, exposure_ratio)
    demand = np.exp(mu + samples["sigma"][:, None] * context["demand_standard_normal"])
    fulfilled = demand
    if context["inventory"] is not None:
        fulfilled = np.minimum(fulfilled, context["inventory"])
    contribution = (fulfilled * context["contribution_per_unit"][None, :]).sum(axis=1)
    spend = scenario_media_spend(context, exposure_ratio, cost_reference)
    return contribution - spend


def lower_tail_cvar(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    tail_count = max(1, int(np.ceil(alpha * len(values))))
    return float(np.partition(values, tail_count - 1)[:tail_count].mean())


def solve_allocation(context, budget_week, cost_reference, baseline_share, lower, upper,
                     objective_name="units", risk_weight=PROFIT_RISK_WEIGHT,
                     cvar_alpha=PROFIT_CVAR_ALPHA):
    def score(share):
        ratio = share * budget_week / cost_reference
        if objective_name == "units":
            return float(scenario_units(context, ratio).mean())
        if objective_name == "expected_profit":
            return float(scenario_expected_profit_draws(
                context, ratio, budget_week, cost_reference
            ).mean())
        if objective_name == "risk_adjusted_profit":
            profit = scenario_profit_draws(context, ratio, budget_week, cost_reference)
            mean_profit = float(profit.mean())
            downside_profit = lower_tail_cvar(profit, cvar_alpha)
            return (1 - risk_weight) * mean_profit + risk_weight * downside_profit
        raise ValueError(f"Unknown allocation objective: {objective_name}")

    baseline_score = score(baseline_share)
    scale = max(abs(baseline_score), 1.0)

    def objective(share):
        return -score(share) / scale
    rng = np.random.default_rng(601)
    starts = [baseline_share]
    starts.extend(rng.dirichlet(25 * baseline_share, size=5))
    solutions = []
    for start in starts:
        result = minimize(
            objective, start, method="SLSQP", bounds=list(zip(lower, upper)),
            constraints=[{"type": "eq", "fun": lambda share: share.sum() - 1}],
            options={"maxiter": 150, "ftol": 1e-9},
        )
        feasible = (
            abs(result.x.sum() - 1) < 1e-7 and np.all(result.x >= lower - 1e-7)
            and np.all(result.x <= upper + 1e-7)
        )
        if result.success and feasible:
            solutions.append(result)
    if not solutions:
        raise RuntimeError("No successful feasible budget allocation; no recommendation exported.")
    best = min(solutions, key=lambda result: result.fun)
    return best.x


def profit_optimization_outputs(samples, data, train_weeks, output, horizon,
                                cost_reference, budget_week, baseline_share, lower, upper):
    draw_count = PROFIT_SCENARIOS
    if len(samples["sigma"]) >= draw_count:
        finance_samples = select_samples(samples, draw_count)
    else:
        rng = np.random.default_rng(20261010)
        indices = rng.integers(0, len(samples["sigma"]), size=draw_count)
        finance_samples = {name: values[indices] for name, values in samples.items()}
    context = planning_context(
        finance_samples, data, train_weeks, horizon,
        sample_state_uncertainty=data["brand_state"], seed=20261011,
    )
    expected_share = solve_allocation(
        context, budget_week, cost_reference, baseline_share, lower, upper,
        objective_name="expected_profit",
    )
    risk_weight = PROFIT_RISK_WEIGHT
    cvar_alpha = PROFIT_CVAR_ALPHA
    risk_adjusted_share = solve_allocation(
        context, budget_week, cost_reference, baseline_share, lower, upper,
        objective_name="risk_adjusted_profit", risk_weight=risk_weight, cvar_alpha=cvar_alpha,
    )
    ratios = {
        "baseline": baseline_share * budget_week / cost_reference,
        "expected_profit": expected_share * budget_week / cost_reference,
        "risk_adjusted": risk_adjusted_share * budget_week / cost_reference,
    }
    charged_spend = {
        name: ratio * cost_reference * horizon for name, ratio in ratios.items()
    }
    if data["amazon"]:
        amazon_search_index = data["channels"].index("amazon_search")
        for name, ratio in ratios.items():
            charged_spend[name] = charged_spend[name].copy()
            charged_spend[name][amazon_search_index] = scenario_amazon_search_spend(
                context, ratio
            ).mean()
    expected_profit = {
        name: scenario_expected_profit_draws(context, ratio, budget_week, cost_reference)
        for name, ratio in ratios.items()
    }
    predictive_profit = {
        name: scenario_profit_draws(context, ratio, budget_week, cost_reference)
        for name, ratio in ratios.items()
    }
    risk_score = {
        name: (1 - risk_weight) * float(draws.mean())
        + risk_weight * lower_tail_cvar(draws, cvar_alpha)
        for name, draws in predictive_profit.items()
    }
    table = pd.DataFrame({
        "channel": list(data["channels"]) + (["sponsorship"] if data["brand_state"] else []),
        "baseline_share": baseline_share,
        "expected_profit_share": expected_share,
        "risk_adjusted_share": risk_adjusted_share,
        "baseline_13week_spend": baseline_share * budget_week * horizon,
        "expected_profit_13week_spend": expected_share * budget_week * horizon,
        "risk_adjusted_13week_spend": risk_adjusted_share * budget_week * horizon,
        "baseline_expected_charged_spend_13week": charged_spend["baseline"],
        "expected_profit_expected_charged_spend_13week": charged_spend["expected_profit"],
        "risk_adjusted_expected_charged_spend_13week": charged_spend["risk_adjusted"],
        "lower_share_bound": lower,
        "upper_share_bound": upper,
    })
    table.to_csv(output / "profit_budget_allocation.csv", index=False)
    budget_error = max(
        abs(float(table.expected_profit_13week_spend.sum()) - budget_week * horizon),
        abs(float(table.risk_adjusted_13week_spend.sum()) - budget_week * horizon),
    )
    return {
        "expected_profit_baseline": float(expected_profit["baseline"].mean()),
        "expected_profit_optimized": float(expected_profit["expected_profit"].mean()),
        "expected_profit_at_risk_allocation": float(expected_profit["risk_adjusted"].mean()),
        "predictive_profit_cvar_5_baseline": lower_tail_cvar(predictive_profit["baseline"], cvar_alpha),
        "predictive_profit_cvar_5_expected_allocation": lower_tail_cvar(predictive_profit["expected_profit"], cvar_alpha),
        "predictive_profit_cvar_5_risk_allocation": lower_tail_cvar(predictive_profit["risk_adjusted"], cvar_alpha),
        "risk_adjusted_score_baseline": risk_score["baseline"],
        "risk_adjusted_score_optimized": risk_score["risk_adjusted"],
        "risk_weight_on_cvar": risk_weight,
        "cvar_tail_probability": cvar_alpha,
        "predictive_draws": draw_count,
        "budget_absolute_error": budget_error,
        "contribution_margin_rates_by_product_tier": data["contribution_margin_rate"].tolist(),
        "profit_definition": "fulfilled units x observed net unit price x assumed contribution-margin rate - media/sponsorship spend",
        "risk_definition": "50% expected net profit + 50% lower-tail CVaR at 5% over posterior-predictive sales and brand-process scenarios",
        "pass": bool(
            expected_profit["expected_profit"].mean() >= expected_profit["baseline"].mean() - 1e-6
            and risk_score["risk_adjusted"] >= risk_score["baseline"] - 1e-6
            and budget_error < 0.01
        ),
    }


def optimization_outputs(samples, data, truth, train_weeks, output):
    horizon = 13
    posterior = planning_context(select_samples(samples, 128), data, train_weeks, horizon)
    oracle = planning_context(truth_samples(truth), data, train_weeks, horizon)
    training_ratio = data["ratio"][:, train_weeks - 26:train_weeks, :].mean(axis=1)
    cost_reference = (data["reference"] * data["unit_cost"]).sum(axis=0)
    baseline_weekly_spend = (training_ratio * data["reference"] * data["unit_cost"]).sum(axis=0)
    if data["amazon"]:
        amazon_search_index = data["channels"].index("amazon_search")
        baseline_weekly_spend[amazon_search_index] = data["amazon_search_spend"][:, train_weeks - 26:train_weeks].sum(axis=0).mean()
        mean_budget_activity = data["amazon_budget_ratio"][:, train_weeks - 26:train_weeks].mean()
        cost_reference[amazon_search_index] = baseline_weekly_spend[amazon_search_index] / mean_budget_activity
    allocation_channels = list(data["channels"])
    if data["brand_state"]:
        sponsorship_cost = data["sponsorship_reference"] * truth["sponsorship_cpm"] / 1000
        cost_reference = np.append(cost_reference, sponsorship_cost.sum())
        sponsor_ratio = data["sponsorship_ratio"][:, train_weeks - 26:train_weeks].mean(axis=1)
        baseline_weekly_spend = np.append(baseline_weekly_spend, np.sum(sponsor_ratio * sponsorship_cost))
        allocation_channels.append("sponsorship")
    budget_week = float(baseline_weekly_spend.sum())
    baseline_share = baseline_weekly_spend / budget_week
    lower = 0.4 * baseline_share
    upper = np.minimum(1.8 * baseline_share, 1.0)
    fitted_share = solve_allocation(posterior, budget_week, cost_reference, baseline_share, lower, upper)
    truth_share = solve_allocation(oracle, budget_week, cost_reference, baseline_share, lower, upper)
    fitted_units = scenario_units(posterior, fitted_share * budget_week / cost_reference)
    baseline_units = scenario_units(posterior, baseline_share * budget_week / cost_reference)
    truth_at_fitted = float(scenario_units(oracle, fitted_share * budget_week / cost_reference)[0])
    truth_at_oracle = float(scenario_units(oracle, truth_share * budget_week / cost_reference)[0])
    truth_at_baseline = float(scenario_units(oracle, baseline_share * budget_week / cost_reference)[0])
    allocation = pd.DataFrame({
        "channel": allocation_channels, "baseline_share": baseline_share,
        "fitted_share": fitted_share, "truth_best_found_share": truth_share,
        "baseline_13week_spend": baseline_share * budget_week * horizon,
        "fitted_13week_spend": fitted_share * budget_week * horizon,
        "truth_best_found_13week_spend": truth_share * budget_week * horizon,
        "lower_share_bound": lower, "upper_share_bound": upper,
    })
    allocation.to_csv(output / "budget_allocation.csv", index=False)
    curves = []
    for channel, name in enumerate(allocation_channels):
        zero_spend = baseline_weekly_spend.copy()
        zero_spend[channel] = 0
        reference_prediction = scenario_units(posterior, zero_spend / cost_reference)
        reference_truth = float(scenario_units(oracle, zero_spend / cost_reference)[0])
        for multiplier in np.linspace(0, 2.0, 21):
            spending = baseline_weekly_spend.copy()
            spending[channel] *= multiplier
            draw_units = scenario_units(posterior, spending / cost_reference)
            true_units = float(scenario_units(oracle, spending / cost_reference)[0])
            lift = draw_units - reference_prediction
            lower_lift, upper_lift = np.quantile(lift, [0.05, 0.95])
            curves.append({
                "channel": name, "spend_multiplier": multiplier,
                "channel_13week_spend": spending[channel] * horizon,
                "incremental_units_mean": float(lift.mean()),
                "incremental_units_p05": float(lower_lift), "incremental_units_p95": float(upper_lift),
                "truth_incremental_units": true_units - reference_truth,
            })
    pd.DataFrame(curves).to_csv(output / "response_curves.csv", index=False)
    regret = max(0.0, (truth_at_oracle - truth_at_fitted) / truth_at_oracle)
    budget_error = abs(allocation.fitted_13week_spend.sum() - budget_week * horizon)
    result = {
        "horizon_weeks": horizon, "total_budget": budget_week * horizon,
        "predicted_baseline_units": float(baseline_units.mean()),
        "predicted_optimized_units": float(fitted_units.mean()),
        "predicted_lift": float(fitted_units.mean() / baseline_units.mean() - 1),
        "truth_units_at_baseline": truth_at_baseline,
        "truth_units_at_fitted_allocation": truth_at_fitted,
        "truth_best_found_units": truth_at_oracle,
        "truth_relative_regret": regret, "budget_absolute_error": float(budget_error),
        "pass": bool(regret <= 0.02 and budget_error < 0.01 and truth_at_fitted >= truth_at_baseline - 1),
        "scope": "constant weekly channel shares; fixed product proportions; best of six local starts, not a global-optimality proof",
        "objective": "expected fulfilled units" if data["inventory"] is not None else "expected demand units",
    }
    result["profit_optimization"] = profit_optimization_outputs(
        samples, data, train_weeks, output, horizon, cost_reference,
        budget_week, baseline_share, lower, upper,
    )
    result["pass"] = bool(result["pass"] and result["profit_optimization"]["pass"])
    return result


def write_charts(output, train_weeks):
    weekly = pd.read_csv(output / "weekly_predictions.csv", parse_dates=["week"])
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(weekly.week, weekly.actual_units, color="#222222", linewidth=1.2, label="Observed synthetic sales")
    axis.plot(weekly.week, weekly.predicted_mean_units, color="#007f73", linewidth=1.3, label="Model expected sales")
    axis.fill_between(weekly.week, weekly.p05_units, weekly.p95_units, color="#007f73", alpha=0.18, label="90% predictive interval")
    axis.axvline(weekly.week.iloc[train_weeks], color="#b94d28", linestyle="--", label="Holdout begins")
    axis.set(title="Sales predictions: training and untouched holdout", ylabel="Weekly units", xlabel="Week")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "sales_forecast.png", dpi=160)
    plt.close(fig)
    curves = pd.read_csv(output / "response_curves.csv")
    curve_channels = curves.channel.unique().tolist()
    rows = int(np.ceil(len(curve_channels) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows), squeeze=False)
    for axis, name in zip(axes.flat, curve_channels):
        curve = curves[curves.channel == name]
        spend = curve.channel_13week_spend / 1000
        axis.plot(spend, curve.incremental_units_mean, color="#007f73", label="Fitted")
        axis.plot(spend, curve.truth_incremental_units, color="#b94d28", linestyle="--", label="Known truth")
        axis.fill_between(spend, curve.incremental_units_p05, curve.incremental_units_p95, color="#007f73", alpha=0.18)
        axis.set(title=name.replace("_", " ").title(), xlabel="13-week spend ($ thousands)", ylabel="Incremental units")
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(curve_channels):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "channel_response_curves.png", dpi=160)
    plt.close(fig)
    allocation = pd.read_csv(output / "budget_allocation.csv")
    fig, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(allocation))
    for shift, column, label, color in [
        (-0.25, "baseline_share", "Baseline", "#777777"),
        (0, "fitted_share", "Fitted allocation", "#007f73"),
        (0.25, "truth_best_found_share", "Known-truth best found", "#b94d28"),
    ]:
        axis.bar(positions + shift, allocation[column] * 100, width=0.25, label=label, color=color)
    axis.set_xticks(positions, [name.replace("_", " ").title() for name in allocation.channel])
    axis.set(title="Same budget and constraints: fitted versus known-truth allocation", ylabel="Budget share (%)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "budget_allocation.png", dpi=160)
    plt.close(fig)
    profit_allocation = pd.read_csv(output / "profit_budget_allocation.csv")
    fig, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(profit_allocation))
    for shift, column, label, color in [
        (-0.24, "baseline_share", "Baseline", "#777777"),
        (0, "expected_profit_share", "Expected profit", "#007f73"),
        (0.24, "risk_adjusted_share", "50% mean / 50% 5% CVaR", "#b94d28"),
    ]:
        axis.bar(positions + shift, profit_allocation[column] * 100, width=0.24, label=label, color=color)
    axis.set_xticks(positions, [name.replace("_", " ").title() for name in profit_allocation.channel])
    axis.set(title="Fixed budget: baseline, expected-profit, and risk-adjusted shares", ylabel="Budget share (%)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "profit_budget_allocation.png", dpi=160)
    plt.close(fig)


def write_report(results, args, output):
    sampling = results["sampling"]
    forecast = results["forecast"]
    recovery = results["recovery"]
    attribution = results.get("attribution", {})
    optimization = results.get("optimization", {})
    status = "PASS" if results["overall_pass"] else "NOT ALL GATES PASSED"
    text = f"""# Synthetic MMM Reference Run

Status: **{status}**. This is a controlled synthetic prototype, not a real-data causal certification.

## Workflow

1. Generate {args.weeks} weeks, four products, three tiers, and {5 + (2 if args.amazon else 0)} media channels.
2. Fit the first {args.weeks - args.holdout_weeks} weeks with hierarchical product/tier intercepts,
   price, macro, base-price growth, promotion, post-promotion dip, PrimeDays, and annual seasonality.
3. Estimate channel coefficients and saturation. Display, Video, Social, CTV, and optional Amazon Display
    use independent adstock; paid-search and optional Amazon Search clicks have no adstock. Social has
    a fixed S-shaped Hill power of 2; other powers are 1.
4. Predict the final {args.holdout_weeks} weeks conditional on their supplied media and nonmedia inputs.
5. Compare estimated parameters and allocated contributions with the predeclared synthetic truth.
6. Allocate a 13-week budget using known CPM/CPC mappings, recent carryover, and identical fitted/truth constraints.

Stockout censoring enabled: {args.stockouts}. Funnel mediation enabled: {args.funnel}.
No latent random-walk sales baseline or stochastic CPM/CPC likelihood is included.
When brand mode is enabled, logit brand scores use a linear-Gaussian Kalman filter with
process shocks and noisy score measurements; nonlinear sales are not fed back as Gaussian filter observations.
The generator and fitted model intentionally match. This makes a reference example understandable;
it does not establish robustness to model misspecification or endogenous real-world spend.

## Checks

- Sampling: R-hat {sampling['max_rhat']}, minimum bulk ESS {sampling['min_bulk_ess']:.0f},
  minimum tail ESS {sampling['min_tail_ess']:.0f}, divergences {sampling['divergences']},
  maximum depth {sampling['max_tree_depth']}/12. Gate: {sampling['pass']}.
- Holdout log-sales RMSE {forecast['rmse_log_sales']:.4f} versus product-tier mean benchmark
  {forecast['benchmark_rmse_log_sales']:.4f}; reduction {forecast['rmse_improvement']:.1%}.
- Nominal 90% predictive coverage {forecast['coverage_90']:.1%}. Forecast gate: {forecast['pass']}.
- Media truths in 90% parameter intervals: {recovery['media_truths_covered']}/{recovery['media_total']}.
  Maximum media relative error {recovery['max_media_relative_error']:.1%}; maximum decay absolute error
  {recovery['max_adstock_absolute_error']:.3f}. Recovery gate: {recovery['pass']}.
- Contribution accounting gate: {attribution.get('pass', 'not run')}.
- Budget and known-truth unit-objective gate: {optimization.get('pass', 'not run')}.

Gates were specified in code before fitting: R-hat below 1.01, bulk ESS at least 400,
tail ESS at least 200, no divergences or depth-cap hits; forecast beats benchmark with
coverage between 85% and 97%; at least four of {recovery['media_total']} media truths covered, maximum
media coefficient error 25%, decay error 0.10; contribution error 25%; budget conserved
and fitted allocation within 2% of the best known-truth objective found under the same constraints.
One synthetic seed is a demonstration, not a calibration study across datasets.

## Interpretation

Media coefficients are maximum log-sales effects, not constant per-dollar returns.
Half-saturation is measured relative to each channel's reference exposure. Decay controls
carryover over at most {MAX_LAG} lag positions, including the current week.
The attribution table distributes total modeled media lift in proportion to each channel's
log lift. This accounting convention sums exactly to expected sales; individual channel-removal
effects would not generally sum because the sales model is multiplicative.
In stockout mode both baseline and media lift are evaluated as expected fulfilled sales
under the same inventory caps, not as unconstrained demand contributions.

Optimization holds one weekly share per channel constant for 13 weeks, keeps product proportions
fixed, and bounds shares to 40%-180% of their baseline values. Existing adstock is carried forward.
The original allocation maximizes expected fulfilled units when stockouts are enabled, otherwise
expected demand units. A separate expected-profit allocation and a risk-adjusted allocation are
also reported. Future prices, controls, and inventory are supplied planning assumptions.
Six local starts are used for both fitted and known-truth optimizers: the oracle is best found,
not a proof of the global optimum. Uncertainty bands on response curves describe expected lift,
whereas sales forecast bands include observation noise.

All posterior parameter draws are preserved in posterior_draws.npz. Derived forecasts and
attribution use up to 512 draws; unit allocation uses up to 128 posterior draws, and financial
allocations use 512 predictive scenarios with common random numbers across candidate budgets.
Truth columns are used for evaluation only, not as fitted predictors or parameter prior centers.
No generating parameters were tuned against this run's results. Stockout mode reuses the baseline
dates and latent demand for a paired extension; its holdout is a controlled comparison, not a new
independent temporal validation after viewing the baseline.
"""
    profit = optimization.get("profit_optimization")
    if profit:
        margin_lines = "\n".join(
            f"- {product}: {', '.join(f'{rate:.0%}' for rate in rates)}"
            for product, rates in zip(PRODUCTS, profit["contribution_margin_rates_by_product_tier"])
        )
        text += f"""
## Profit And Risk Optimization

The synthetic profit proxy is fulfilled units times list price after discount times an assumed
product-tier contribution margin, minus media and sponsorship spend. Assumed margins are not
observed costs and should be replaced with validated contribution economics before business use:

{margin_lines}

Expected-profit allocation maximizes posterior expected net contribution. Risk-adjusted allocation
maximizes 50% expected net profit plus 50% lower-tail CVaR at the 5% level, across 512 predictive
scenarios. Scenarios include posterior parameter draws, log-Normal demand residuals, inventory caps,
and brand process shocks when enabled. Fixed total spend is held constant; future prices, controls,
and inventory are supplied. Taxes, returns, fulfillment/fixed costs, uncertain media prices, and
value beyond the 13-week horizon are excluded. This is a synthetic decision demonstration, not
validated business profit or a global-optimum guarantee.

When Amazon is enabled, its planned Search allocation is a spend cap/activity input. Scenario CPC
charges are expected paid clicks times the synthetic $2 CPC; unspent cap remains unspent. The profit
allocation CSV reports planned budget and expected charged spend separately.

- Expected net profit: baseline ${profit['expected_profit_baseline']:,.0f}; optimized ${profit['expected_profit_optimized']:,.0f}.
- 5% predictive profit CVaR: baseline ${profit['predictive_profit_cvar_5_baseline']:,.0f};
    risk-adjusted allocation ${profit['predictive_profit_cvar_5_risk_allocation']:,.0f}.
- Risk weight on CVaR: {profit['risk_weight_on_cvar']:.0%}; total-budget error: ${profit['budget_absolute_error']:.6f}.
"""
    if args.stockouts:
        text += f"""
## Stockout Extension

The underlying realized demand and predictors are identical to the {'uncapped funnel' if args.funnel else 'baseline'} run at the same seed.
Inventory uses an independent random stream and conditional demand quantiles with product/tier
variation and temporary supply reductions. It is not constructed from realized demand shocks.
Observed sales equal min(latent demand, inventory). The fit uses a right-censored Normal likelihood
in log units: a capped observation contributes a survival probability, not a density at the cap.
Inventory and stockout flags do not enter the demand predictor. Synthetic latent demand is evaluation
truth only. No separate classifier or supply-forecast model is fitted; event probability is implied
by the fitted demand distribution and supplied cap.

- Training stockouts: {forecast['train_stockout_rate']:.1%}; holdout: {forecast['holdout_stockout_rate']:.1%}.
- Holdout latent-demand log RMSE: {forecast['latent_rmse_log_demand']:.4f}.
- Latent-demand 90% coverage: {forecast['latent_coverage_90']:.1%}.
- Predicted holdout stockout rate: {forecast['holdout_predicted_stockout_rate']:.1%}.
- Stockout Brier score: {forecast['stockout_brier']:.4f}; constant training-rate benchmark:
    {forecast['stockout_brier_training_rate_benchmark']:.4f}.

Fulfilled-sales means and scenario objectives use the analytic expectation E[min(D, cap)] for
log-Normal demand, not min(E[D], cap). Predictive draws are capped, while latent intervals are not.
Latent-demand coverage is evaluated unconditionally across all held-out demand realizations;
conditioning on realized stockouts selects an upper tail and does not imply nominal subgroup coverage.
The forecast gate additionally requires latent log RMSE below 0.10 and latent coverage 85%-97%.
Event probabilities are descriptive calibration checks, not an additional pass gate at this stage.

The optimizer holds provided inventory fixed while changing media, so spending can lose marginal
value at capacity. It does not optimize replenishment or carry unused inventory between weeks.
Caps were generated using synthetic expected demand, but only numerical caps are given to planning;
the cap-generating truth is not used as a demand predictor. Future caps must be supplied in real use.
"""
    if args.funnel:
        funnel = results["funnel"]
        text += f"""
## Funnel Extension

CTV saturation changes latent branded-search intensity together with macro, seasonality,
and an observed exogenous organic-demand index. Observed search volume is rounded log-Normal.
Paid clicks are Binomial conditional on observed search counts, with capture probability
max_rate * paid_activity / (paid_activity + budget_half). Zero spend implies zero paid clicks.
Likelihoods are fitted once per product-week, never repeated across price tiers. Mediator
likelihoods exclude the entire holdout, just like the sales likelihood.

Sales use latent search intensity times capture probability, not observed holdout searches
or clicks. Independent exogenous variation in CTV, organic demand, and paid-search spend
is deliberately included to identify this matched synthetic example. There is no separate
organic-search sales coefficient, and paid clicks still have no adstock. Display, Video,
Social, and CTV retain their own adstock and saturation.

- Search holdout log RMSE: {funnel['search_holdout_log_rmse']:.4f}; 90% coverage:
    {funnel['search_holdout_coverage_90']:.1%}.
- Click holdout log RMSE: {funnel['clicks_holdout_log_rmse']:.4f}; 90% coverage:
    {funnel['clicks_holdout_coverage_90']:.1%}.
- CTV mediated log-lift relative recovery error: {funnel['ctv_mediated_log_lift_relative_error']:.1%}.
- Funnel gate: {funnel['pass']}.

Attribution assigns CTV its direct log lift plus the additional click-related log lift
caused by CTV. Paid search receives the remaining click-related log lift with CTV removed.
This is an explicitly chosen ordered mediation accounting convention; other interaction
allocation conventions are possible. The components sum exactly without counting the
same mediated pathway twice. Counterfactual response curves and optimization recompute
search intensity and clicks when CTV or paid-search spending changes.

Paid-search reference spend is nominal reference clicks times $1.50; it is a bid/capture
activity index, not a promise that actual clicks always cost $1.50. Because search opportunity
changes, effective realized CPC varies. Search intensity exp(log_mean) is a structural latent
driver; expected observed search counts additionally include log-Normal measurement noise.
The sales equation uses the structural driver, not the noisy realized count.

Funnel gates were set before this fit: mediator holdout log RMSE below 0.20; search coverage
85%-97%, click coverage 85%-99% (zero-spend rows have deterministic zero clicks); mediated
log-lift error below 25%; search CTV effect and both click-capture parameter errors below 20%.
Stockout censoring in this run: {args.stockouts}. {'Sales likelihoods and planning use inventory caps after funnel-driven demand.' if args.stockouts else 'This isolates funnel behavior without inventory constraints.'}
"""
        if args.amazon:
                amazon = results["amazon"]
                text += f"""
## Amazon Sponsored Channels

Amazon Display is generated as impression exposure and priced at a synthetic $14 CPM. Amazon Search
generates on-platform query opportunities from exogenous organic demand and controls; paid clicks are
Binomial with spend-dependent capture. The sales model uses expected clicks as a saturated media
feature, not observed holdout clicks. Amazon Search has a synthetic $2 CPC: realized synthetic spend
equals paid clicks times CPC, so zero clicks produce zero click charges. The budget-activity input is
kept separately from realized CPC spend. These assumptions are illustrative and are not Amazon billing
or attribution data.

- Amazon Search opportunity holdout log RMSE: {amazon['amazon_search_opportunities_holdout_log_rmse']:.4f};
    90% coverage: {amazon['amazon_search_opportunities_holdout_coverage_90']:.1%}.
- Amazon paid-click holdout log RMSE: {amazon['amazon_paid_clicks_holdout_log_rmse']:.4f};
    90% coverage: {amazon['amazon_paid_clicks_holdout_coverage_90']:.1%}.
- CPC spend reconciliation error: ${amazon['cpc_spend_reconciliation_error']:.8f}.
- Amazon mediator gate: {amazon['pass']}.

Amazon Search and Display are separate channels from generic Display and Paid Search. Marketplace
search intent, sponsored placement rules, click capture, and platform attribution are not assumed to
match Google or other networks. This extension has a matched synthetic generator/model by design; its
recovery gates do not establish real-world incrementality.
"""
    if getattr(args, "hierarchical_media", False):
        text += f"""
## Hierarchical Media Shrinkage

Each channel has a positive population response and product-specific positive responses.
For product p and channel c, beta[p,c] = population_beta[c] * exp(deviation[p,c]).
Deviations are zero-sum on the log scale, so the population coefficient is the geometric
center of the four product coefficients. A noncentered orthonormal contrast parameterization
uses standard-Normal contrast values and one learned HalfNormal(0.3) scale per channel.
Smaller scales pool product responses more tightly; larger supported scales permit more
heterogeneity. This adds hierarchical shrinkage of media coefficients, not just intercepts.

Product media truths covered by 90% intervals: {recovery['product_media_truths_covered']}/{recovery['product_media_total']}.
Maximum product media coefficient relative error: {recovery['max_product_media_relative_error']:.1%}.
The product-recovery gate uses the same 25% maximum coefficient error tolerance and at least
80% truth inclusion, now evaluated over 20 product-by-channel coefficients. The planted
product deviations were declared before fitting; priors are not centered on those truths.

Attribution, CTV-mediated lift, response curves, and allocation all use product coefficients.
Adstock, half-saturation, nonmedia controls, and mediator parameters remain shared.
Price tiers within a product share its media coefficient. Four groups limit precision of
the learned hierarchy scales. Recovery and the prior structure demonstrate the implementation;
they do not prove shrinkage improves out-of-sample performance over a no-pooling alternative.
No no-pooling comparison or weak-product-data stress test has been run yet.
See product_media_recovery.csv and media_hierarchy_scales.csv for estimates and intervals.
"""
    if getattr(args, "brand_state", False):
        brand = results["brand_state"]
        text += f"""
## Persistent Brand Equity And Baseline Sales

Brand equity is a measured 0-100 score and a persistent latent logit state, not a separate
outcome disconnected from sales. Each product has one weekly score measurement shared across
its tiers. A logistic-normal measurement likelihood is jointly fitted with sales and mediators.
The state blends its prior state with a target driven by {5 + (2 if args.amazon else 0)} media response features,
current sponsorship impressions, and distribution coverage. Measurement noise is estimated;
there are no unobserved process innovations in this first deterministic-state version.

Sponsorship is an impression measure with a separate spend column at a known synthetic $18 CPM.
It affects baseline sales through brand equity, not a new immediate-sales coefficient. Volume-
weighted distribution is 100 * covered-store sales-volume weights / all available-store weights,
per product and week. It measures where the product is currently assorted; it is not an
unweighted store count and there is no separate assortment-breadth variable or direct physical
distribution sales term in this initial design. Coverage is supplied, not optimized or forecast.

Previous-week latent brand score changes log baseline sales relative to a score of 50, per
10 score points. All posterior uncertainty in media transforms and brand parameters is propagated
jointly. Observed held-out brand scores are never sales inputs. Brand baseline contribution is
exported separately from immediate/funnel media attribution. Long-run lift is not double-counted
in the short-run media allocation table, nor attributed retrospectively to particular historic
funding channels. Such an allocation requires a separately agreed counterfactual convention.

- Brand holdout RMSE: {brand['holdout_rmse_score_points']:.3f} score points versus last-known-score
    benchmark {brand['last_score_benchmark_rmse']:.3f}.
- Nominal 90% score coverage: {brand['holdout_coverage_90']:.1%}.
- Estimated weekly persistence: {brand['estimated_persistence']:.3f}.
- Estimated baseline log-sales effect per 10 prior-week score points:
    {brand['estimated_sales_log_effect_per_10_points']:.3f}.
- Maximum brand-response relative recovery error: {brand['max_response_relative_error']:.1%}.
- Brand gate: {brand['pass']}.

Brand gates were set before fitting: beat the last-known-score benchmark, 85%-97% coverage,
maximum driver/sales/persistence relative coefficient error 25%, persistence absolute error 0.10.
The distribution and sponsorship schedules are independently varied in the synthetic design;
real-data feedback/endogeneity and brand measurement construct validity are not established.

The optimizer now has six spend lines including sponsorship. It carries the historical brand
state into 13 future weeks and updates it under each proposed media/sponsorship plan, then
optimizes fulfilled units subject to fixed inventory. Distribution remains a supplied operational
scenario without an invented cost model. The objective remains units, not equity points or profit;
benefits beyond the 13-week horizon are not valued. Brand response coefficients are shared across
products; sales media coefficients retain the selected product hierarchy.

![Persistent brand score forecast](brand_state_forecast.png)
"""
    if optimization:
        text += f"""
## Budget Result

- Fixed 13-week budget: ${optimization['total_budget']:,.0f}.
- Model-predicted lift over baseline allocation: {optimization['predicted_lift']:.1%}.
- Known-truth units at baseline: {optimization['truth_units_at_baseline']:,.0f}.
- Known-truth units at fitted allocation: {optimization['truth_units_at_fitted_allocation']:,.0f}.
- Known-truth best found: {optimization['truth_best_found_units']:,.0f}.
- Known-truth objective shortfall: {optimization['truth_relative_regret']:.2%}.

![Sales forecast](sales_forecast.png)
![Response curves](channel_response_curves.png)
![Budget allocation](budget_allocation.png)
"""
    (output / "run_report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the identifiable synthetic MMM reference workflow.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stockouts", action="store_true")
    parser.add_argument("--funnel", action="store_true")
    parser.add_argument("--hierarchical-media", action="store_true")
    parser.add_argument("--brand-state", action="store_true")
    parser.add_argument("--amazon", action="store_true")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--weeks", type=int, default=156)
    parser.add_argument("--holdout-weeks", type=int, default=26)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.target_accept < 1:
        parser.error("--target-accept must be between 0 and 1")
    if args.weeks - args.holdout_weeks < 104 or args.holdout_weeks < 13:
        parser.error("Use at least 104 training weeks and at least 13 holdout weeks.")
    if args.funnel and args.stockouts:
        default_folder = "outputs_combined"
    else:
        default_folder = "outputs_funnel" if args.funnel else ("outputs_stockouts" if args.stockouts else "outputs")
    if args.hierarchical_media:
        default_folder += "_hierarchical"
    if args.brand_state:
        default_folder += "_brand_state"
    if args.amazon:
        default_folder += "_amazon"
    output = (args.output or (ROOT / default_folder)).resolve()
    if args.funnel and args.stockouts:
        protected = {"outputs", "outputs_stockouts", "outputs_funnel"}
    else:
        protected = {"outputs", "outputs_stockouts"} if args.funnel else ({"outputs"} if args.stockouts else set())
    if args.hierarchical_media:
        protected.update({"outputs", "outputs_stockouts", "outputs_funnel", "outputs_combined"})
    if args.brand_state:
        protected.update({"outputs", "outputs_stockouts", "outputs_funnel", "outputs_combined", "outputs_combined_hierarchical", "outputs_brand_equity"})
    if args.amazon:
        protected.update({
            "outputs", "outputs_stockouts", "outputs_funnel", "outputs_combined",
            "outputs_combined_hierarchical", "outputs_combined_hierarchical_brand_state",
            "outputs_kalman_profit_validated", "outputs_brand_equity",
            "outputs_brand_state_validated",
        })
    if output in {(ROOT / name).resolve() for name in protected}:
        parser.error("Extension runs must not overwrite the preserved reference output folders.")
    output.mkdir(parents=True, exist_ok=True)
    frame, truth = generate_data(
        args.weeks, args.seed, stockouts=args.stockouts, funnel=args.funnel,
        hierarchical_media=args.hierarchical_media,
        brand_state=args.brand_state,
        amazon=args.amazon,
    )
    save_data(frame, truth, output)
    if args.generate_only:
        print(f"Generated {len(frame)} rows and saved truth in {output}")
        return
    train_weeks = args.weeks - args.holdout_weeks
    data = prepare_data(frame)
    data["hierarchical_media"] = args.hierarchical_media
    model = build_model(data, train_weeks)
    print(f"Fit {train_weeks} weeks; preserve {args.holdout_weeks} holdout weeks. Stockouts={args.stockouts}; funnel={args.funnel}; hierarchical_media={args.hierarchical_media}; brand_state={args.brand_state}; amazon={args.amazon}.")
    with model:
        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=args.chains,
            random_seed=args.seed + 1,
            step=pm.NUTS(target_accept=args.target_accept, max_treedepth=12),
        )
    samples = extract_samples(idata.posterior)
    np.savez_compressed(output / "posterior_draws.npz", **samples)
    sampling = sampler_diagnostics(model, idata, output)
    recovery = parameter_recovery(samples, truth, output)
    forecast, expected, base, effects = forecast_outputs(
        select_samples(samples), data, train_weeks, output, args.seed + 2
    )
    results = {"sampling": sampling, "forecast": forecast, "recovery": recovery}
    if args.funnel:
        results["funnel"] = funnel_outputs(select_samples(samples), data, truth, train_weeks, output, args.seed + 3)
    if args.amazon:
        results["amazon"] = amazon_search_outputs(
            select_samples(samples), data, truth, train_weeks, output, args.seed + 5
        )
    if args.brand_state:
        results["brand_state"] = brand_state_outputs(select_samples(samples), data, truth, train_weeks, output, args.seed + 4)
    if sampling["pass"]:
        derived_samples = select_samples(samples)
        results["attribution"] = attribution_outputs(
            derived_samples, data, truth, expected, base, effects, train_weeks, output
        )
        results["optimization"] = optimization_outputs(samples, data, truth, train_weeks, output)
        write_charts(output, train_weeks)
    results["overall_pass"] = all(result["pass"] for result in results.values()) and "optimization" in results
    config = {
        "seed": args.seed, "weeks": args.weeks, "train_weeks": train_weeks,
        "holdout_weeks": args.holdout_weeks, "chains": args.chains,
        "tune": args.tune, "draws": args.draws,
        "target_accept": args.target_accept,
        "likelihood": "censored log-Normal" if args.stockouts else "uncensored log-Normal",
        "stockouts": args.stockouts, "funnel": args.funnel,
        "amazon": args.amazon,
        "channels": data["channels"],
        "channel_unit_costs": data["unit_cost"].tolist(),
        "hierarchical_media": args.hierarchical_media,
        "brand_state": args.brand_state,
        "brand_state_model": (
            "linear-Gaussian Kalman filter on logit score observations; process noise included; "
            "sales depend on moment-adjusted latent state and are not filter measurements"
            if args.brand_state else None
        ),
        "brand_process_sigma_prior": "HalfNormal(0.05)" if args.brand_state else None,
        "contribution_margin_rates_by_product_tier": CONTRIBUTION_MARGIN_RATE.tolist(),
        "profit_definition": "fulfilled units x net unit price x assumed contribution-margin rate - media/sponsorship spend",
        "risk_adjustment": {
            "objective": "equal-weight expected net profit and lower-tail CVaR",
            "cvar_tail_probability": PROFIT_CVAR_ALPHA,
            "cvar_weight": PROFIT_RISK_WEIGHT,
            "predictive_scenarios": PROFIT_SCENARIOS,
        },
        "conditional_forecast": True,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args, output)
    print(json.dumps(results, indent=2))
    print(f"Results saved in {output}")


if __name__ == "__main__":
    main()