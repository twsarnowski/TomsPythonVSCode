from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, expit


CHANNELS = ["display", "video", "social", "ctv", "paid_search"]
AMAZON_CHANNELS = ["amazon_display", "amazon_search"]
PRODUCTS = ["Entry-Level Desktops", "Gaming PCs", "Premium Desktops", "Premium Laptops"]
TIERS = ["Core", "Premium", "Value"]
REFERENCE_EXPOSURE = np.array([1_000_000, 1_200_000, 1_000_000, 700_000, 8_000.0])
AMAZON_REFERENCE_EXPOSURE = np.array([900_000.0, 12_000.0])
PRODUCT_FACTORS = np.array([0.8, 1.2, 1.0, 1.1])
UNIT_COST = np.array([4 / 1000, 12 / 1000, 9 / 1000, 22 / 1000, 1.5])
AMAZON_UNIT_COST = np.array([14 / 1000, 2.0])
CONTRIBUTION_MARGIN_RATE = np.array([
    [0.18, 0.20, 0.16],
    [0.22, 0.28, 0.19],
    [0.25, 0.30, 0.22],
    [0.20, 0.26, 0.18],
])
HILL_POWER = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
MAX_LAG = 13
CONTROL_NAMES = [
    "price", "macro", "base_price", "promotion", "post_promotion",
    "prime_days", "seasonal_sin", "seasonal_cos",
]


def lagged_exposures(exposures):
    product_count, week_count, channel_count = exposures.shape
    lagged = np.zeros((product_count, week_count, MAX_LAG, channel_count))
    valid = np.zeros((week_count, MAX_LAG))
    for lag in range(MAX_LAG):
        if lag < week_count:
            lagged[:, lag:, lag, :] = exposures[:, :week_count - lag, :]
            valid[lag:, lag] = 1
    return lagged, valid


def media_features(exposures, decay, half_saturation):
    channel_count = exposures.shape[-1]
    lagged, valid = lagged_exposures(exposures)
    weights = np.asarray(decay)[None, :] ** np.arange(MAX_LAG)[:, None]
    denominator = np.einsum("tl,lc->tc", valid, weights)
    adstock = np.einsum("ptlc,lc->ptc", lagged, weights) / denominator[None, :, :]
    hill_power = hill_power_for_channels(channel_count)
    powered = adstock ** hill_power
    return powered / (powered + np.asarray(half_saturation) ** hill_power)


def channel_costs(channels):
    costs = dict(zip(CHANNELS, UNIT_COST))
    costs.update(zip(AMAZON_CHANNELS, AMAZON_UNIT_COST))
    return np.array([costs[name] for name in channels])


def channel_reference(channels):
    references = dict(zip(CHANNELS, REFERENCE_EXPOSURE))
    references.update(zip(AMAZON_CHANNELS, AMAZON_REFERENCE_EXPOSURE))
    return np.array([references[name] for name in channels])


def hill_power_for_channels(channel_count):
    extended = np.concatenate([HILL_POWER, np.ones(len(AMAZON_CHANNELS))])
    return extended[:channel_count]


def activity_schedule(rng, week_count):
    levels = np.array([0.0, 0.05, 0.20, 0.50, 1.0, 2.0, 4.0, 8.0])
    schedule = []
    while len(schedule) < week_count:
        for level in rng.permutation(levels):
            schedule.extend([float(level)] * int(rng.integers(1, 5)))
    return np.array(schedule[:week_count])


def expected_fulfilled_units(log_mean, sigma, inventory=None):
    log_mean = np.asarray(log_mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    demand = np.exp(log_mean + sigma**2 / 2)
    if inventory is None:
        return demand
    inventory = np.asarray(inventory, dtype=float)
    if np.any(inventory <= 0) or np.any(sigma <= 0):
        raise ValueError("Inventory and observation noise must be positive.")
    log_inventory = np.log(inventory)
    below_cap = ndtr((log_inventory - log_mean - sigma**2) / sigma)
    above_cap = ndtr((log_mean - log_inventory) / sigma)
    return demand * below_cap + inventory * above_cap


def add_stockouts(frame, truth, seed):
    frame = frame.copy()
    truth = dict(truth)
    rng = np.random.default_rng(seed + 700)
    weeks = pd.DatetimeIndex(sorted(frame.week.unique()))
    risk = np.empty(len(frame))
    product_risk = [0.12, 0.16, 0.08, 0.10]
    tier_adjustment = [0.0, -0.02, 0.03]
    for product, product_name in enumerate(PRODUCTS):
        disruption = np.zeros(len(weeks))
        for segment in [1, 2, 3]:
            start = int(rng.integers(segment * len(weeks) // 4, min((segment + 1) * len(weeks) // 4, len(weeks) - 5)))
            disruption[start:start + 4] = 0.25
        for tier, tier_name in enumerate(TIERS):
            selected = (frame["product"] == product_name) & (frame["tier"] == tier_name)
            week_index = weeks.get_indexer(frame.loc[selected, "week"])
            risk[selected] = np.clip(
                product_risk[product] + tier_adjustment[tier]
                + disruption[week_index] + rng.uniform(-0.02, 0.02, selected.sum()),
                0.03, 0.60,
            )
    cap = np.maximum(
        1,
        np.ceil(np.exp(frame.truth_log_sales_mean.to_numpy() + truth["sigma"] * ndtri(1 - risk))).astype(np.int64),
    )
    frame["latent_demand_units"] = frame["units"].to_numpy()
    frame["inventory_available"] = cap
    frame["stockout"] = (frame.latent_demand_units.to_numpy() >= cap).astype(int)
    frame["units"] = np.minimum(frame.latent_demand_units.to_numpy(), cap)
    frame["truth_expected_fulfilled_units"] = expected_fulfilled_units(
        frame.truth_log_sales_mean.to_numpy(), truth["sigma"], cap
    )
    truth["stockout_design"] = {
        "inventory_seed": seed + 700,
        "product_event_probabilities": product_risk,
        "tier_probability_adjustments": tier_adjustment,
        "supply_disruption_probability_increment": 0.25,
        "disruption_length_weeks": 4,
        "cap_rule": "conditional demand quantile plus independent supply variation; not realized demand",
    }
    return frame, truth


def funnel_search_log_mean(ctv_feature, controls, intercept, ctv_beta, control_beta):
    return (
        np.asarray(intercept)[..., :, None]
        + np.asarray(ctv_beta)[..., None, None] * ctv_feature
        + np.einsum("...k,ptk->...pt", np.asarray(control_beta), controls)
    )


def funnel_click_probability(budget_ratio, max_rate, budget_half):
    return np.asarray(max_rate)[..., None, None] * budget_ratio / (
        budget_ratio + np.asarray(budget_half)[..., None, None]
    )


def add_funnel(frame, truth, seed):
    frame = frame.copy()
    truth = dict(truth)
    data = prepare_data(frame)
    rng = np.random.default_rng(seed + 900)
    week_count = len(data["weeks"])
    organic_log = rng.normal(0, 0.45, (len(PRODUCTS), week_count))
    macro = np.log(frame.groupby("week").macro_index.first().reindex(data["weeks"]).to_numpy())
    angle = 2 * np.pi * data["weeks"].isocalendar().week.to_numpy(dtype=float) / 52
    search_controls = np.stack([
        np.broadcast_to(macro, organic_log.shape),
        np.broadcast_to(np.sin(angle), organic_log.shape),
        np.broadcast_to(np.cos(angle), organic_log.shape), organic_log,
    ], axis=-1)
    search_reference = 80000 * PRODUCT_FACTORS
    truth.update({
        "search_intercept": np.log(search_reference).tolist(),
        "search_ctv_beta": 0.65,
        "search_control_beta": [0.45, 0.10, -0.08, 0.70],
        "search_sigma": 0.08,
        "click_max_rate": 0.25,
        "click_budget_half": 0.70,
        "funnel_design": {
            "seed": seed + 900,
            "search_family": "rounded log-Normal",
            "click_family": "Binomial conditional on observed searches and exogenous paid spend",
            "organic_index": "independent observed search-demand driver",
            "paid_budget_reference": "paid-search reference clicks times nominal CPC",
        },
    })
    features = media_features(data["ratio"], truth["decay"], truth["half_saturation"])
    search_log_mean = funnel_search_log_mean(
        features[:, :, 3], search_controls, truth["search_intercept"],
        truth["search_ctv_beta"], truth["search_control_beta"],
    )
    searches = np.maximum(1, np.rint(np.exp(rng.normal(search_log_mean, truth["search_sigma"])))).astype(np.int64)
    budget_ratio = data["ratio"][:, :, 4]
    probability = funnel_click_probability(budget_ratio, truth["click_max_rate"], truth["click_budget_half"])
    clicks = rng.binomial(searches, probability)
    product = data["product"]
    week = data["week"]
    expected_clicks = np.exp(search_log_mean) * probability
    click_ratio = expected_clicks / data["reference"][:, 4:5]
    no_ctv_log_mean = funnel_search_log_mean(
        np.zeros_like(features[:, :, 3]), search_controls, truth["search_intercept"],
        truth["search_ctv_beta"], truth["search_control_beta"],
    )
    no_ctv_click_ratio = np.exp(no_ctv_log_mean) * probability / data["reference"][:, 4:5]
    paid_beta = np.asarray(truth.get("product_media_beta", np.tile(truth["media_beta"], (len(PRODUCTS), 1))))[:, 4:5]
    paid_effect = paid_beta * click_ratio / (click_ratio + truth["half_saturation"][4])
    no_ctv_paid_effect = paid_beta * no_ctv_click_ratio / (no_ctv_click_ratio + truth["half_saturation"][4])
    original_mu = frame.truth_log_sales_mean.to_numpy()
    new_mu = (
        original_mu - frame.truth_paid_search_log_effect.to_numpy()
        + paid_effect[product, week]
    )
    original_noise = np.log(frame.units.to_numpy(float)) - original_mu
    frame["units"] = np.maximum(1, np.rint(np.exp(new_mu + original_noise))).astype(np.int64)
    frame["truth_log_sales_mean"] = new_mu
    frame["truth_expected_units"] = np.exp(new_mu + truth["sigma"]**2 / 2)
    frame["truth_ctv_direct_log_effect"] = frame.truth_ctv_log_effect.to_numpy()
    frame["truth_ctv_mediated_log_effect"] = (paid_effect - no_ctv_paid_effect)[product, week]
    frame["truth_ctv_log_effect"] += frame.truth_ctv_mediated_log_effect
    frame["truth_paid_search_log_effect"] = no_ctv_paid_effect[product, week]
    frame["paid_search_exposure"] = clicks[product, week]
    frame["branded_search_volume"] = searches[product, week]
    frame["organic_demand_index"] = np.exp(organic_log[product, week])
    frame["search_reference"] = search_reference[product]
    frame["truth_click_mean"] = expected_clicks[product, week]
    frame["truth_search_log_mean"] = search_log_mean[product, week]
    frame["truth_click_probability"] = probability[product, week]
    truth["funnel_design"]["sales_driver"] = "latent search intensity times spend-dependent capture probability"
    truth["funnel_design"]["attribution_convention"] = "CTV receives direct plus mediated log lift; paid search receives no-CTV click log lift"
    return frame, truth


def add_amazon_search(frame, truth, seed):
    frame = frame.copy()
    truth = dict(truth)
    data = prepare_data(frame)
    rng = np.random.default_rng(seed + 1300)
    week_count = len(data["weeks"])
    organic_log = rng.normal(0, 0.4, (len(PRODUCTS), week_count))
    macro = np.log(frame.groupby("week").macro_index.first().reindex(data["weeks"]).to_numpy())
    angle = 2 * np.pi * data["weeks"].isocalendar().week.to_numpy(dtype=float) / 52
    controls = np.stack([
        np.broadcast_to(macro, organic_log.shape),
        np.broadcast_to(np.sin(angle), organic_log.shape),
        np.broadcast_to(np.cos(angle), organic_log.shape), organic_log,
    ], axis=-1)
    search_reference = 100_000 * PRODUCT_FACTORS
    search_intercept = np.log(search_reference)
    search_control_beta = np.array([0.35, 0.08, -0.06, 0.65])
    search_sigma = 0.09
    click_max_rate = 0.22
    click_budget_half = 0.65
    amazon_search_index = data["channels"].index("amazon_search")
    amazon_display_index = data["channels"].index("amazon_display")
    beta = np.asarray(truth.get(
        "product_media_beta",
        np.tile(truth["media_beta"], (len(PRODUCTS), 1)),
    ))[:, amazon_search_index]
    half = truth["half_saturation"][amazon_search_index]
    log_search = (
        search_intercept[:, None]
        + np.einsum("ptk,k->pt", controls, search_control_beta)
    )
    search_counts = np.maximum(1, np.rint(
        np.exp(rng.normal(log_search, search_sigma))
    )).astype(np.int64)
    budget_ratio = data["ratio"][:, :, amazon_search_index]
    planned_budget = budget_ratio * data["reference"][:, amazon_search_index:amazon_search_index + 1] * AMAZON_UNIT_COST[1]
    probability = funnel_click_probability(budget_ratio, click_max_rate, click_budget_half)
    clicks = rng.binomial(search_counts, probability)
    expected_clicks = np.exp(log_search) * probability
    click_ratio = expected_clicks / data["reference"][:, amazon_search_index:amazon_search_index + 1]
    feature = click_ratio / (click_ratio + half)
    product, week = data["product"], data["week"]
    effect = beta[product] * feature[product, week]
    original_mean = frame.truth_log_sales_mean.to_numpy()
    noise = np.log(frame.units.to_numpy(float)) - original_mean
    updated_mean = original_mean - frame.truth_amazon_search_log_effect.to_numpy() + effect
    frame["units"] = np.maximum(1, np.rint(np.exp(updated_mean + noise))).astype(np.int64)
    frame["truth_log_sales_mean"] = updated_mean
    frame["truth_expected_units"] = np.exp(updated_mean + truth["sigma"]**2 / 2)
    frame["truth_amazon_search_log_effect"] = effect
    frame["amazon_search_exposure"] = clicks[product, week]
    frame["amazon_search_budget_activity"] = planned_budget[product, week]
    frame["amazon_search_spend"] = clicks[product, week] * AMAZON_UNIT_COST[1]
    frame["amazon_search_volume"] = search_counts[product, week]
    frame["amazon_organic_demand_index"] = np.exp(organic_log[product, week])
    frame["amazon_search_reference"] = search_reference[product]
    frame["truth_amazon_search_click_mean"] = expected_clicks[product, week]
    frame["truth_amazon_search_log_mean"] = log_search[product, week]
    frame["truth_amazon_search_click_probability"] = probability[product, week]
    truth.update({
        "amazon_search_intercept": search_intercept.tolist(),
        "amazon_search_control_beta": search_control_beta.tolist(),
        "amazon_search_sigma": search_sigma,
        "amazon_click_max_rate": click_max_rate,
        "amazon_click_budget_half": click_budget_half,
        "amazon_search_design": {
            "seed": seed + 1300,
            "search_family": "rounded log-Normal query opportunities",
            "click_family": "Binomial conditional on on-Amazon query opportunities and synthetic CPC budget activity",
            "billing": "synthetic CPC; realized spend equals valid paid clicks times nominal CPC",
            "budget_activity": "planned spend-equivalent activity index controls click capture",
            "sales_driver": "latent query intensity times spend-dependent click capture",
            "organic_index": "independent exogenous product-query demand driver",
        },
    })
    truth["channels"] = data["channels"]
    truth["unit_cost"] = channel_costs(data["channels"]).tolist()
    truth["reference_exposure"] = data["reference"].tolist()
    return frame, truth


def persistent_brand_state(target, persistence, initial, innovations=None):
    target = np.asarray(target)
    persistence = np.asarray(persistence)
    initial = np.asarray(initial)
    weeks = target.shape[-1]
    lag = np.arange(weeks)[:, None] - np.arange(weeks)[None, :]
    weights = np.where(lag >= 0, persistence[..., None, None] ** np.maximum(lag, 0), 0.0)
    initial_path = initial[..., :, None] * persistence[..., None, None] ** np.arange(1, weeks + 1)
    state = initial_path + (1 - persistence[..., None, None]) * np.matmul(target, np.swapaxes(weights, -1, -2))
    if innovations is not None:
        state += np.matmul(np.asarray(innovations), np.swapaxes(weights, -1, -2))
    return state


def kalman_filter_brand_state(target, persistence, initial, process_sigma,
                              measurement_sigma, observations, observed_weeks):
    target = np.asarray(target, dtype=float)
    persistence = np.asarray(persistence, dtype=float)
    initial = np.asarray(initial, dtype=float)
    process_sigma = np.asarray(process_sigma, dtype=float)
    measurement_sigma = np.asarray(measurement_sigma, dtype=float)
    observations = np.asarray(observations, dtype=float)
    draw_count, product_count, week_count = target.shape
    means = np.empty_like(target)
    variances = np.empty_like(target)
    previous_mean = np.broadcast_to(initial, (draw_count, product_count)).copy()
    previous_variance = np.zeros((draw_count, product_count))
    for week in range(week_count):
        predicted_mean = (
            persistence[:, None] * previous_mean
            + (1 - persistence[:, None]) * target[:, :, week]
        )
        predicted_variance = (
            persistence[:, None] ** 2 * previous_variance + process_sigma[:, None] ** 2
        )
        if week < observed_weeks:
            innovation_variance = predicted_variance + measurement_sigma[:, None] ** 2
            gain = predicted_variance / innovation_variance
            innovation = observations[None, :, week] - predicted_mean
            previous_mean = predicted_mean + gain * innovation
            previous_variance = (1 - gain) * predicted_variance
        else:
            previous_mean = predicted_mean
            previous_variance = predicted_variance
        means[:, :, week] = previous_mean
        variances[:, :, week] = previous_variance
    return means, variances


def add_brand_state(frame, truth, seed):
    frame = frame.copy()
    truth = dict(truth)
    data = prepare_data(frame)
    weeks = len(data["weeks"])
    rng = np.random.default_rng(seed + 1100)
    sponsorship_reference = 800000 * PRODUCT_FACTORS
    sponsorship_ratio = np.stack([activity_schedule(rng, weeks) for _ in PRODUCTS])
    sponsorship = sponsorship_ratio * sponsorship_reference[:, None]
    coverage = np.empty((len(PRODUCTS), weeks))
    store_weights = rng.lognormal(0, 0.7, 40)
    for product in range(len(PRODUCTS)):
        week = 0
        while week < weeks:
            store_count = int(rng.choice([8, 14, 20, 26, 32, 38]))
            available = rng.choice(40, store_count, replace=False)
            weighted_fraction = store_weights[available].sum() / store_weights.sum()
            end = min(weeks, week + int(rng.integers(4, 9)))
            coverage[product, week:end] = weighted_fraction
            week = end
    features = media_features(data["ratio"], truth["decay"], truth["half_saturation"])
    if data["funnel"]:
        log_search = funnel_search_log_mean(
            features[:, :, 3], data["search_controls"], truth["search_intercept"],
            truth["search_ctv_beta"], truth["search_control_beta"],
        )
        capture = funnel_click_probability(data["paid_budget_ratio"], truth["click_max_rate"], truth["click_budget_half"])
        clicks = np.exp(log_search) * capture / data["reference"][:, 4:5]
        features[:, :, 4] = clicks / (clicks + truth["half_saturation"][4])
    if data["amazon"]:
        amazon_search_index = data["channels"].index("amazon_search")
        log_search = (
            np.asarray(truth["amazon_search_intercept"])[:, None]
            + np.einsum(
                "ptk,k->pt", data["amazon_search_controls"],
                np.asarray(truth["amazon_search_control_beta"]),
            )
        )
        capture = funnel_click_probability(
            data["amazon_budget_ratio"], truth["amazon_click_max_rate"],
            truth["amazon_click_budget_half"],
        )
        clicks = np.exp(log_search) * capture / data["reference"][:, amazon_search_index:amazon_search_index + 1]
        features[:, :, amazon_search_index] = clicks / (
            clicks + truth["half_saturation"][amazon_search_index]
        )
    truth.update({
        "brand_intercept": -0.35,
        "brand_product_effect": [-0.20, 0.10, -0.05, 0.15],
        "brand_initial": [-0.20, 0.10, -0.05, 0.15],
        "brand_media_beta": [0.35, 0.50, 0.60, 0.45, 0.30] + ([0.38, 0.28] if data["amazon"] else []),
        "brand_sponsorship_beta": 0.45,
        "brand_distribution_beta": 0.90,
        "brand_persistence": 0.90,
        "brand_process_sigma": 0.025,
        "brand_measurement_sigma": 0.025,
        "brand_sales_beta": 0.08,
        "sponsorship_reference": sponsorship_reference.tolist(),
        "sponsorship_cpm": 18.0,
        "brand_design": {
            "seed": seed + 1100, "score_scale": "0-100 logistic-normal survey score",
            "state": "logit autoregression with Gaussian process innovations and noisy measurements",
            "distribution": "100 * covered-store volume / available-store volume",
            "available_store_volume_weights": store_weights.tolist(),
            "assortment": "captured by product distribution coverage; no separate SKU breadth",
            "sales": "previous-week score relative to 50, scaled per 10 points",
        },
    })
    sponsor_feature = sponsorship_ratio / (sponsorship_ratio + 1)
    target = (
        truth["brand_intercept"] + np.asarray(truth["brand_product_effect"])[:, None]
        + features @ truth["brand_media_beta"] + truth["brand_sponsorship_beta"] * sponsor_feature
        + truth["brand_distribution_beta"] * (coverage - 0.5)
    )
    process_innovations = rng.normal(0, truth["brand_process_sigma"], size=target.shape)
    state = persistent_brand_state(
        target, truth["brand_persistence"], truth["brand_initial"], process_innovations
    )
    initial = np.asarray(truth["brand_initial"])
    lag_state = np.concatenate([initial[:, None], state[:, :-1]], axis=1)
    log_sales_effect = truth["brand_sales_beta"] * (100 * expit(lag_state) - 50) / 10
    observed_score = 100 * expit(state + rng.normal(0, truth["brand_measurement_sigma"], state.shape))
    product, week = data["product"], data["week"]
    prior_mean = frame.truth_log_sales_mean.to_numpy()
    noise = np.log(frame.units.to_numpy()) - prior_mean
    updated_mean = prior_mean + log_sales_effect[product, week]
    frame["units"] = np.maximum(1, np.rint(np.exp(updated_mean + noise))).astype(np.int64)
    frame["truth_base_log_sales"] += log_sales_effect[product, week]
    frame["truth_log_sales_mean"] = updated_mean
    frame["truth_expected_units"] = np.exp(updated_mean + truth["sigma"]**2 / 2)
    frame["brand_equity_score"] = observed_score[product, week]
    frame["truth_brand_state_logit"] = state[product, week]
    frame["truth_brand_score"] = 100 * expit(state[product, week])
    frame["truth_brand_sales_log_effect"] = log_sales_effect[product, week]
    frame["sponsorship_impressions"] = sponsorship[product, week]
    frame["sponsorship_spend"] = sponsorship[product, week] * truth["sponsorship_cpm"] / 1000
    frame["sponsorship_reference"] = sponsorship_reference[product]
    frame["distribution_volume_weighted_pct"] = 100 * coverage[product, week]
    return frame, truth


def generate_data(weeks=156, seed=20261007, stockouts=False, funnel=False, hierarchical_media=False,
                  brand_state=False, amazon=False):
    if weeks < 104:
        raise ValueError("The reference example requires at least 104 weeks.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2026-10-04", periods=weeks, freq="W-SUN")
    channels = CHANNELS + (AMAZON_CHANNELS if amazon else [])
    channel_reference_values = channel_reference(channels)
    channel_cost_values = channel_costs(channels)
    hill_power = hill_power_for_channels(len(channels))
    exposure_ratio = np.empty((len(PRODUCTS), weeks, len(channels)))
    for product in range(len(PRODUCTS)):
        for channel in range(len(channels)):
            exposure_ratio[product, :, channel] = activity_schedule(rng, weeks)
    reference = PRODUCT_FACTORS[:, None] * channel_reference_values[None, :]
    exposures = exposure_ratio * reference[:, None, :]
    spend = exposures * channel_cost_values
    macro_log = np.zeros(weeks)
    for week in range(1, weeks):
        macro_log[week] = 0.45 * macro_log[week - 1] + rng.normal(0, 0.065)
    base_log = np.empty((len(PRODUCTS), weeks))
    for product in range(len(PRODUCTS)):
        increments = rng.uniform(0.0001, 0.0013, weeks)
        for change in [40 + product * 3, 90 + product * 4, 130 + product * 2]:
            if change < weeks:
                increments[change] += 0.04 + 0.01 * product
        base_log[product] = np.cumsum(increments)
    prime = np.zeros(weeks)
    for year in dates.year.unique():
        for month, day in [(7, 15), (10, 10)]:
            available = np.flatnonzero((dates.year == year) & (dates.month == month))
            if len(available):
                target = pd.Timestamp(year=int(year), month=month, day=day)
                prime[min(available, key=lambda index: abs(dates[index] - target))] = 1
    interactions = rng.normal(0, 0.045, (len(PRODUCTS), len(TIERS)))
    interactions = (
        interactions - interactions.mean(axis=0, keepdims=True)
        - interactions.mean(axis=1, keepdims=True) + interactions.mean()
    )
    truth = {
        "intercept": float(np.log(2400)),
        "product_effect": [-0.30, 0.10, -0.15, 0.35],
        "tier_effect": [0.0, -0.45, 0.45],
        "interaction": interactions.tolist(),
        "media_beta": [0.45, 0.65, 0.75, 0.60, 0.45] + ([0.40, 0.35] if amazon else []),
        "decay": [0.25, 0.45, 0.55, 0.65, 0.0] + ([0.35, 0.0] if amazon else []),
        "half_saturation": [0.80, 1.00, 0.70, 1.15, 0.90] + ([0.95, 0.80] if amazon else []),
        "control_beta": [-1.2, 0.9, -0.85, 0.16, -0.08, float(np.log(1.22)), 0.15, -0.08],
        "sigma": 0.06,
        "seed": seed,
        "weeks": weeks,
        "channels": channels,
        "max_lag_weeks": MAX_LAG,
        "hill_power": hill_power.tolist(),
        "unit_cost": channel_cost_values.tolist(),
        "contribution_margin_rate_by_product_tier": CONTRIBUTION_MARGIN_RATE.tolist(),
        "profit_assumption": "list price after discount times assumed product-tier contribution margin; excludes tax, returns, fulfillment, and fixed costs",
        "reference_exposure": reference.tolist(),
    }
    product_media_beta = np.tile(truth["media_beta"], (len(PRODUCTS), 1))
    if hierarchical_media:
        log_deviations = np.array([
            [-0.28, 0.20, -0.15, 0.25, -0.22, -0.12, -0.10],
            [0.20, -0.24, 0.30, -0.18, 0.12, 0.08, 0.12],
            [-0.10, 0.12, -0.25, -0.10, 0.25, -0.04, 0.04],
            [0.18, -0.08, 0.10, 0.03, -0.15, 0.08, -0.06],
        ])
        log_deviations = log_deviations[:, :len(channels)]
        log_deviations -= log_deviations.mean(axis=0, keepdims=True)
        product_media_beta *= np.exp(log_deviations)
        truth["product_media_beta"] = product_media_beta.tolist()
        truth["media_log_deviations"] = log_deviations.tolist()
        truth["hierarchical_media_design"] = "predeclared zero-sum product log deviations; population coefficient is geometric center"
    features = media_features(exposure_ratio, truth["decay"], truth["half_saturation"])
    angle = 2 * np.pi * dates.isocalendar().week.to_numpy(dtype=float) / 52
    rows = []
    for product, product_name in enumerate(PRODUCTS):
        for tier, tier_name in enumerate(TIERS):
            nominal_price = [649, 1799, 1399, 1499][product] * [1, 1.38, 0.72][tier]
            prior_discount = 0.0
            for week, date in enumerate(dates):
                discount = float(rng.uniform(0.10, 0.30)) if rng.random() < 0.40 else 0.0
                price_log = float(rng.normal(0, 0.10))
                controls = np.array([
                    price_log, macro_log[week], base_log[product, week], discount / 0.1,
                    prior_discount / 0.1, prime[week], np.sin(angle[week]), np.cos(angle[week]),
                ])
                base = (
                    truth["intercept"] + truth["product_effect"][product]
                    + truth["tier_effect"][tier] + interactions[product, tier]
                    + controls @ truth["control_beta"]
                )
                eta = base + features[product, week] @ product_media_beta[product]
                units = max(1, int(np.rint(np.exp(rng.normal(eta, truth["sigma"])))))
                row = {
                    "week": date, "product": product_name, "tier": tier_name,
                    "units": units, "list_price": nominal_price * np.exp(base_log[product, week] + price_log),
                    "nominal_price": nominal_price, "macro_index": np.exp(macro_log[week]),
                    "base_price_index": np.exp(base_log[product, week]),
                    "discount": discount, "PrimeDays": int(prime[week]),
                    "truth_base_log_sales": base, "truth_log_sales_mean": eta,
                    "truth_expected_units": float(np.exp(eta + truth["sigma"]**2 / 2)),
                }
                for channel, name in enumerate(channels):
                    row[f"{name}_exposure"] = exposures[product, week, channel]
                    row[f"{name}_spend"] = spend[product, week, channel]
                    row[f"{name}_reference"] = reference[product, channel]
                    row[f"truth_{name}_log_effect"] = features[product, week, channel] * product_media_beta[product, channel]
                rows.append(row)
                prior_discount = discount
    frame = pd.DataFrame(rows).sort_values(["week", "product", "tier"]).reset_index(drop=True)
    if funnel:
        frame, truth = add_funnel(frame, truth, seed)
    if amazon:
        frame, truth = add_amazon_search(frame, truth, seed)
    if brand_state:
        frame, truth = add_brand_state(frame, truth, seed)
    if stockouts:
        return add_stockouts(frame, truth, seed)
    return frame, truth


def prepare_data(frame):
    frame = frame.sort_values(["week", "product", "tier"]).reset_index(drop=True).copy()
    weeks = pd.DatetimeIndex(sorted(frame.week.unique()))
    product = pd.Categorical(frame["product"], categories=PRODUCTS).codes
    tier = pd.Categorical(frame["tier"], categories=TIERS).codes
    week = pd.Categorical(frame["week"], categories=weeks).codes
    if min(product.min(), tier.min(), week.min()) < 0:
        raise ValueError("Unknown product, tier, or week.")
    lag_discount = frame.groupby(["product", "tier"])["discount"].shift(1).fillna(0).to_numpy()
    angle = 2 * np.pi * frame.week.dt.isocalendar().week.to_numpy(dtype=float) / 52
    controls = np.column_stack([
        np.log(frame.list_price / frame.nominal_price / frame.base_price_index),
        np.log(frame.macro_index), np.log(frame.base_price_index),
        frame.discount / 0.1, lag_discount / 0.1, frame.PrimeDays,
        np.sin(angle), np.cos(angle),
    ])
    product_week = frame.groupby(["product", "week"]).first().reindex(
        pd.MultiIndex.from_product([PRODUCTS, weeks], names=["product", "week"])
    )
    channels = CHANNELS + (AMAZON_CHANNELS if "amazon_display_exposure" in frame else [])
    carry_channels = [name for name in channels if name not in {"paid_search", "amazon_search"}]
    cost_values = channel_costs(channels)
    reference_values = channel_reference(channels)
    ratio = np.stack([
        (product_week[f"{name}_exposure"] / product_week[f"{name}_reference"]).to_numpy().reshape(len(PRODUCTS), len(weeks))
        for name in channels
    ], axis=-1)
    result = {
        "frame": frame, "weeks": weeks, "product": product, "tier": tier,
        "week": week, "controls": controls, "ratio": ratio, "channels": channels,
        "carry_indices": np.array([channels.index(name) for name in carry_channels]),
        "hill_power": hill_power_for_channels(len(channels)), "unit_cost": cost_values,
        "log_units": np.log(frame.units.to_numpy(float)),
        "net_unit_price": (frame.list_price * (1 - frame.discount)).to_numpy(float),
        "contribution_per_unit": (
            frame.list_price * (1 - frame.discount)
            * CONTRIBUTION_MARGIN_RATE[product, tier]
        ).to_numpy(float),
        "contribution_margin_rate": CONTRIBUTION_MARGIN_RATE.copy(),
        "inventory": frame["inventory_available"].to_numpy(float) if "inventory_available" in frame else None,
        "hierarchical_media": False,
        "reference": PRODUCT_FACTORS[:, None] * reference_values[None, :],
    }
    result["funnel"] = "branded_search_volume" in frame
    result["brand_state"] = "brand_equity_score" in frame
    result["amazon"] = "amazon_search_volume" in frame
    if result["brand_state"]:
        score = product_week.brand_equity_score.to_numpy().reshape(len(PRODUCTS), len(weeks))
        result.update({
            "brand_observed_logit": np.log(score / (100 - score)),
            "sponsorship_ratio": (
                product_week.sponsorship_impressions / product_week.sponsorship_reference
            ).to_numpy().reshape(len(PRODUCTS), len(weeks)),
            "sponsorship_reference": product_week.sponsorship_reference.to_numpy().reshape(len(PRODUCTS), len(weeks))[:, 0],
            "distribution_fraction": product_week.distribution_volume_weighted_pct.to_numpy().reshape(len(PRODUCTS), len(weeks)) / 100,
        })
    if result["funnel"]:
        search_counts = product_week.branded_search_volume.to_numpy().reshape(len(PRODUCTS), len(weeks)).astype(np.int64)
        paid_spend = product_week.paid_search_spend.to_numpy().reshape(len(PRODUCTS), len(weeks))
        budget_reference = result["reference"][:, 4:5] * cost_values[4]
        organic_log = np.log(product_week.organic_demand_index.to_numpy().reshape(len(PRODUCTS), len(weeks)))
        macro = np.log(product_week.macro_index.to_numpy().reshape(len(PRODUCTS), len(weeks)))
        angle_week = 2 * np.pi * weeks.isocalendar().week.to_numpy(dtype=float) / 52
        result.update({
            "search_counts": search_counts,
            "click_counts": product_week.paid_search_exposure.to_numpy().reshape(len(PRODUCTS), len(weeks)).astype(np.int64),
            "paid_budget_ratio": paid_spend / budget_reference,
            "search_controls": np.stack([
                macro, np.broadcast_to(np.sin(angle_week), macro.shape),
                np.broadcast_to(np.cos(angle_week), macro.shape), organic_log,
            ], axis=-1),
        })
        result["ratio"][:, :, 4] = result["paid_budget_ratio"]
        result["search_reference"] = (
            product_week.search_reference.to_numpy().reshape(len(PRODUCTS), len(weeks))[:, 0]
        )
    if result["amazon"]:
        amazon_search_index = channels.index("amazon_search")
        amazon_spend = product_week.amazon_search_budget_activity.to_numpy().reshape(len(PRODUCTS), len(weeks))
        amazon_budget_reference = result["reference"][:, amazon_search_index:amazon_search_index + 1] * cost_values[amazon_search_index]
        organic_log = np.log(product_week.amazon_organic_demand_index.to_numpy().reshape(len(PRODUCTS), len(weeks)))
        macro = np.log(product_week.macro_index.to_numpy().reshape(len(PRODUCTS), len(weeks)))
        angle_week = 2 * np.pi * weeks.isocalendar().week.to_numpy(dtype=float) / 52
        result.update({
            "amazon_search_counts": product_week.amazon_search_volume.to_numpy().reshape(len(PRODUCTS), len(weeks)).astype(np.int64),
            "amazon_click_counts": product_week.amazon_search_exposure.to_numpy().reshape(len(PRODUCTS), len(weeks)).astype(np.int64),
            "amazon_search_spend": product_week.amazon_search_spend.to_numpy().reshape(len(PRODUCTS), len(weeks)),
            "amazon_budget_ratio": amazon_spend / amazon_budget_reference,
            "amazon_search_controls": np.stack([
                macro, np.broadcast_to(np.sin(angle_week), macro.shape),
                np.broadcast_to(np.cos(angle_week), macro.shape), organic_log,
            ], axis=-1),
            "amazon_search_reference": product_week.amazon_search_reference.to_numpy().reshape(len(PRODUCTS), len(weeks))[:, 0],
        })
        result["ratio"][:, :, amazon_search_index] = result["amazon_budget_ratio"]
    return result


def save_data(frame, truth, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "synthetic_weekly.csv", index=False, date_format="%Y-%m-%d")
    (output / "truth_parameters.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")