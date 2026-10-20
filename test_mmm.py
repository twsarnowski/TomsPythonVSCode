import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pymc as pm

from mmm_data import (
    CHANNELS, UNIT_COST, CONTRIBUTION_MARGIN_RATE, generate_data, media_features, prepare_data,
    expected_fulfilled_units, kalman_filter_brand_state,
)
from run_mmm import (
    build_model, extract_samples, predictions, planning_context,
    scenario_units, solve_allocation, truth_samples, brand_predictions, optimization_outputs,
    scenario_amazon_search_spend,
)
from run_brand_equity import brand_features, generate_brand_panel, build_brand_model


class ReferenceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.truth = generate_data()
        cls.data = prepare_data(cls.frame)

    def test_generator_matches_prediction_equation(self):
        samples = truth_samples(self.truth)
        mu, base, effects = predictions(samples, self.data)
        np.testing.assert_allclose(mu[0], self.frame.truth_log_sales_mean, atol=1e-10)
        self.assertEqual(len(self.frame), 156 * 4 * 3)
        self.assertFalse(self.frame.duplicated(["week", "product", "tier"]).any())
        self.assertFalse({"stockout", "inventoryavailable"}.intersection(self.frame.columns))
        for channel, cost in zip(CHANNELS, UNIT_COST):
            np.testing.assert_allclose(self.frame[f"{channel}_spend"], self.frame[f"{channel}_exposure"] * cost)
        repeated, _ = generate_data()
        self.assertTrue(self.frame.equals(repeated))

    def test_paid_clicks_have_no_carryover(self):
        exposures = np.zeros((1, 20, 5))
        exposures[0, 0, :] = 1
        features = media_features(exposures, self.truth["decay"], self.truth["half_saturation"])
        self.assertEqual(features[0, 1, -1], 0)
        self.assertGreater(features[0, 1, 0], 0)
        self.assertGreater(features[0, 1, 3], 0)

    def test_kalman_filter_update_and_holdout_forecast(self):
        target = np.array([[[0.4, 0.2]]])
        persistence = np.array([0.8])
        initial = np.array([[0.0]])
        process_sigma = np.array([0.2])
        measurement_sigma = np.array([0.1])
        observations = np.array([[0.5, 50.0]])
        mean, variance = kalman_filter_brand_state(
            target, persistence, initial, process_sigma, measurement_sigma, observations, 1
        )
        predicted_mean = 0.2 * 0.4
        predicted_variance = 0.2**2
        gain = predicted_variance / (predicted_variance + 0.1**2)
        filtered_mean = predicted_mean + gain * (0.5 - predicted_mean)
        filtered_variance = (1 - gain) * predicted_variance
        np.testing.assert_allclose(mean[0, 0, 0], filtered_mean)
        np.testing.assert_allclose(variance[0, 0, 0], filtered_variance)
        np.testing.assert_allclose(mean[0, 0, 1], 0.8 * filtered_mean + 0.2 * 0.2)
        np.testing.assert_allclose(variance[0, 0, 1], 0.8**2 * filtered_variance + 0.2**2)

    def test_brand_index_unique_and_bounded(self):
        data = self.data
        panel, truth, controls = generate_brand_panel(data, self.truth, 20262107)
        self.assertEqual(len(panel), 4 * 156)
        self.assertFalse(panel.duplicated(["product", "week"]).any())
        self.assertTrue(panel.brand_equity_index.between(0, 100, inclusive="neither").all())
        features = brand_features(truth_samples(self.truth), data)[0]
        observed = panel.brand_equity_index.to_numpy().reshape(4, 156) / 100
        model = build_brand_model(features, controls, observed, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        altered = observed.copy()
        altered[:, 130:] = 0.01
        changed = build_brand_model(features, controls, altered, 130)
        np.testing.assert_allclose(model.compile_logp()(model.initial_point()), changed.compile_logp()(changed.initial_point()))

    def test_stockout_extension_preserves_demand_and_caps_sales(self):
        frame, truth = generate_data(stockouts=True)
        np.testing.assert_array_equal(frame.latent_demand_units, self.frame.units)
        for column in self.frame.columns.difference(["units"]):
            self.assertTrue(frame[column].equals(self.frame[column]))
        np.testing.assert_array_equal(frame.units, np.minimum(frame.latent_demand_units, frame.inventory_available))
        np.testing.assert_array_equal(frame.stockout, frame.latent_demand_units >= frame.inventory_available)
        self.assertGreater(frame.stockout.mean(), 0.05)
        self.assertLess(frame.stockout.mean(), 0.30)
        self.assertTrue(np.all(frame.truth_expected_fulfilled_units <= frame.inventory_available))

    def test_expected_capped_sales_matches_simulation(self):
        rng = np.random.default_rng(91)
        demand = rng.lognormal(np.log(100), 0.3, 300000)
        expectation = expected_fulfilled_units(np.log(100), 0.3, 105)
        self.assertAlmostEqual(float(expectation), float(np.minimum(demand, 105).mean()), delta=0.2)
        self.assertLess(float(expectation), float(expected_fulfilled_units(np.log(100), 0.3)))
        self.assertLess(float(expectation), 105)

    def test_censored_gradient_and_inventory_planning(self):
        frame, truth = generate_data(stockouts=True)
        data = prepare_data(frame)
        model = build_model(data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        altered = dict(data)
        altered["log_units"] = data["log_units"].copy()
        altered["log_units"][data["week"] >= 130] += 20
        changed = build_model(altered, 130)
        np.testing.assert_allclose(model.compile_logp()(model.initial_point()), changed.compile_logp()(changed.initial_point()))
        context = planning_context(truth_samples(truth), data, 130)
        self.assertLessEqual(float(scenario_units(context, np.full(5, 10))[0]), float(context["inventory"].sum()) + 1e-6)

    def test_funnel_generator_and_accounting(self):
        frame, truth = generate_data(funnel=True)
        data = prepare_data(frame)
        self.assertTrue(data["funnel"])
        self.assertTrue((frame.paid_search_exposure <= frame.branded_search_volume).all())
        self.assertTrue((frame.loc[frame.paid_search_spend == 0, "paid_search_exposure"] == 0).all())
        np.testing.assert_array_equal(frame.paid_search_spend, self.frame.paid_search_spend)
        mu, base, effects = predictions(truth_samples(truth), data)
        np.testing.assert_allclose(mu[0], frame.truth_log_sales_mean, atol=1e-10)
        np.testing.assert_allclose(base[0] + effects[0].sum(axis=1), mu[0], atol=1e-10)
        np.testing.assert_allclose(
            frame.truth_base_log_sales + frame[[f"truth_{name}_log_effect" for name in CHANNELS]].sum(axis=1),
            frame.truth_log_sales_mean, atol=1e-10,
        )

    def test_amazon_channels_generator_model_and_cpc_billing(self):
        frame, truth = generate_data(amazon=True)
        data = prepare_data(frame)
        self.assertEqual(data["channels"], CHANNELS + ["amazon_display", "amazon_search"])
        np.testing.assert_allclose(
            frame.amazon_display_spend,
            frame.amazon_display_exposure * 14 / 1000,
        )
        np.testing.assert_allclose(frame.amazon_search_spend, frame.amazon_search_exposure * 2.0)
        no_budget = frame.amazon_search_budget_activity == 0
        self.assertTrue((frame.loc[no_budget, "amazon_search_exposure"] == 0).all())
        self.assertTrue((frame.loc[no_budget, "amazon_search_spend"] == 0).all())
        mu, _, _ = predictions(truth_samples(truth), data)
        np.testing.assert_allclose(mu[0], frame.truth_log_sales_mean, atol=1e-10)

        combined_frame, combined_truth = generate_data(
            amazon=True, funnel=True, hierarchical_media=True
        )
        combined_data = prepare_data(combined_frame)
        combined_data["hierarchical_media"] = True
        combined_mu, _, _ = predictions(truth_samples(combined_truth), combined_data)
        np.testing.assert_allclose(
            combined_mu[0], combined_frame.truth_log_sales_mean, atol=1e-10
        )
        combined_model = build_model(combined_data, 130)
        self.assertTrue(np.isfinite(
            combined_model.compile_dlogp()(combined_model.initial_point())
        ).all())

        model = build_model(data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        altered = dict(data)
        altered["amazon_search_counts"] = data["amazon_search_counts"].copy()
        altered["amazon_search_counts"][:, 130:] += 99999
        altered["amazon_click_counts"] = data["amazon_click_counts"].copy()
        altered["amazon_click_counts"][:, 130:] += 999
        changed = build_model(altered, 130)
        np.testing.assert_allclose(
            model.compile_logp()(model.initial_point()),
            changed.compile_logp()(changed.initial_point()),
        )
        context = planning_context(truth_samples(truth), data, 130)
        zero_plan = np.zeros(len(data["channels"]))
        self.assertEqual(scenario_amazon_search_spend(context, zero_plan)[0], 0)
        active_plan = data["ratio"][:, 104:130, :].mean(axis=(0, 1))
        self.assertGreater(scenario_amazon_search_spend(context, active_plan)[0], 0)

    def test_funnel_graph_and_mediator_holdout_exclusion(self):
        frame, truth = generate_data(funnel=True)
        data = prepare_data(frame)
        model = build_model(data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        altered = dict(data)
        altered["search_counts"] = data["search_counts"].copy()
        altered["search_counts"][:, 130:] += 99999
        altered["click_counts"] = data["click_counts"].copy()
        altered["click_counts"][:, 130:] += 999
        changed = build_model(altered, 130)
        np.testing.assert_allclose(model.compile_logp()(model.initial_point()), changed.compile_logp()(changed.initial_point()))
        with model:
            prior = pm.sample_prior_predictive(draws=2, random_seed=77, var_names=[
                "intercept", "product_effect", "tier_effect", "interaction", "control_beta",
                "media_beta", "decay", "half_saturation", "sigma", "search_intercept",
                "search_ctv_beta", "search_control_beta", "search_sigma", "click_max_rate",
                "click_budget_half", "log_sales_mean",
            ])
        samples = extract_samples(prior.prior)
        mu, _, _ = predictions(samples, data)
        np.testing.assert_allclose(mu, prior.prior["log_sales_mean"].values.reshape(2, -1), atol=1e-9)

    def test_funnel_planning_recomputes_mediators(self):
        frame, truth = generate_data(funnel=True)
        data = prepare_data(frame)
        samples = truth_samples(truth)
        context = planning_context(samples, data, 130)
        ratio = np.array([1.2, 0.8, 1.6, 0.9, 1.1])
        altered = dict(data)
        altered["ratio"] = data["ratio"].copy()
        altered["ratio"][:, 130:143, :] = ratio
        altered["paid_budget_ratio"] = data["paid_budget_ratio"].copy()
        altered["paid_budget_ratio"][:, 130:143] = ratio[4]
        mu, _, _ = predictions(samples, altered)
        selected = (data["week"] >= 130) & (data["week"] < 143)
        expected = np.exp(mu[0, selected] + truth["sigma"]**2 / 2).sum()
        np.testing.assert_allclose(scenario_units(context, ratio)[0], expected, rtol=1e-12)

    def test_combined_generator_preserves_uncapped_funnel(self):
        funnel, _ = generate_data(funnel=True)
        combined, _ = generate_data(funnel=True, stockouts=True)
        np.testing.assert_array_equal(combined.latent_demand_units, funnel.units)
        for column in funnel.columns.difference(["units"]):
            self.assertTrue(combined[column].equals(funnel[column]), column)
        np.testing.assert_array_equal(combined.units, np.minimum(combined.latent_demand_units, combined.inventory_available))
        self.assertTrue((combined.paid_search_exposure <= combined.branded_search_volume).all())

    def test_hierarchical_media_generator_and_prediction_equivalence(self):
        frame, truth = generate_data(funnel=True, stockouts=True, hierarchical_media=True)
        data = prepare_data(frame)
        data["hierarchical_media"] = True
        samples = truth_samples(truth)
        np.testing.assert_allclose(np.exp(np.log(samples["product_media_beta"]).mean(axis=1)), samples["media_beta"], atol=1e-12)
        mu, _, _ = predictions(samples, data)
        np.testing.assert_allclose(mu[0], frame.truth_log_sales_mean, atol=1e-10)
        model = build_model(data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        with model:
            prior = pm.sample_prior_predictive(draws=2, random_seed=73, var_names=[
                "intercept", "product_effect", "tier_effect", "interaction", "control_beta",
                "media_beta", "decay", "half_saturation", "sigma", "product_media_beta",
                "media_product_sd", "search_intercept", "search_ctv_beta", "search_control_beta",
                "search_sigma", "click_max_rate", "click_budget_half", "log_sales_mean",
            ])
        draws = extract_samples(prior.prior)
        computed, _, _ = predictions(draws, data)
        np.testing.assert_allclose(computed, prior.prior["log_sales_mean"].values.reshape(2, -1), atol=1e-8)
        self.assertTrue(np.all(draws["product_media_beta"] > 0))
        np.testing.assert_allclose(np.exp(np.log(draws["product_media_beta"]).mean(axis=1)), draws["media_beta"], rtol=1e-10)

    def test_media_shrinkage_zero_scale_full_pooling_limit(self):
        data = dict(self.data)
        data["hierarchical_media"] = True
        model = build_model(data, 130)
        pooled = pm.do(model, {"media_product_sd": np.zeros(5)})
        with pooled:
            prior = pm.sample_prior_predictive(draws=2, random_seed=88, var_names=["media_beta", "product_media_beta"])
        centers = prior.prior["media_beta"].values.reshape(2, 5)
        product_values = prior.prior["product_media_beta"].values.reshape(2, 4, 5)
        np.testing.assert_allclose(product_values, np.broadcast_to(centers[:, None, :], (2, 4, 5)), atol=1e-12)

    def test_hierarchical_media_capped_funnel_planning(self):
        frame, truth = generate_data(funnel=True, stockouts=True, hierarchical_media=True)
        data = prepare_data(frame)
        samples = truth_samples(truth)
        context = planning_context(samples, data, 130)
        ratio = np.array([1.2, 0.8, 1.6, 0.9, 1.1])
        altered = dict(data)
        altered["ratio"] = data["ratio"].copy()
        altered["ratio"][:, 130:143, :] = ratio
        altered["paid_budget_ratio"] = data["paid_budget_ratio"].copy()
        altered["paid_budget_ratio"][:, 130:143] = ratio[4]
        mu, _, _ = predictions(samples, altered)
        selected = (data["week"] >= 130) & (data["week"] < 143)
        expected = expected_fulfilled_units(mu[0, selected], truth["sigma"], data["inventory"][selected]).sum()
        np.testing.assert_allclose(scenario_units(context, ratio)[0], expected, rtol=1e-12)

    def test_persistent_brand_joint_graph_and_holdout_exclusion(self):
        frame, truth = generate_data(funnel=True, stockouts=True, hierarchical_media=True, brand_state=True)
        data = prepare_data(frame)
        data["hierarchical_media"] = True
        samples = truth_samples(truth)
        state, score, effect = brand_predictions(samples, data)
        self.assertEqual(state.shape, (1, len(truth["brand_initial"]), len(data["weeks"])))
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.isfinite(effect).all())
        self.assertTrue(np.all((score > 0) & (score < 100)))
        self.assertTrue(frame.distribution_volume_weighted_pct.between(0, 100).all())
        np.testing.assert_allclose(frame.sponsorship_spend, frame.sponsorship_impressions * 18 / 1000)
        repeated, repeated_truth = generate_data(
            funnel=True, stockouts=True, hierarchical_media=True, brand_state=True
        )
        np.testing.assert_array_equal(frame.brand_equity_score, repeated.brand_equity_score)
        self.assertEqual(repeated_truth["brand_process_sigma"], truth["brand_process_sigma"])
        expected_margin = (
            frame.list_price.to_numpy() * (1 - frame.discount.to_numpy())
            * CONTRIBUTION_MARGIN_RATE[data["product"], data["tier"]]
        )
        np.testing.assert_allclose(data["contribution_per_unit"], expected_margin)
        changed_scores = dict(data)
        changed_scores["brand_observed_logit"] = data["brand_observed_logit"].copy()
        changed_scores["brand_observed_logit"][:, 130:] += 20
        changed_state, _, _ = brand_predictions(samples, changed_scores)
        np.testing.assert_allclose(state, changed_state)
        model = build_model(data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        altered = dict(data)
        altered["brand_observed_logit"] = data["brand_observed_logit"].copy()
        altered["brand_observed_logit"][:, 130:] += 20
        changed = build_model(altered, 130)
        np.testing.assert_allclose(model.compile_logp()(model.initial_point()), changed.compile_logp()(changed.initial_point()))
        with model:
            prior = pm.sample_prior_predictive(draws=2, random_seed=92, var_names=[
                "intercept", "product_effect", "tier_effect", "interaction", "control_beta",
                "media_beta", "decay", "half_saturation", "sigma", "product_media_beta",
                "media_product_sd", "search_intercept", "search_ctv_beta", "search_control_beta",
                "search_sigma", "click_max_rate", "click_budget_half", "brand_intercept",
                "brand_product_effect", "brand_initial", "brand_media_beta", "brand_sponsorship_beta",
                "brand_distribution_beta", "brand_persistence", "brand_process_sigma", "brand_measurement_sigma",
                "brand_sales_beta", "log_sales_mean",
            ])
        draws = extract_samples(prior.prior)
        mu, _, _ = predictions(draws, data)
        np.testing.assert_allclose(mu, prior.prior["log_sales_mean"].values.reshape(2, -1), atol=1e-8)

    def test_brand_sponsorship_lag_and_six_line_budget(self):
        frame, truth = generate_data(funnel=True, stockouts=True, hierarchical_media=True, brand_state=True)
        data = prepare_data(frame)
        samples = truth_samples(truth)
        _, _, original_effect = brand_predictions(samples, data)
        changed = dict(data)
        changed["sponsorship_ratio"] = data["sponsorship_ratio"].copy()
        changed["sponsorship_ratio"][:, 130] += 10
        _, _, new_effect = brand_predictions(samples, changed)
        np.testing.assert_allclose(original_effect[:, :, 130], new_effect[:, :, 130], atol=1e-12)
        self.assertTrue(np.all(new_effect[:, :, 131] > original_effect[:, :, 131]))
        context = planning_context(samples, data, 130)
        ratio = np.array([1.2, 0.8, 1.6, 0.9, 1.1, 2.0])
        changed["ratio"] = data["ratio"].copy()
        changed["ratio"][:, 130:143, :] = ratio[:5]
        changed["paid_budget_ratio"] = data["paid_budget_ratio"].copy()
        changed["paid_budget_ratio"][:, 130:143] = ratio[4]
        changed["sponsorship_ratio"][:, 130:143] = ratio[5]
        mu, _, _ = predictions(samples, changed)
        selected = (data["week"] >= 130) & (data["week"] < 143)
        expected = expected_fulfilled_units(mu[0, selected], truth["sigma"], data["inventory"][selected]).sum()
        np.testing.assert_allclose(scenario_units(context, ratio)[0], expected, rtol=1e-12)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            result = optimization_outputs(samples, data, truth, 130, output)
            import pandas as pd
            allocation = pd.read_csv(output / "budget_allocation.csv")
            self.assertEqual(len(allocation), 6)
            self.assertIn("sponsorship", allocation.channel.tolist())
            self.assertTrue(result["pass"])
            profit_allocation = pd.read_csv(output / "profit_budget_allocation.csv")
            self.assertEqual(len(profit_allocation), 6)
            self.assertTrue(result["profit_optimization"]["pass"])
            self.assertAlmostEqual(
                profit_allocation.expected_profit_13week_spend.sum(),
                result["total_budget"], places=5,
            )
            self.assertAlmostEqual(
                profit_allocation.risk_adjusted_13week_spend.sum(),
                result["total_budget"], places=5,
            )
            self.assertEqual(result["profit_optimization"]["cvar_tail_probability"], 0.05)
            self.assertEqual(result["profit_optimization"]["risk_weight_on_cvar"], 0.5)
            self.assertLessEqual(
                result["profit_optimization"]["predictive_profit_cvar_5_risk_allocation"],
                result["profit_optimization"]["expected_profit_at_risk_allocation"],
            )

    def test_combined_gradient_holdout_and_capped_planning(self):
        frame, truth = generate_data(funnel=True, stockouts=True)
        data = prepare_data(frame)
        model = build_model(data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        altered = dict(data)
        altered["log_units"] = data["log_units"].copy()
        altered["log_units"][data["week"] >= 130] += 20
        altered["search_counts"] = data["search_counts"].copy()
        altered["search_counts"][:, 130:] += 99999
        altered["click_counts"] = data["click_counts"].copy()
        altered["click_counts"][:, 130:] += 999
        changed = build_model(altered, 130)
        np.testing.assert_allclose(model.compile_logp()(model.initial_point()), changed.compile_logp()(changed.initial_point()))
        samples = truth_samples(truth)
        context = planning_context(samples, data, 130)
        ratio = np.array([1.2, 0.8, 1.6, 0.9, 1.1])
        scenario_data = dict(data)
        scenario_data["ratio"] = data["ratio"].copy()
        scenario_data["ratio"][:, 130:143, :] = ratio
        scenario_data["paid_budget_ratio"] = data["paid_budget_ratio"].copy()
        scenario_data["paid_budget_ratio"][:, 130:143] = ratio[4]
        mu, _, _ = predictions(samples, scenario_data)
        selected = (data["week"] >= 130) & (data["week"] < 143)
        expected = expected_fulfilled_units(mu[0, selected], truth["sigma"], data["inventory"][selected]).sum()
        np.testing.assert_allclose(scenario_units(context, ratio)[0], expected, rtol=1e-12)
        self.assertLessEqual(float(expected), float(context["inventory"].sum()) + 1e-6)

    def test_numpy_matches_pytensor_and_holdout_is_excluded(self):
        model = build_model(self.data, 130)
        self.assertTrue(np.isfinite(model.compile_dlogp()(model.initial_point())).all())
        with model:
            prior = pm.sample_prior_predictive(
                draws=2, random_seed=91,
                var_names=[
                    "intercept", "product_effect", "tier_effect", "interaction",
                    "control_beta", "media_beta", "decay", "half_saturation",
                    "sigma", "log_sales_mean",
                ],
            )
        samples = extract_samples(prior.prior)
        mu, _, _ = predictions(samples, self.data)
        np.testing.assert_allclose(mu, prior.prior["log_sales_mean"].values.reshape(2, -1), atol=1e-10)
        changed = dict(self.data)
        changed["log_units"] = self.data["log_units"].copy()
        changed["log_units"][self.data["week"] >= 130] += 20
        altered_model = build_model(changed, 130)
        np.testing.assert_allclose(
            model.compile_logp()(model.initial_point()),
            altered_model.compile_logp()(altered_model.initial_point()),
        )

    def test_planning_keeps_history_and_budget_constraints(self):
        context = planning_context(truth_samples(self.truth), self.data, 130)
        ratio = np.array([1.2, 0.8, 1.6, 0.9, 1.1])
        exposures = self.data["ratio"].copy()
        exposures[:, 130:143, :] = ratio
        features = media_features(exposures, self.truth["decay"], self.truth["half_saturation"])
        selected = (self.data["week"] >= 130) & (self.data["week"] < 143)
        mu = context["base"][0] + features[self.data["product"][selected], self.data["week"][selected]] @ self.truth["media_beta"]
        expected = np.exp(mu + self.truth["sigma"]**2 / 2).sum()
        np.testing.assert_allclose(scenario_units(context, ratio)[0], expected, rtol=1e-12)
        shares = np.repeat(0.2, 5)
        costs = np.array([16000, 20000, 14000, 18000, 17000])
        optimum = solve_allocation(context, 150000, costs, shares, 0.4 * shares, 1.8 * shares)
        self.assertAlmostEqual(float(optimum.sum()), 1.0, places=7)
        self.assertTrue(np.all(optimum >= 0.4 * shares - 1e-7))
        self.assertTrue(np.all(optimum <= 1.8 * shares + 1e-7))
        self.assertGreaterEqual(
            float(scenario_units(context, optimum * 150000 / costs)[0]),
            float(scenario_units(context, shares * 150000 / costs)[0]) - 1,
        )


if __name__ == "__main__":
    unittest.main()