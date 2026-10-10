import argparse
import ast
import json
from pathlib import Path
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parent
INK = colors.HexColor("#202A2D")
TEAL = colors.HexColor("#007F73")
MUTED = colors.HexColor("#536166")
PALE = colors.HexColor("#EDF5F3")
RULE = colors.HexColor("#C9D4D1")


def paragraph(pdf, text, x, y, width, size=9.2, leading=12.1, color=INK):
    style = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=size, leading=leading,
        textColor=color, alignment=TA_LEFT, spaceAfter=0,
    )
    block = Paragraph(text, style)
    _, height = block.wrap(width, 1000)
    block.drawOn(pdf, x, y - height)
    return y - height


def heading(pdf, text, x, y, width):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(0.6)
    pdf.line(x, y, x + width, y)
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 10.2)
    pdf.drawString(x, y - 15, text)
    return y - 23


def test_count(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def draw_comparison_page(pdf, page_width, page_height, margin):
    available = page_width - 2 * margin
    gap = 22
    column_width = (available - gap) / 2
    left_x = margin
    right_x = margin + column_width + gap
    y = page_height - 36
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(margin, y, "LEARNING BRIEF  |  ALTERNATIVE METHODS")
    y -= 31
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 24)
    pdf.drawString(margin, y, "Alternative Methods: Compare and Contrast")
    y -= 13
    y = paragraph(
        pdf,
        "A frequentist Ridge model is a useful benchmark, but it is not a drop-in equivalent "
        "to this Bayesian hierarchical MMM.",
        margin, y, available, size=10.1, leading=13,
    ) - 17

    left_y = heading(pdf, "BAYESIAN HIERARCHICAL MMM", left_x, y, column_width)
    for text in [
        "<b>What it means.</b> Parameters are represented by probability distributions updated "
        "with observed data and prior assumptions. Posterior draws describe uncertainty about "
        "effects and predictions, conditional on the model and data.",
        "<b>What this prototype adds.</b> Product effects partially pool toward shared channel "
        "centers. The model also represents nonlinear media response, a search/click funnel, "
        "censored sales, persistent brand state, and uncertainty for downstream decisions.",
        "<b>Costs and caveats.</b> MCMC takes longer and requires convergence diagnostics. "
        "Results depend on model structure and priors; posterior uncertainty does not by itself "
        "establish causal effects or protect against misspecification.",
    ]:
        left_y = paragraph(pdf, text, left_x, left_y, column_width, size=9.0, leading=12) - 10

    right_y = heading(pdf, "FREQUENTIST RIDGE REGRESSION", right_x, y, column_width)
    for text in [
        "<b>What it means.</b> Coefficients are treated as fixed but unknown. Inference asks how "
        "estimates or procedures would vary across repeated samples. Ridge fits coefficients "
        "with an L2 penalty, usually tuned by time-aware validation.",
        "<b>What it buys.</b> Fast fitting, stable estimates under correlated predictors, and a "
        "simple predictive benchmark. It can show whether the more elaborate model materially "
        "improves held-out forecasts.",
        "<b>What it gives up.</b> Ordinary Ridge shrinks toward zero, not toward a learned "
        "population effect; hierarchy needs extra structure. It gives point estimates, so "
        "uncertainty needs methods such as bootstrap. Standard Ridge also allows negative "
        "coefficients unless constrained.",
    ]:
        right_y = paragraph(pdf, text, right_x, right_y, column_width, size=9.0, leading=12) - 10

    y = min(left_y, right_y) - 2
    y = heading(pdf, "AT A GLANCE", margin, y, available)
    rows = [
        ("Pooling", "Bayesian model: explicit partial pooling across products.",
         "Ridge: global shrinkage; group-specific pooling must be designed."),
        ("Uncertainty", "Bayesian model: posterior and predictive scenarios feed profit/CVaR.",
         "Ridge: bootstrap intervals and financial scenarios need additional work."),
        ("Model structure", "Bayesian model: supports the prototype's funnel, censoring, and brand state.",
         "Ridge: nonlinear features can be engineered, but these mechanisms need separate treatment."),
        ("Best fit", "Bayesian model: richer inference and uncertainty-aware decisions.",
         "Ridge: rapid baseline, prediction benchmark, and simpler repeated refits."),
    ]
    label_width = 76
    half_gap = 12
    cell_width = (available - label_width - 2 * half_gap) / 2
    row_y = y
    for label, bayes_text, ridge_text in rows:
        pdf.setFillColor(PALE)
        pdf.roundRect(margin, row_y - 35, label_width - 7, 35, 3, stroke=0, fill=1)
        paragraph(pdf, f"<b>{label}</b>", margin + 6, row_y - 11, label_width - 18,
                  size=8.2, leading=10)
        paragraph(pdf, bayes_text, margin + label_width, row_y - 3, cell_width,
                  size=8.2, leading=10.2)
        paragraph(pdf, ridge_text, margin + label_width + cell_width + half_gap, row_y - 3,
                  cell_width, size=8.2, leading=10.2)
        pdf.setStrokeColor(RULE)
        pdf.setLineWidth(0.4)
        pdf.line(margin, row_y - 39, margin + available, row_y - 39)
        row_y -= 43

    y = row_y - 2
    y = heading(pdf, "A FAIR TEST FOR THIS MMM", margin, y, available)
    y = paragraph(
        pdf,
        "Fit Ridge to the same training window and inputs, using time-based cross-validation "
        "to choose its penalty. Keep all feature construction and scaling training-only; compare "
        "both models on the same untouched holdout for forecast error and calibration. Then compare "
        "attribution and optimized budgets: similar RMSE alone does not mean the decisions agree.",
        margin, y, available, size=9.0, leading=12,
    ) - 8
    y = paragraph(
        pdf,
        "<b>Practical recommendation:</b> use the Bayesian model to compare unit, expected-profit, "
        "and CVaR allocations; replace illustrative margins with real costs before business use, "
        "and add Ridge as a fast predictive benchmark. In a linear Gaussian model with fixed Gaussian "
        "priors, Ridge corresponds to a Bayesian MAP estimate, but not the full posterior uncertainty "
        "or this prototype's hierarchical structure.",
        margin, y, available, size=9.0, leading=12,
    )
    if y < 49:
        raise RuntimeError(f"Comparison-page layout overflow: body ends at {y:.1f} points.")
    pdf.setStrokeColor(RULE)
    pdf.line(margin, 38, page_width - margin, 38)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.2)
    pdf.drawString(margin, 25, "Interpretation is conditional on data, model structure, and validation design.")
    pdf.drawRightString(page_width - margin, 25, "2 / 6")


def draw_appendix_page(pdf, page_width, page_height, margin, appendix, title,
                       introduction, sections, page_number):
    available = page_width - 2 * margin
    y = page_height - 36
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(margin, y, f"IMPLEMENTATION BLUEPRINT  |  APPENDIX {appendix}")
    y -= 31
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 22)
    pdf.drawString(margin, y, title)
    y -= 14
    y = paragraph(pdf, introduction, margin, y, available, size=9.6, leading=12.4) - 12
    for section_title, paragraphs in sections:
        y = heading(pdf, section_title.upper(), margin, y, available)
        for text in paragraphs:
            y = paragraph(pdf, text, margin, y, available, size=8.8, leading=11.2) - 6
        y -= 2
    if y < 49:
        raise RuntimeError(f"Appendix {appendix} layout overflow: body ends at {y:.1f} points.")
    pdf.setStrokeColor(RULE)
    pdf.line(margin, 38, page_width - margin, 38)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.2)
    pdf.drawString(margin, 25, "Blueprint detail is a starting point; validate definitions and assumptions with data owners.")
    pdf.drawRightString(page_width - margin, 25, f"{page_number} / 6")


def build_pdf(output, results_dir=ROOT / "outputs_combined"):
    results_dir = Path(results_dir)
    metrics = json.loads((results_dir / "metrics.json").read_text())
    config = json.loads((results_dir / "run_config.json").read_text())
    hierarchical = config.get("hierarchical_media", False)
    brand_state = config.get("brand_state", False)
    variants = ["outputs", "outputs_stockouts", "outputs_funnel", "outputs_combined"]
    if not all(json.loads((ROOT / folder / "metrics.json").read_text())["overall_pass"] for folder in variants):
        raise ValueError("This verified-run summary requires passing saved reference variants.")
    if not metrics["overall_pass"]:
        raise ValueError("The selected summary run must pass its recorded gates.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=letter, pageCompression=1)
    pdf.setTitle("Synthetic MMM - Summary, Alternative Methods, and Real-Data Implementation Blueprint")
    pdf.setAuthor("Synthetic MMM Reference Project")
    pdf.setSubject("Summary, alternative methods, detailed workflow, assumptions, and real-data blueprint")
    pdf.setCreator("build_summary_pdf.py / ReportLab")
    width, height = letter
    margin = 38
    available = width - 2 * margin
    y = height - 36
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(margin, y, "LEARNING BRIEF  |  VERIFIED SYNTHETIC REFERENCE  |  08 OCT 2026")
    y -= 28
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 24)
    pdf.drawString(margin, y, "Bayesian Marketing Mix Prototype")
    y -= 12
    y = paragraph(
        pdf,
        "<b>Goal:</b> learn how marketing affects synthetic sales and brand, then forecast and plan budgets.",
        margin, y, available, size=10.1, leading=13,
    ) - 8
    y = paragraph(
        pdf,
        f"<b>Built and tested:</b> {config['weeks']} weeks x 4 products x 3 price tiers ({config['weeks'] * 12:,} observations); "
        f"{config['train_weeks']} training weeks and {config['holdout_weeks']} held-out weeks. "
        "Baseline, stockout-only, funnel-only, and combined variants passed predefined gates; "
        f"{test_count(ROOT / 'test_mmm.py')} deterministic regression tests passed.",
        margin, y, available,
    ) - 8
    pdf.setFillColor(PALE)
    pdf.rect(margin, y - 29, available, 29, stroke=0, fill=1)
    paragraph(
        pdf,
        "<b>Methods implemented:</b> Bayesian partial pooling (shrinkage), ad carryover, search funnel, "
        "Kalman-filtered brand scores, and profit/CVaR budget optimization.",
        margin + 8, y - 7, available - 16, size=8.7, leading=11,
    )
    y -= 35 if brand_state else 43

    gap = 19
    left_width = 319
    right_width = available - left_width - gap
    left_x = margin
    right_x = margin + left_width + gap
    left_y = heading(pdf, "MODEL: WHAT IT REPRESENTS", left_x, y, left_width)
    media_text = (
        "<b>Media response and shrinkage.</b> Display, Video, Social, and CTV have estimated "
        "adstock (13 lag positions) and saturation; paid clicks have saturation only. Social's "
        "Hill power is 2; the others use 1. Product responses partially pool toward learned "
        "channel centers, with estimated deviation scales. Tier media slopes, decay, and "
        "half-saturation remain shared. Comparative benefits versus no pooling are untested."
        if hierarchical else
        "<b>Media response.</b> Display, Video, Social, and CTV each have estimated geometric carryover "
        "(current week + 12 lags), saturation, and nonnegative log-sales coefficients. Social has "
        "an S-shaped Hill curve; the others have diminishing-return Hill curves. Paid clicks have "
        "saturation but no adstock. Response coefficients are shared across products/tiers."
    )
    if brand_state:
        blocks = [
            "<b>Demand and hierarchy.</b> Product, tier, and product x tier baseline effects partially "
            "pool. Controls: price, macro, base-price growth, promotion, post-promotion dip, PrimeDays, "
            "and annual seasonality. No random-walk baseline.",
            "<b>Media shrinkage.</b> Positive product responses partially pool toward learned channel "
            "centers. Display, Video, Social, and CTV have estimated adstock and saturation; paid "
            "clicks have no adstock. Social is S-shaped. Tier slopes and transforms remain shared; "
            "no-pooling superiority is untested.",
            "<b>Funnel.</b> CTV, organic demand, and controls drive branded search. Search has "
            "log-Normal measurement noise; clicks are Binomial with spend-dependent capture. "
            "Sales use latent mean clicks, not observed holdout outcomes; CTV also has a direct effect.",
            "<b>Kalman brand state.</b> A jointly estimated 0-100 score follows a linear-Gaussian "
            "logit state process with shocks and noisy score measurements. A Kalman filter updates "
            "states from training scores; nonlinear sales are not Kalman observations.",
            "<b>New business inputs.</b> Sponsorship has impressions and separate spend at a synthetic "
            "$18 CPM. Weekly product distribution is the volume-weighted % of available stores/market "
            "carrying the product, not store count. It is fixed in planning; no separate SKU-breadth variable.",
            "<b>Sales and decisions.</b> Censored log-Normal sales = min(demand, inventory). Six spend "
            "lines include sponsorship. Compare units, expected contribution-profit, and 50/50 "
            "expected/5%-CVaR objectives over 13 weeks. Margin rates are illustrative, not observed "
            "costs; budget and product proportions are fixed. CTV attribution avoids double counting.",
        ]
    else:
        blocks = [
        "<b>Demand and hierarchy.</b> Latent log sales combine global, product, tier, and product x tier "
        "intercepts with partial pooling. Controls cover price, macro demand, base-price growth, "
        "promotion, post-promotion dip, PrimeDays, and annual seasonality. No random-walk baseline.",
        media_text,
        "<b>Funnel mediation.</b> CTV plus observed organic demand and controls drive latent branded "
        "search. Search has log-Normal measurement noise; clicks are Binomial conditional on searches, "
        "with spend-dependent capture. Sales use latent mean clicks, not held-out mediator outcomes. "
        "CTV also has a direct sales effect.",
        "<b>Inventory and sales.</b> Observed sales = min(demand, supplied inventory). A censored "
        "log-Normal likelihood learns demand behind capped sales. Forecasts distinguish latent demand "
        "from fulfillment; scenario evaluation uses E[min(demand, cap)], not a clipped mean.",
        "<b>Attribution and decisions.</b> An explicit accounting rule assigns CTV direct plus mediated "
        "log lift and paid search its no-CTV click lift. Contributions reconcile to sales without "
        "double counting. A 13-week optimizer maximizes expected fulfilled units at fixed budget, "
        "constant weekly shares, fixed product proportions, and channel bounds, retaining carryover.",
        ]
    for text in blocks:
        left_y = paragraph(pdf, text, left_x, left_y, left_width) - 6

    right_y = heading(pdf, "COMBINED RUN: EVIDENCE", right_x, y, right_width)
    sampling = metrics["sampling"]
    forecast = metrics["forecast"]
    recovery = metrics["recovery"]
    funnel = metrics["funnel"]
    optimization = metrics["optimization"]
    regret_text = (
        "below 0.0001%" if optimization["truth_relative_regret"] < 0.000001
        else f"{optimization['truth_relative_regret']:.4%}"
    )
    evidence = [
        ("Sampling reliability", f"R-hat {sampling['max_rhat']:.4f}; {sampling['divergences']} divergences"),
        ("Holdout log-sales error", f"{forecast['rmse_improvement']:.1%} lower RMSE*"),
        ("90% predictive coverage", f"Demand {forecast['latent_coverage_90']:.1%}; sales {forecast['coverage_90']:.1%}"),
        ("90% mediator coverage", f"Search {funnel['search_holdout_coverage_90']:.1%}; clicks {funnel['clicks_holdout_coverage_90']:.1%}"),
        ("Known media-effect recovery", (
            f"{recovery['product_media_truths_covered']}/20 product intervals; max error {recovery['max_product_media_relative_error']:.2%}"
            if hierarchical else
            f"{recovery['media_truths_covered']}/5 intervals cover truth; max error {recovery['max_media_relative_error']:.2%}"
        )),
        ("CTV mediated lift recovery", f"{funnel['ctv_mediated_log_lift_relative_error']:.1%} relative error"),
        ("13-week allocation scenario", f"${optimization['total_budget']/1_000_000:.2f}m budget; +{optimization['predicted_lift']:.2%} units"),
    ]
    if brand_state:
        brand = metrics["brand_state"]
        profit = optimization["profit_optimization"]
        evidence[3] = (
            "Funnel coverage / CTV",
            f"Search {funnel['search_holdout_coverage_90']:.1%}; clicks {funnel['clicks_holdout_coverage_90']:.1%}; "
            f"CTV lift error {funnel['ctv_mediated_log_lift_relative_error']:.1%}",
        )
        evidence.insert(5, (
            "Brand score holdout",
            f"RMSE {brand['holdout_rmse_score_points']:.2f} points; {brand['holdout_coverage_90']:.1%} coverage",
        ))
        evidence.insert(6, (
            "Persistence / baseline demand",
            f"{brand['estimated_persistence']:.3f} weekly; +{np.expm1(brand['estimated_sales_log_effect_per_10_points']):.1%} per +10 prior score pts",
        ))
        evidence.pop(7)
        evidence.insert(7, (
            "5% profit CVaR",
            f"${profit['predictive_profit_cvar_5_baseline']/1_000_000:.1f}m baseline; "
            f"${profit['predictive_profit_cvar_5_risk_allocation']/1_000_000:.1f}m risk allocation",
        ))
        evidence[8] = (
            "13-week decisions",
            f"${optimization['total_budget']/1_000_000:.2f}m; units +{optimization['predicted_lift']:.2%}; "
            f"profit +${(profit['expected_profit_optimized']-profit['expected_profit_baseline'])/1_000_000:.1f}m",
        )
    for label, value in evidence:
        right_y = paragraph(pdf, label, right_x, right_y, right_width, size=8.0, leading=10.2, color=MUTED)
        right_y = paragraph(pdf, f"<b>{value}</b>", right_x, right_y - 1, right_width, size=9.0, leading=11.5) - 7
    right_y = paragraph(
        pdf,
        "*Versus a product-tier mean log-sales benchmark. All gates passed. Budget conserved; "
        f"unit-objective shortfall is {regret_text} versus the "
        "known-truth best found under the same constraints. "
        "Six local starts are not a global-optimum proof.",
        right_x, right_y, right_width, size=8.0, leading=10.2, color=MUTED,
    )

    y = min(left_y, right_y) - 9
    y = heading(pdf, "NEXT STEPS: MMM BEST PRACTICE AND QUANT RESEARCH", margin, y, available)
    roadmap = [
        "<b>1. Identification and experiments.</b> Calibrate with geo/lift studies; address spend "
        "endogeneity, mediator confounding, and measurement error before causal interpretation.",
        "<b>2. Robustness and backtesting.</b> Repeat seeds and leakage-free rolling/untouched holdouts; "
        "add simulation-based calibration, prior sensitivity, and misspecification stress tests.",
        "<b>3. Real-data pilots.</b> Audit exposure, spend, sales, and lag alignment; use stronger "
        "forecast benchmarks and explicit future-control, organic-demand, and inventory scenarios.",
        "<b>4. Quant decision layer.</b> Optimize margin/profit over posterior scenarios with tail-loss "
        "(CVaR) risk constraints; research dynamic pacing, uncertain CPM/CPC, and replenishment.",
        "<b>5. Dynamic response research.</b> Test time-varying/hierarchical effects, alternative lag "
        "and saturation models, and state-space baselines; justify complexity by decision value.",
    ]
    if brand_state:
        roadmap = [
            "<b>1. Identification.</b> Use geo/lift evidence; address spend endogeneity, mediation "
            "confounding, brand construct validity, and physical distribution effects.",
            "<b>2. Robustness.</b> Repeat seeds, untouched/rolling holdouts, simulation-based "
            "calibration, prior sensitivity, and misspecification stress tests.",
            "<b>3. Real-data pilots.</b> Audit score surveys, exposures, cost, sales, and weighted "
            "distribution; specify plausible future demand, supply, and coverage scenarios.",
            "<b>4. Quant decisions.</b> Add profit and posterior tail-loss (CVaR) constraints; "
            "research longer-horizon brand value, pacing, uncertain costs, and replenishment.",
            "<b>5. Dynamic models.</b> Test richer brand shocks, nonlinear state filters, time-varying "
            "response and lag forms, and joint brand/supply dynamics; require decision value.",
        ]
    for text in roadmap:
        y = paragraph(pdf, text, margin, y, available, size=9.0, leading=11.5) - 4
    y -= 4
    boundary = (
        "<b>Boundary:</b> one deliberately informative matched synthetic seed; not independent "
        "real-data evidence. Future inputs/coverage/inventory and margins are supplied assumptions. "
        "Kalman filtering uses scores, not nonlinear sales observations. No direct distribution "
        "constraint or separate assortment breadth. Attribution is a convention; 13 weeks omit later "
        "brand value. <b>Not real-world causal validation or deployment advice.</b>"
        if brand_state else
        "<b>Boundary:</b> one deliberately informative synthetic seed with matching model assumptions. "
        "Reference variants preserve comparability; the paired holdout is not independent new evidence. "
        "Future drivers/inventory are supplied, not forecast. Attribution is a convention; the objective "
        "is units, not profit. This is <b>not real-world causal validation or a deployment recommendation.</b>"
    )
    y = paragraph(
        pdf,
        boundary,
        margin, y, available, size=8.3, leading=10.5, color=MUTED,
    )
    if y < 49:
        raise RuntimeError(f"One-page layout overflow: body ends at {y:.1f} points.")
    pdf.setStrokeColor(RULE)
    pdf.line(margin, 38, width - margin, 38)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.2)
    command = "python run_mmm.py --funnel --stockouts" + (" --hierarchical-media" if hierarchical else "")
    if brand_state:
        pdf.drawString(margin, 25, f"Source: {results_dir.name} | Kalman state + synthetic profit | See README")
    else:
        pdf.drawString(margin, 25, "Reproduce: " + command + " | Source: saved run metrics")
    pdf.drawRightString(width - margin, 25, "1 / 6")
    pdf.showPage()
    draw_comparison_page(pdf, width, height, margin)
    pdf.showPage()

    appendices = [
        (
            "A", "Data Contract and Identification",
            "Start with stable entities, explicit measurement definitions, and strict time alignment. "
            "The synthetic panel is unusually clean; a real-data implementation should treat data "
            "quality and identification as first-class model components.",
            [
                ("Current analysis grain", [
                    "Sales are weekly at product x price-tier grain: 156 weeks, four products, three tiers. "
                    "The current fit uses the first 130 weeks and reserves the final 26 weeks for a conditional "
                    "holdout. Brand scores, sponsorship, distribution, search, and clicks are product-week "
                    "quantities and must not be duplicated across the three price tiers.",
                    "The current generator carries sales, list price, discount, nominal/base price, macro and "
                    "event controls; five media exposures/spend series and reference exposures; optional search "
                    "and click mediators; inventory caps; product-week brand scores; sponsorship impressions/spend; "
                    "and volume-weighted product distribution. Preserve keys, units, currencies, time zones, "
                    "week-ending convention, and source lineage for every field.",
                ]),
                ("Real-data additions to the contract", [
                    "Obtain a canonical product/SKU hierarchy and tier mapping; transactions or weekly units; "
                    "realized net price and promotion depth; returns/cancellations; variable COGS and contribution "
                    "costs; campaign-level spend, impressions, clicks, reach/frequency, vendor, channel, and "
                    "flight dates; brand survey scores with respondent counts, sampling weights, wording, field "
                    "dates, and uncertainty; store availability and store-volume weights; and inventory, lost "
                    "sales, allocation, and replenishment records.",
                    "Document aggregation rules before modeling: media-to-week attribution, exchange-rate and "
                    "invoice treatment, price net of discount, availability versus assortment, brand-score "
                    "reliability, and whether product substitutions move demand between tiers. Add validation "
                    "for duplicate keys, missing periods, late revisions, negative spend, impossible prices, "
                    "and inconsistent stockout flags.",
                ]),
                ("Temporal split and leakage controls", [
                    "Choose rolling-origin backtests that mimic the decision date. Fit transformations, "
                    "scalers, priors informed by data, and feature selection on training data only. Never use "
                    "future sales, future survey responses, realized future media, or known generating truth "
                    "to choose an allocation. Clearly label future media, prices, events, organic demand, "
                    "coverage, and inventory as forecasts or scenario assumptions.",
                    "Carry adstock history and the last filtered brand state across each forecast origin. "
                    "Use holdout brand measurements for evaluation only, not as state updates. Keep a final "
                    "untouched period after model selection; a synthetic paired holdout is not independent "
                    "evidence of real-world generalization.",
                ]),
                ("Identification before interpretation", [
                    "A well-mixing sampler cannot fix confounded media spend. Map budget decisions, targeting, "
                    "and demand shocks; inspect channel correlation and time-varying omitted variables. Use "
                    "geo experiments, randomized holdouts, or credible quasi-experimental variation to anchor "
                    "incrementality and mediation. Define brand construct validity and distribution reach "
                    "before assigning those effects a causal or financial interpretation.",
                ]),
            ],
            3,
        ),
        (
            "B", "Model Blocks, Priors, and Likelihoods",
            "The current PyMC model combines pooled demand baselines, nonlinear media response, funnel "
            "mediation, censored fulfillment, and a stochastic brand state. The values below describe "
            "this synthetic reference, not recommended universal priors.",
            [
                ("Baseline and partial pooling", [
                    "Log expected sales combine a global intercept; zero-sum product, tier, and product x tier "
                    "contrasts; relative price, macro, base-price growth, promotion, post-promotion dip, "
                    "PrimeDays, and annual sine/cosine controls; media response; and the lagged brand effect. "
                    "Contrast scales use HalfNormal priors (product 0.8, tier 0.6, interaction 0.15). These "
                    "encode shrinkage toward group centers while allowing supported groups to differ.",
                ]),
                ("Media response and hierarchy", [
                    "Display, Video, Social, and CTV use estimated geometric adstock over current week plus "
                    "12 prior weeks, with normalized weights; paid-search clicks have no carryover. Hill "
                    "saturation uses power 2 for Social and power 1 otherwise. Media coefficients are "
                    "positive HalfNormal(0.8) log-sales effects. Carryover decays use Beta(2,2); "
                    "half-saturation uses LogNormal(0,0.6).",
                    "In hierarchical-media mode, product coefficients equal a positive population coefficient "
                    "times exp(zero-sum log deviation). Learned product-deviation scales use HalfNormal(0.3) "
                    "priors. Product slopes are shared across tiers; transformations and funnel parameters "
                    "remain shared. Only four product groups are available, so pooling sensitivity and "
                    "no-pooling comparisons remain important.",
                ]),
                ("Search and click funnel", [
                    "CTV saturation, macro, seasonality, and an observed organic-demand index drive latent "
                    "branded-search intensity. Search observations are rounded log-Normal; paid clicks are "
                    "Binomial conditional on searches with spend-dependent capture. Zero spend gives zero "
                    "clicks. Sales use structural expected clicks, not noisy observed holdout mediators. "
                    "CTV receives direct and induced-click lift; paid search receives its no-CTV click lift "
                    "under the chosen ordered attribution convention.",
                ]),
                ("Brand state and Kalman scope", [
                    "On the logit score scale, z[t] = rho*z[t-1] + (1-rho)*target[t] + process shock[t]. "
                    "The target is an intercept plus product effect, brand_media_beta times channel response "
                    "features, brand_sponsorship_beta times sponsorship_ratio/(1+sponsorship_ratio), and "
                    "brand_distribution_beta times (distribution_fraction-0.5). Brand channel coefficients "
                    "are distinct from sales media coefficients. Process/measurement scales use "
                    "HalfNormal(0.05); persistence uses Beta(8,2). A Gaussian measurement links the logit "
                    "state to the observed 0-100 survey score.",
                    "The Kalman recursion predicts each state from its prior filtered state, then corrects "
                    "it with training survey measurements; held-out scores are not used to update it. It "
                    "propagates state uncertainty forward. Candidate future media and sponsorship therefore "
                    "recompute the brand path, with process shocks included in financial predictive scenarios.",
                    "Sales have a separate current-period media term using media_beta or product_media_beta "
                    "on adstock/saturation and modeled paid-click features. The previous week's filtered "
                    "brand score, transformed with brand_sales_beta, adds to baseline log sales. Thus media "
                    "can affect sales directly now and indirectly through later brand-driven baseline; "
                    "sponsorship has no immediate-sales coefficient and works through later brand state. "
                    "Sales are nonlinear/censored, not Gaussian Kalman measurements; the brand sales term "
                    "uses a logistic-normal moment approximation. Validate against richer nonlinear "
                    "state-space alternatives.",
                ]),
                ("Sales and censoring likelihood", [
                    "The reference uses Normal noise on log sales (equivalently log-Normal demand), with "
                    "rounded unit counts treated continuously. When inventory is enabled, observed sales "
                    "are min(demand, cap) and the likelihood is right-censored at log inventory. Forecast "
                    "fulfilled means use E[min(D,cap)] analytically, not min(E[D],cap). The model does not "
                    "forecast supply or replenishment.",
                ]),
            ],
            4,
        ),
        (
            "C", "Reproducible Workflow and Validation",
            "Use the stages below as an implementation checklist. Preserve the ordering: lock data and "
            "assumptions first, evaluate the model second, and only publish allocation recommendations "
            "after diagnostics and decision checks pass.",
            [
                ("Run and preserve the reference", [
                    "Current end-to-end command: <font name='Courier'>python run_mmm.py --funnel --stockouts "
                    "--hierarchical-media --brand-state --target-accept 0.99 --output outputs_kalman_profit_validated</font>. "
                    "In this workspace, invoke it with <font name='Courier'>.venv/Scripts/python.exe</font>. "
                    "The reference uses seed 20261007, 156 weeks, 130 training weeks, 26 holdout weeks, "
                    "four NUTS chains, 1,500 tune steps, and 1,000 retained draws per chain. Use a new "
                    "output directory for each specification or seed.",
                ]),
                ("Pipeline", [
                    "1. Generate/import and validate the panel. 2. Fit only training sales and training "
                    "mediators/scores. 3. Check sampler diagnostics and posterior predictive behavior. "
                    "4. Forecast conditional on explicit future inputs. 5. Check effect recovery/calibration "
                    "where truth or experiments exist. 6. Reconcile attribution. 7. Optimize against the "
                    "same fixed scenarios and constraints for posterior and benchmark/oracle comparisons. "
                    "8. Export data, configuration, diagnostics, metrics, plots, allocations, and report.",
                ]),
                ("Required gates", [
                    "Sampling: max R-hat < 1.01, bulk ESS >= 400, tail ESS >= 200, zero divergences, "
                    "and tree depth below the configured cap. Forecast: beat the product-tier mean "
                    "benchmark and achieve 85%-97% nominal 90% coverage. Media recovery: at least four "
                    "of five truths covered, maximum coefficient error <= 25%, decay error <= 0.10. "
                    "Attribution error <= 25%; budget conserved; unit allocation within 2% of the best "
                    "known-truth solution found under the same constraints. These are lightweight "
                    "synthetic gates, not sufficient production acceptance criteria.",
                ]),
                ("Additional real-data checks", [
                    "Add multiple independent temporal origins, geographies, product groups, and campaign "
                    "periods; simulation-based calibration; prior predictive checks and prior sensitivity; "
                    "posterior predictive checks by product/tier/channel and stockout regime; missingness "
                    "and measurement-error sensitivity; and model misspecification stress tests. Compare "
                    "Ridge and seasonal/naive baselines on identical splits. Report forecast calibration "
                    "and decision stability, not RMSE alone.",
                ]),
                ("Artifacts and failure behavior", [
                    "Keep <font name='Courier'>run_config.json</font>, <font name='Courier'>metrics.json</font>, "
                    "posterior draws, run report, sales/mediator/brand predictions, recovery tables, "
                    "unit decomposition, channel contributions, response curves, and both unit and profit "
                    "allocation CSVs. A failed sampler gate must stop recommendation generation. Record "
                    "code version, data snapshot, feature definitions, assumptions, and random seeds with "
                    "each real-data run; never silently retune against the final holdout.",
                ]),
            ],
            5,
        ),
        (
            "D", "Decision Layer and Real-Data Roadmap",
            "The optimizer is a transparent scenario tool, not an autonomous budget authority. "
            "Its outputs are conditional on the response model, future inputs, economics, risk preference, "
            "and the bounds supplied by the decision owner.",
            [
                ("Current planning contract", [
                    "Plan 13 weeks with one constant share per channel, fixed product proportions, fixed total "
                    "weekly budget based on recent spend, and each share bounded to 40%-180% of baseline. "
                    "Known CPM/CPC mappings convert spend to exposure; sponsorship is a sixth line in brand "
                    "mode. Historical adstock and filtered brand state carry into planning. Future prices, "
                    "controls, organic demand, distribution, and inventory are supplied scenarios.",
                    "The workflow reports three objectives: expected fulfilled units; expected net "
                    "contribution profit; and 50% expected profit + 50% lower-tail CVaR at 5% across "
                    "512 predictive scenarios. CVaR is the mean profit in the worst 5% of scenarios. "
                    "Scenarios include posterior draws, sales noise, inventory caps, and brand shocks. "
                    "Six SLSQP starts find a best feasible point, not a global-optimality certificate.",
                ]),
                ("Synthetic economics: replace before use", [
                    "Profit proxy = fulfilled units x list price after discount x assumed product-tier "
                    "contribution margin - media/sponsorship spend. The current illustrative rates by "
                    "product and Core/Premium/Value tier are: Entry-Level Desktops 18/20/16%; Gaming PCs "
                    "22/28/19%; Premium Desktops 25/30/22%; Premium Laptops 20/26/18%. Media reference "
                    "costs are Display/Video/Social/CTV CPM $4/$12/$9/$22, paid-search CPC $1.50, and "
                    "sponsorship CPM $18.",
                    "These are not observed unit costs. Add verified variable COGS, rebates, fulfillment, "
                    "returns, taxes, and any contribution definitions approved by Finance. Model uncertainty "
                    "in price, margin, media costs, supply, and forecast inputs where decision material. "
                    "A fixed total spend is common to all candidate allocations; clarify whether budget "
                    "spend is a hard constraint, a cost in the objective, or both in the real decision.",
                ]),
                ("Implementation phases", [
                    "Phase 0 - Data and economics: sign off on keys, grain, measurement definitions, "
                    "time alignment, COGS/margins, inventory semantics, and future-input ownership.",
                    "Phase 1 - Baselines: build time-aware seasonal-naive and Ridge benchmarks; establish "
                    "rolling-origin forecast/calibration results before the hierarchical MMM.",
                    "Phase 2 - Estimation: fit the simplest identified model first, then add adstock, pooling, "
                    "censoring, funnel, and brand state one at a time. Require prior sensitivity and "
                    "out-of-sample evidence for added complexity.",
                    "Phase 3 - Causal validation: reconcile media effects with experiments or credible "
                    "quasi-experiments; validate brand-score construct, state persistence, distribution "
                    "measurement, and mediation assumptions.",
                    "Phase 4 - Decision pilot: run unit, expected-profit, and risk-aware allocations in "
                    "shadow mode. Review constraint sensitivity, CVaR level/weight, scenario coverage, "
                    "profit reconciliation, and decision stability with Marketing, Finance, and Supply.",
                    "Phase 5 - Operations: version data/model/config; log recommendations and overrides; "
                    "monitor drift, calibration, inventory, spend costs, and realized outcomes; define "
                    "rollback and retraining rules. Promote only after agreed acceptance criteria.",
                ]),
                ("Open limitations and next research", [
                    "No stochastic CPM/CPC likelihood, explicit SKU breadth, distribution investment "
                    "constraint, replenishment, substitution, latent organic-demand model, or long-horizon "
                    "brand-value objective is fitted. The Kalman filter uses score observations; sales are "
                    "not Gaussian measurement updates. Brand attribution to historical funders remains an "
                    "accounting convention. One matched synthetic seed and a successful fit do not establish "
                    "real-world causal validity. Prioritize experiments, independent rolling validation, "
                    "richer state-space alternatives, and decision-value comparisons.",
                ]),
            ],
            6,
        ),
    ]
    for appendix, title, introduction, sections, page_number in appendices:
        draw_appendix_page(
            pdf, width, height, margin, appendix, title, introduction, sections, page_number
        )
        if page_number < 6:
            pdf.showPage()
    pdf.save()
    print(f"Created six-page implementation blueprint: {output}")


def main():
    parser = argparse.ArgumentParser(description="Build a learning brief from verified MMM results.")
    parser.add_argument("--output", type=Path, default=ROOT / "MMM_Implementation_Blueprint.pdf")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs_combined")
    args = parser.parse_args()
    build_pdf(args.output, args.results_dir)


if __name__ == "__main__":
    main()