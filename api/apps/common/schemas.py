"""Response/request documentation serializers mirroring ``frontend/src/api/types.ts``.

Each class corresponds 1:1 to a TypeScript interface (``FooSerializer`` ⇔ ``Foo``): same field
names, same nullability (``X | null`` → ``allow_null=True``), optional request fields
(``foo?:``) → ``required=False``. They drive the OpenAPI schema (``/api/schema/``) and the
contract tests; computed payloads are built as plain dicts by the services and returned
directly (serialising 500-row pages field by field through DRF would be needlessly slow).

Model-backed resources (Job, Dataset, Fit, Simulation, Review) have their real serializers in
their apps; this module only holds the payload types.
"""

from __future__ import annotations

from drf_spectacular.utils import PolymorphicProxySerializer
from rest_framework import serializers

from .serializers import JobSerializer

POLICY_HELP = (
    "Policy name: thompson | ucb1 | epsilon_greedy | greedy | historical | random "
    "(any other name accepted by the core policy factory, e.g. thompson_se, is passed through)"
)


class ApiErrorBodySerializer(serializers.Serializer):
    detail = serializers.CharField(required=False)
    code = serializers.CharField(required=False)


class HistogramBinSerializer(serializers.Serializer):
    lo = serializers.FloatField()
    hi = serializers.FloatField()
    count = serializers.IntegerField()


class NumericFeatureSummarySerializer(serializers.Serializer):
    feature = serializers.CharField()
    kind = serializers.ChoiceField(choices=["numeric"])
    stage = serializers.ChoiceField(choices=["pre_decision", "policy", "disbursement", "interim", "terminal"])
    unit = serializers.ChoiceField(choices=["usd", "fraction", "pct", "count", "score", "years", "months", "none"])
    n = serializers.IntegerField()
    null_share = serializers.FloatField()
    mean = serializers.FloatField()
    std = serializers.FloatField()
    p05 = serializers.FloatField()
    p50 = serializers.FloatField()
    p95 = serializers.FloatField()
    bins = HistogramBinSerializer(many=True)


class CategoricalFeatureSummaryLevelsItemSerializer(serializers.Serializer):
    level = serializers.CharField()
    count = serializers.IntegerField()


class CategoricalFeatureSummarySerializer(serializers.Serializer):
    feature = serializers.CharField()
    kind = serializers.ChoiceField(choices=["categorical"])
    stage = serializers.ChoiceField(choices=["pre_decision", "policy", "disbursement", "interim", "terminal"])
    n = serializers.IntegerField()
    null_share = serializers.FloatField()
    levels = CategoricalFeatureSummaryLevelsItemSerializer(many=True)


class MonthlyArrivalSerializer(serializers.Serializer):
    epoch = serializers.CharField()
    n = serializers.IntegerField()
    n_resolved = serializers.IntegerField()
    n_default = serializers.IntegerField()


FEATURE_SUMMARY_FIELD = PolymorphicProxySerializer(
    component_name="FeatureSummary",
    serializers=[NumericFeatureSummarySerializer, CategoricalFeatureSummarySerializer],
    resource_type_field_name="kind",
    many=True,
)


class DatasetSummarySerializer(serializers.Serializer):
    dataset = serializers.IntegerField()
    n_loans = serializers.IntegerField()
    features = FEATURE_SUMMARY_FIELD
    monthly_arrivals = MonthlyArrivalSerializer(many=True)
    status_counts = serializers.DictField(child=serializers.IntegerField())
    grade_counts = serializers.DictField(child=serializers.IntegerField())
    term_counts = serializers.DictField(child=serializers.IntegerField())


class BorrowerContextSerializer(serializers.Serializer):
    annual_inc = serializers.FloatField(allow_null=True)
    annual_inc_joint = serializers.FloatField(allow_null=True)
    application_type = serializers.CharField(allow_null=True)
    dti = serializers.FloatField(allow_null=True)
    dti_joint = serializers.FloatField(allow_null=True)
    emp_length = serializers.FloatField(allow_null=True)
    fico_range_low = serializers.FloatField(allow_null=True)
    fico_range_high = serializers.FloatField(allow_null=True)
    home_ownership = serializers.CharField(allow_null=True)
    purpose = serializers.CharField(allow_null=True)
    verification_status = serializers.CharField(allow_null=True)
    verified_status_joint = serializers.CharField(allow_null=True)
    delinq_2yrs = serializers.FloatField(allow_null=True)
    delinq_amnt = serializers.FloatField(allow_null=True)
    inq_last_6mths = serializers.FloatField(allow_null=True)
    inq_last_12m = serializers.FloatField(allow_null=True)
    open_acc = serializers.FloatField(allow_null=True)
    total_acc = serializers.FloatField(allow_null=True)
    revol_bal = serializers.FloatField(allow_null=True)
    revol_util = serializers.FloatField(allow_null=True)
    installment = serializers.FloatField(allow_null=True)
    mort_acc = serializers.FloatField(allow_null=True)
    pub_rec = serializers.FloatField(allow_null=True)
    pub_rec_bankruptcies = serializers.FloatField(allow_null=True)
    collections_12_mths_ex_med = serializers.FloatField(allow_null=True)
    num_accts_ever_120_pd = serializers.FloatField(allow_null=True)
    pct_tl_nvr_dlq = serializers.FloatField(allow_null=True)
    num_sats = serializers.FloatField(allow_null=True)
    tot_coll_amt = serializers.FloatField(allow_null=True)
    tot_cur_bal = serializers.FloatField(allow_null=True)
    tot_hi_cred_lim = serializers.FloatField(allow_null=True)
    total_bal_ex_mort = serializers.FloatField(allow_null=True)
    total_bc_limit = serializers.FloatField(allow_null=True)
    bc_util = serializers.FloatField(allow_null=True)
    all_util = serializers.FloatField(allow_null=True)
    grade = serializers.CharField(allow_null=True)
    sub_grade = serializers.CharField(allow_null=True)


class LoanPolicySerializer(serializers.Serializer):
    loan_amnt = serializers.FloatField()
    int_rate = serializers.FloatField()
    term = serializers.IntegerField()


class LoanOutcomeSerializer(serializers.Serializer):
    loan_status = serializers.CharField()
    resolved = serializers.BooleanField()
    defaulted = serializers.BooleanField(allow_null=True)
    recovery_rate = serializers.FloatField(allow_null=True)
    realized_revenue = serializers.FloatField(allow_null=True)
    realized_loss = serializers.FloatField(allow_null=True)
    realized_reward = serializers.FloatField(allow_null=True)
    months_on_book = serializers.FloatField(allow_null=True)
    resolution_epoch = serializers.CharField(allow_null=True)


class LoanRecordSerializer(serializers.Serializer):
    loan_id = serializers.IntegerField()
    issue_d = serializers.CharField()
    funded_amnt = serializers.FloatField()
    group = serializers.CharField()
    historical_arm = serializers.IntegerField()
    context = BorrowerContextSerializer()
    policy = LoanPolicySerializer()
    outcome = LoanOutcomeSerializer()


class ArmSpaceConfigSerializer(serializers.Serializer):
    amounts = serializers.ListField(child=serializers.FloatField())
    rates = serializers.ListField(child=serializers.FloatField())
    terms = serializers.ListField(child=serializers.IntegerField())
    include_null = serializers.BooleanField()
    max_payment_to_income = serializers.FloatField(
        allow_null=True, help_text="installment <= x * annual_inc / 12 (0.20)"
    )
    rate_corridor = serializers.FloatField(
        required=False, allow_null=True, help_text="|rate - sub-grade reference| <= x (0.05)"
    )
    max_amount_to_income = serializers.FloatField(
        required=False, allow_null=True, help_text="amount <= x * annual_inc (0.50)"
    )
    max_total_dti = serializers.FloatField(
        required=False, allow_null=True, help_text="dti/100 + 12 * installment / annual_inc <= x (0.43, QM rule)"
    )


class HierarchyConfigSerializer(serializers.Serializer):
    kappa_group = serializers.JSONField()  # union
    kappa_leaf = serializers.JSONField()  # union
    alpha0 = serializers.FloatField()
    beta0 = serializers.FloatField()
    policy_effect = serializers.BooleanField(required=False, help_text="structured leaf prior (policy-effect tilt)")
    policy_effect_cfg = serializers.JSONField(required=False, help_text="PolicyEffectConfig")


class RecoveryConfigSerializer(serializers.Serializer):
    min_obs = serializers.IntegerField()
    basis = serializers.ChoiceField(choices=["principal", "post_chargeoff"])
    term_level = serializers.BooleanField(required=False, help_text="leaf -> (group, term) -> group -> global")


class MctmConfigSerializer(serializers.Serializer):
    structure = serializers.ChoiceField(choices=["proposal", "extended"])
    prior_strength = serializers.FloatField()
    chargeoff_lag_months = serializers.IntegerField()


class RewardConfigSerializer(serializers.Serializer):
    funding_rate = serializers.FloatField()
    risk_aversion = serializers.FloatField()
    lendable_threshold = serializers.FloatField()
    prepayment_adjustment = serializers.ChoiceField(
        choices=["empirical"],
        allow_null=True,
        required=False,
        help_text="realised-interest factor (null = full schedule)",
    )


class FitConfigSegmentationSerializer(serializers.Serializer):
    dimensions = serializers.ListField(
        child=serializers.ChoiceField(
            choices=[
                "grade",
                "sub_grade",
                "purpose_group",
                "home_ownership",
                "fico_band",
                "dti_band",
                "income_band",
                "emp_band",
                "verification_status",
                "application_type",
            ]
        )
    )


class FitConfigSerializer(serializers.Serializer):
    segmentation = FitConfigSegmentationSerializer()
    arm_space = ArmSpaceConfigSerializer()
    hierarchy = HierarchyConfigSerializer()
    recovery = RecoveryConfigSerializer()
    mctm = MctmConfigSerializer()
    reward = RewardConfigSerializer()
    train_end = serializers.CharField()


class FitMetricsSerializer(serializers.Serializer):
    brier = serializers.FloatField()
    brier_baseline = serializers.FloatField()
    auc = serializers.FloatField()
    ece = serializers.FloatField()
    log_loss = serializers.FloatField()
    recovery_mae = serializers.FloatField()
    recovery_rmse = serializers.FloatField()
    transition_log_lik = serializers.FloatField()
    transition_log_lik_per_obs = serializers.FloatField()
    transition_prediction_error = serializers.FloatField()
    n_train = serializers.IntegerField()
    n_test = serializers.IntegerField()
    n_defaults_train = serializers.IntegerField()
    base_rate_train = serializers.FloatField()
    base_rate_test = serializers.FloatField()
    fit_seconds = serializers.FloatField()
    interest_realization = serializers.JSONField(
        required=False, allow_null=True, help_text="prepayment table rho(grade, term)"
    )
    policy_effect = serializers.JSONField(
        required=False, allow_null=True, help_text="leaf-prior tilt: beta, n_fit, status"
    )
    guardrails = serializers.JSONField(required=False, allow_null=True)
    reference_rates = serializers.JSONField(required=False, allow_null=True)


class RecoverySummarySerializer(serializers.Serializer):
    mean = serializers.FloatField()
    alpha = serializers.FloatField()
    beta = serializers.FloatField()
    n_obs = serializers.IntegerField()
    source = serializers.ChoiceField(choices=["global", "group", "leaf"])


class HierarchyRowSerializer(serializers.Serializer):
    level = serializers.ChoiceField(choices=["global", "group", "leaf"])
    key = serializers.CharField()
    parent = serializers.CharField(allow_null=True)
    group = serializers.CharField(allow_null=True)
    arm = serializers.IntegerField(allow_null=True)
    dims = serializers.DictField(child=serializers.CharField())
    alpha = serializers.FloatField()
    beta = serializers.FloatField()
    n_obs = serializers.IntegerField()
    defaults = serializers.IntegerField()
    mean = serializers.FloatField()
    hdi_low = serializers.FloatField()
    hdi_high = serializers.FloatField()
    prior_mean = serializers.FloatField(allow_null=True)
    kappa = serializers.FloatField(allow_null=True)
    shrinkage_weight = serializers.FloatField()
    empirical_rate = serializers.FloatField(allow_null=True)
    recovery = RecoverySummarySerializer(allow_null=True)
    n_children = serializers.IntegerField()


class BetaSummarySerializer(serializers.Serializer):
    alpha = serializers.FloatField()
    beta = serializers.FloatField()
    mean = serializers.FloatField()
    std = serializers.FloatField()
    hdi_low = serializers.FloatField()
    hdi_high = serializers.FloatField()


class DensityPointSerializer(serializers.Serializer):
    x = serializers.FloatField()
    density = serializers.FloatField()


class LendingPolicySerializer(serializers.Serializer):
    amount = serializers.FloatField()
    rate = serializers.FloatField()
    term = serializers.IntegerField()


class RewardBreakdownSerializer(serializers.Serializer):
    mean = serializers.FloatField()
    std = serializers.FloatField()
    revenue = serializers.FloatField()
    expected_loss = serializers.FloatField()
    funding_cost = serializers.FloatField()
    risk_adjusted = serializers.FloatField()


class RiskEstimatePDefaultSerializer(BetaSummarySerializer):
    level = serializers.ChoiceField(choices=["global", "group", "leaf"])
    n_obs = serializers.IntegerField()
    defaults = serializers.IntegerField()
    shrinkage_weight = serializers.FloatField()


class RiskEstimateRecoverySerializer(BetaSummarySerializer):
    level = serializers.ChoiceField(choices=["global", "group", "leaf"])
    n_obs = serializers.IntegerField()


class RiskEstimatePdfSerializer(serializers.Serializer):
    p_default = DensityPointSerializer(many=True)
    p_default_prior = DensityPointSerializer(many=True)
    recovery = DensityPointSerializer(many=True)


class RiskEstimateSerializer(serializers.Serializer):
    fit = serializers.IntegerField()
    group = serializers.CharField()
    arm = serializers.IntegerField(allow_null=True)
    policy = LendingPolicySerializer(allow_null=True)
    label = serializers.CharField()
    p_default = RiskEstimatePDefaultSerializer()
    recovery = RiskEstimateRecoverySerializer()
    expected_loss_fraction = serializers.FloatField()
    reward = RewardBreakdownSerializer(allow_null=True)
    pdf = RiskEstimatePdfSerializer()


class ReliabilityBinSerializer(serializers.Serializer):
    bin = serializers.IntegerField()
    lo = serializers.FloatField()
    hi = serializers.FloatField()
    n = serializers.IntegerField()
    mean_predicted = serializers.FloatField()
    observed_rate = serializers.FloatField()


class RocPointSerializer(serializers.Serializer):
    fpr = serializers.FloatField()
    tpr = serializers.FloatField()
    threshold = serializers.FloatField()


class RecoveryFitByGradeItemSerializer(serializers.Serializer):
    grade = serializers.CharField()
    mean = serializers.FloatField()
    n = serializers.IntegerField()


class RecoveryFitSerializer(serializers.Serializer):
    alpha = serializers.FloatField()
    beta = serializers.FloatField()
    mean = serializers.FloatField()
    n_defaults = serializers.IntegerField()
    histogram = HistogramBinSerializer(many=True)
    pdf = DensityPointSerializer(many=True)
    mae = serializers.FloatField()
    rmse = serializers.FloatField()
    by_grade = RecoveryFitByGradeItemSerializer(many=True)


class CalibrationSerializer(serializers.Serializer):
    fit = serializers.IntegerField()
    n_test = serializers.IntegerField()
    base_rate = serializers.FloatField()
    brier = serializers.FloatField()
    auc = serializers.FloatField()
    ece = serializers.FloatField()
    log_loss = serializers.FloatField()
    bins = ReliabilityBinSerializer(many=True)
    roc = RocPointSerializer(many=True)
    recovery = RecoveryFitSerializer()


class MctmTrajectoryRowSerializer(serializers.Serializer):
    t = serializers.IntegerField()
    current = serializers.FloatField()
    delinquent = serializers.FloatField()
    recovered = serializers.FloatField()
    loss = serializers.FloatField()


class PDefaultPointSerializer(serializers.Serializer):
    t = serializers.IntegerField()
    p_default = serializers.FloatField()
    p_loss = serializers.FloatField()
    p_delinquent = serializers.FloatField()


class MctmMaturitySerializer(serializers.Serializer):
    t = serializers.IntegerField()
    before = serializers.ListField(child=serializers.FloatField())
    after = serializers.ListField(child=serializers.FloatField())
    p_default = serializers.FloatField()


class MctmSerializer(serializers.Serializer):
    fit = serializers.IntegerField()
    structure = serializers.ChoiceField(choices=["proposal", "extended"])
    states = serializers.ListField(
        child=serializers.ChoiceField(choices=["current", "delinquent", "recovered", "loss"])
    )
    state_labels = serializers.ListField(child=serializers.CharField())
    matrix = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    counts = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    structural_zeros = serializers.ListField(child=serializers.ListField(child=serializers.BooleanField()))
    matrix_low = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    matrix_high = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    absorption = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    expected_steps = serializers.ListField(child=serializers.FloatField())
    start = serializers.ChoiceField(choices=["current", "delinquent"])
    horizon = serializers.IntegerField()
    trajectory = MctmTrajectoryRowSerializer(many=True)
    p_default_curve = PDefaultPointSerializer(many=True)
    maturity = MctmMaturitySerializer()
    log_likelihood = serializers.FloatField()
    n_transitions = serializers.IntegerField()
    row_prediction_error = serializers.ListField(child=serializers.FloatField())
    prior_strength = serializers.FloatField()


class ArmRowSerializer(serializers.Serializer):
    arm = serializers.IntegerField()
    amount = serializers.FloatField()
    rate = serializers.FloatField()
    term = serializers.IntegerField()
    is_null = serializers.BooleanField()
    installment = serializers.FloatField()
    label = serializers.CharField()


class FitArmRowSerializer(ArmRowSerializer):
    expected_reward = serializers.FloatField()
    reward_std = serializers.FloatField()
    p_default = serializers.FloatField()
    expected_recovery = serializers.FloatField()
    historical_count = serializers.IntegerField()
    historical_share = serializers.FloatField()


class FitArmsSerializer(serializers.Serializer):
    fit = serializers.IntegerField()
    amounts = serializers.ListField(child=serializers.FloatField())
    rates = serializers.ListField(child=serializers.FloatField())
    terms = serializers.ListField(child=serializers.IntegerField())
    arms = FitArmRowSerializer(many=True)


class ShadowEvaluateByLoanSerializer(serializers.Serializer):
    loan_id = serializers.IntegerField()


class ShadowEvaluateByContextSerializer(serializers.Serializer):
    context = BorrowerContextSerializer()
    historical_policy = LendingPolicySerializer(required=False, allow_null=True)


class ShadowArmRowSerializer(ArmRowSerializer):
    eligible = serializers.BooleanField()
    p_default_mean = serializers.FloatField()
    p_default_hdi_low = serializers.FloatField()
    p_default_hdi_high = serializers.FloatField()
    expected_recovery = serializers.FloatField()
    reward_mean = serializers.FloatField()
    reward_std = serializers.FloatField()
    reward_q05 = serializers.FloatField()
    reward_q50 = serializers.FloatField()
    reward_q95 = serializers.FloatField()
    prob_positive = serializers.FloatField()
    risk_adjusted = serializers.FloatField()
    realization = serializers.FloatField(required=False, help_text="prepayment factor rho of the arm")


class ShadowReportSerializer(serializers.Serializer):
    fit = serializers.IntegerField()
    loan_id = serializers.IntegerField(allow_null=True)
    issue_d = serializers.CharField(allow_null=True)
    group = serializers.CharField()
    context = BorrowerContextSerializer()
    context_columns = serializers.ListField(child=serializers.CharField())
    leakage_safe = serializers.BooleanField()
    n_samples = serializers.IntegerField()
    risk_aversion = serializers.FloatField()
    lendable_threshold = serializers.FloatField()
    max_payment_to_income = serializers.FloatField()
    arms = ShadowArmRowSerializer(many=True)
    recommended_arm = serializers.IntegerField()
    historical_policy = LendingPolicySerializer(allow_null=True)
    historical_arm = serializers.IntegerField(allow_null=True)
    historical_expected_reward = serializers.FloatField(allow_null=True)
    historical_realized_reward = serializers.FloatField(allow_null=True)
    uplift_expected = serializers.FloatField(allow_null=True)
    uplift_vs_realized = serializers.FloatField(allow_null=True)
    lendable = serializers.BooleanField()
    outcome = LoanOutcomeSerializer(allow_null=True)
    eligibility = serializers.JSONField(
        required=False,
        help_text="reference_rate, n_eligible, n_eligible_lending, historical_admitted, guardrails, rate_corridor_applied",
    )


class PortfolioFiltersSerializer(serializers.Serializer):
    grade = serializers.ListField(child=serializers.CharField(), required=False)
    purpose = serializers.ListField(child=serializers.CharField(), required=False)
    home_ownership = serializers.ListField(child=serializers.CharField(), required=False)
    issue_from = serializers.CharField(required=False)
    issue_to = serializers.CharField(required=False)
    resolved_only = serializers.BooleanField(required=False)


class PortfolioRequestSerializer(serializers.Serializer):
    max_loans = serializers.IntegerField()
    seed = serializers.IntegerField()
    filters = PortfolioFiltersSerializer(required=False)


class PortfolioSummarySerializer(serializers.Serializer):
    n_loans = serializers.IntegerField()
    n_lendable = serializers.IntegerField()
    share_lendable = serializers.FloatField()
    expected_profit_historical = serializers.FloatField()
    expected_profit_shadow = serializers.FloatField()
    uplift = serializers.FloatField()
    uplift_pct = serializers.FloatField()
    realized_profit_historical = serializers.FloatField(allow_null=True)
    counts = serializers.DictField(child=serializers.IntegerField())
    recommendation_mix = serializers.JSONField(required=False, allow_null=True)
    eligibility = serializers.JSONField(required=False, allow_null=True)
    uplift_decomposition = serializers.JSONField(required=False, allow_null=True)
    by_risk_aversion = serializers.JSONField(
        required=False, allow_null=True, help_text='{"0.0": {...}, "0.5": ..., "1.0": ...}'
    )


class PortfolioGradeRowSerializer(serializers.Serializer):
    grade = serializers.CharField()
    n = serializers.IntegerField()
    expected_profit_historical = serializers.FloatField()
    expected_profit_shadow = serializers.FloatField()
    realized_profit_historical = serializers.FloatField(allow_null=True)
    counts = serializers.DictField(child=serializers.IntegerField())


class InclusionRowSerializer(serializers.Serializer):
    grade = serializers.CharField()
    n = serializers.IntegerField()
    share_lendable = serializers.FloatField()
    decline_rate_shadow = serializers.FloatField()
    mean_rate_historical = serializers.FloatField()
    mean_rate_shadow = serializers.FloatField(allow_null=True)
    mean_amount_historical = serializers.FloatField()
    mean_amount_shadow = serializers.FloatField(allow_null=True)


class PortfolioLoanRowSerializer(serializers.Serializer):
    loan_id = serializers.IntegerField()
    grade = serializers.CharField()
    group = serializers.CharField()
    historical_arm = serializers.IntegerField()
    recommended_arm = serializers.IntegerField()
    historical_label = serializers.CharField()
    recommended_label = serializers.CharField()
    historical_expected_reward = serializers.FloatField()
    shadow_expected_reward = serializers.FloatField()
    realized_reward = serializers.FloatField(allow_null=True)


class PortfolioResultSerializer(serializers.Serializer):
    job = JobSerializer()
    fit = serializers.IntegerField()
    request = PortfolioRequestSerializer()
    summary = PortfolioSummarySerializer(allow_null=True)
    by_grade = PortfolioGradeRowSerializer(many=True)
    inclusion = InclusionRowSerializer(many=True)
    top_loans = PortfolioLoanRowSerializer(many=True)


class PortfolioRunSerializer(serializers.Serializer):
    job = JobSerializer()
    request = PortfolioRequestSerializer()
    summary = PortfolioSummarySerializer(allow_null=True)


class PsiSettingsThresholdsSerializer(serializers.Serializer):
    watch = serializers.FloatField()
    breaker = serializers.FloatField()


class PsiSettingsSerializer(serializers.Serializer):
    features = serializers.ListField(child=serializers.CharField())
    window_months = serializers.IntegerField()
    thresholds = PsiSettingsThresholdsSerializer()
    aggregate = serializers.ChoiceField(choices=["max", "mean"])
    recovery_readings = serializers.IntegerField()
    fallback = serializers.ChoiceField(choices=["champion_snapshot", "decline", "historical"])
    auto_action = serializers.ChoiceField(choices=["resume", "resegment", "rollback"])
    auto_ack_epochs = serializers.IntegerField()


class SimulationConfigSerializer(serializers.Serializer):
    policies = serializers.ListField(child=serializers.CharField(help_text=POLICY_HELP))
    feedback_mode = serializers.ChoiceField(choices=["simulated", "replay", "hybrid"])
    start = serializers.CharField()
    end = serializers.CharField()
    max_loans = serializers.IntegerField(allow_null=True)
    seed = serializers.IntegerField()
    interim_months = serializers.IntegerField()
    interim_weight = serializers.FloatField()
    psi = PsiSettingsSerializer()
    breaker_mode = serializers.ChoiceField(choices=["pause", "auto_ack", "off"])
    checkpoint_every = serializers.IntegerField()
    hindsight_sample = serializers.ChoiceField(choices=["elapsed", "resolved"], required=False)


class PolicyHeadlineSerializer(serializers.Serializer):
    policy = serializers.CharField(help_text=POLICY_HELP)
    n_decisions = serializers.IntegerField()
    cum_reward = serializers.FloatField()
    cum_realized = serializers.FloatField()
    cum_regret = serializers.FloatField()
    mean_reward = serializers.FloatField()
    decline_rate = serializers.FloatField()
    mean_amount = serializers.FloatField(allow_null=True)
    mean_rate = serializers.FloatField(allow_null=True)


class RecoveryInfoSerializer(serializers.Serializer):
    killed_at = serializers.DateTimeField()
    killed_at_epoch = serializers.CharField(allow_null=True)
    resumed_at = serializers.DateTimeField(allow_null=True)
    resumed_from_epoch = serializers.CharField(allow_null=True)
    replayed_epochs = serializers.IntegerField()
    deterministic_match = serializers.BooleanField(allow_null=True)
    recovery_seconds = serializers.FloatField(allow_null=True)


class SimulationWorkerSerializer(serializers.Serializer):
    pid = serializers.IntegerField(allow_null=True)
    alive = serializers.BooleanField()
    last_heartbeat = serializers.DateTimeField(allow_null=True)


class TimeseriesRowSerializer(serializers.Serializer):
    epoch = serializers.CharField()
    policy = serializers.CharField(help_text=POLICY_HELP)
    n_decisions = serializers.IntegerField()
    reward = serializers.FloatField()
    cum_reward = serializers.FloatField()
    realized = serializers.FloatField()
    cum_realized = serializers.FloatField()
    regret = serializers.FloatField()
    cum_regret = serializers.FloatField()
    rolling_reward = serializers.FloatField()
    decline_rate = serializers.FloatField()
    mean_amount = serializers.FloatField(allow_null=True)
    mean_rate = serializers.FloatField(allow_null=True)
    breaker_state = serializers.ChoiceField(choices=["closed", "open", "half_open"])
    psi_aggregate = serializers.FloatField(allow_null=True)


class TimeseriesSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    policies = serializers.ListField(child=serializers.CharField(help_text=POLICY_HELP))
    epochs = serializers.ListField(child=serializers.CharField())
    rows = TimeseriesRowSerializer(many=True)


class SimArmRowSerializer(ArmRowSerializer):
    mu = serializers.FloatField(allow_null=True)
    sigma = serializers.FloatField(allow_null=True)
    n = serializers.FloatField()
    selections = serializers.IntegerField()
    cum_selections = serializers.IntegerField()
    share = serializers.FloatField()
    cum_share = serializers.FloatField()


class SimArmsGroupsItemSerializer(serializers.Serializer):
    group = serializers.CharField()
    n = serializers.IntegerField()


class SimArmsSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    policy = serializers.CharField(help_text=POLICY_HELP)
    epoch = serializers.CharField()
    group = serializers.CharField(allow_null=True)
    n_decisions = serializers.IntegerField()
    amounts = serializers.ListField(child=serializers.FloatField())
    rates = serializers.ListField(child=serializers.FloatField())
    terms = serializers.ListField(child=serializers.IntegerField())
    arms = SimArmRowSerializer(many=True)
    groups = SimArmsGroupsItemSerializer(many=True)


class PosteriorSnapshotRowSerializer(serializers.Serializer):
    epoch = serializers.CharField()
    arm = serializers.IntegerField()
    mu = serializers.FloatField()
    sigma = serializers.FloatField()
    n = serializers.FloatField()


class PosteriorHistorySerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    policy = serializers.CharField(help_text=POLICY_HELP)
    group = serializers.CharField(allow_null=True)
    snapshot_every = serializers.IntegerField()
    arms = ArmRowSerializer(many=True)
    rows = PosteriorSnapshotRowSerializer(many=True)


class SimpsonGroupRowSerializer(serializers.Serializer):
    group = serializers.CharField()
    dims = serializers.DictField(child=serializers.CharField())
    n = serializers.IntegerField()
    group_best_arm = serializers.IntegerField()
    group_best_label = serializers.CharField()
    group_best_mu = serializers.FloatField()
    pooled_arm_mu = serializers.FloatField()
    gap = serializers.FloatField()
    agrees = serializers.BooleanField()


class SimpsonPanelSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    policy = serializers.CharField(help_text=POLICY_HELP)
    epoch = serializers.CharField()
    pooled_best_arm = serializers.IntegerField()
    pooled_best_label = serializers.CharField()
    pooled_best_mu = serializers.FloatField()
    n_groups = serializers.IntegerField()
    n_agree = serializers.IntegerField()
    forfeited_per_decision = serializers.FloatField()
    groups = SimpsonGroupRowSerializer(many=True)


class SimulationEventSerializer(serializers.Serializer):
    seq = serializers.IntegerField()
    type = serializers.ChoiceField(
        choices=[
            "RunStarted",
            "BorrowerArrived",
            "PolicySelected",
            "FeedbackReleased",
            "PosteriorUpdated",
            "SentinelEvaluated",
            "BreakerTripped",
            "ReviewAcknowledged",
            "CheckpointWritten",
            "EpochCompleted",
            "RunPaused",
            "RunResumed",
            "WorkerKilled",
            "RunCompleted",
            "RunCancelled",
            "RunFailed",
        ]
    )
    epoch = serializers.CharField(allow_null=True)
    ts = serializers.DateTimeField()
    level = serializers.ChoiceField(choices=["info", "warning", "critical"])
    message = serializers.CharField()
    payload = serializers.DictField(child=serializers.JSONField())


class EventPageSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    results = SimulationEventSerializer(many=True)
    last_seq = serializers.IntegerField()
    has_more = serializers.BooleanField()


class PsiPointSerializer(serializers.Serializer):
    epoch = serializers.CharField()
    feature = serializers.CharField()
    psi = serializers.FloatField()
    level = serializers.ChoiceField(choices=["normal", "watch", "breaker"])


class PsiAggregatePointSerializer(serializers.Serializer):
    epoch = serializers.CharField()
    psi_max = serializers.FloatField()
    psi_mean = serializers.FloatField()
    aggregate = serializers.FloatField()
    level = serializers.ChoiceField(choices=["normal", "watch", "breaker"])
    breaker_state = serializers.ChoiceField(choices=["closed", "open", "half_open"])
    top_feature = serializers.CharField()


class BreakerTransitionSerializer(serializers.Serializer):
    epoch = serializers.CharField()
    ts = serializers.DateTimeField()
    from_state = serializers.ChoiceField(choices=["closed", "open", "half_open"])
    to_state = serializers.ChoiceField(choices=["closed", "open", "half_open"])
    reason = serializers.CharField()
    psi_aggregate = serializers.FloatField(allow_null=True)
    review = serializers.IntegerField(allow_null=True)
    action = serializers.ChoiceField(choices=["resume", "resegment", "rollback"], allow_null=True)


class SentinelTimelineThresholdsSerializer(serializers.Serializer):
    watch = serializers.FloatField()
    breaker = serializers.FloatField()


class SentinelTimelineReferenceSerializer(serializers.Serializer):
    to = serializers.CharField()
    n = serializers.IntegerField()
    rebased_at = serializers.ListField(child=serializers.CharField())


class SentinelTimelineCurrentSerializer(serializers.Serializer):
    epoch = serializers.CharField(allow_null=True)
    breaker_state = serializers.ChoiceField(choices=["closed", "open", "half_open"])
    level = serializers.ChoiceField(choices=["normal", "watch", "breaker"])
    psi_aggregate = serializers.FloatField(allow_null=True)
    top_feature = serializers.CharField(allow_null=True)


class SentinelTimelineShiftCountsSerializer(serializers.Serializer):
    watch = serializers.IntegerField()
    breaker = serializers.IntegerField()


class SentinelTimelineSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    features = serializers.ListField(child=serializers.CharField())
    thresholds = SentinelTimelineThresholdsSerializer()
    aggregate_mode = serializers.ChoiceField(choices=["max", "mean"])
    window_months = serializers.IntegerField()
    reference = SentinelTimelineReferenceSerializer()
    features_psi = PsiPointSerializer(many=True)
    aggregate = PsiAggregatePointSerializer(many=True)
    transitions = BreakerTransitionSerializer(many=True)
    current = SentinelTimelineCurrentSerializer()
    shift_counts = SentinelTimelineShiftCountsSerializer()


class PsiBucketSerializer(serializers.Serializer):
    bucket = serializers.CharField()
    lo = serializers.FloatField(allow_null=True)
    hi = serializers.FloatField(allow_null=True)
    expected = serializers.FloatField()
    actual = serializers.FloatField()
    contribution = serializers.FloatField()


class PsiBucketTableSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    feature = serializers.CharField()
    epoch = serializers.CharField()
    kind = serializers.ChoiceField(choices=["numeric", "categorical"])
    psi = serializers.FloatField()
    level = serializers.ChoiceField(choices=["normal", "watch", "breaker"])
    n_reference = serializers.IntegerField()
    n_window = serializers.IntegerField()
    buckets = PsiBucketSerializer(many=True)


class ReviewTopFeaturesItemSerializer(serializers.Serializer):
    feature = serializers.CharField()
    psi = serializers.FloatField()


class ReportMetricSerializer(serializers.Serializer):
    key = serializers.CharField()
    label = serializers.CharField()
    value = serializers.FloatField(allow_null=True)
    unit = serializers.ChoiceField(
        choices=["usd", "usd_per_decision", "fraction", "ratio", "count", "seconds", "megabytes", "nats", "score"]
    )
    better = serializers.ChoiceField(choices=["higher", "lower"], allow_null=True)
    baseline = serializers.FloatField(allow_null=True)
    baseline_label = serializers.CharField(allow_null=True)
    by_policy = serializers.DictField(child=serializers.FloatField(), allow_null=True)
    note = serializers.CharField(allow_null=True)


class ReportDimensionSerializer(serializers.Serializer):
    key = serializers.ChoiceField(
        choices=[
            "default_prediction",
            "recovery_prediction",
            "credit_state_modeling",
            "policy_performance",
            "bandit_performance",
            "counterfactual_evaluation",
            "distributional_monitoring",
            "computational_feasibility",
            "infrastructure_resilience",
        ]
    )
    label = serializers.CharField()
    axis = serializers.ChoiceField(choices=["predictive", "decision", "adaptability", "feasibility"])
    metrics = ReportMetricSerializer(many=True)


class RuntimeStatsSerializer(serializers.Serializer):
    cpu_seconds = serializers.FloatField()
    cpu_utilization = serializers.FloatField()
    peak_memory_mb = serializers.FloatField()
    wall_seconds = serializers.FloatField()
    epochs = serializers.IntegerField()
    decisions = serializers.IntegerField()
    decisions_per_second = serializers.FloatField()
    restarts = serializers.IntegerField()
    checkpoints_written = serializers.IntegerField()
    failed_workloads = serializers.IntegerField()
    recovery_seconds = serializers.FloatField(allow_null=True)


class EvaluationReportConfigSerializer(serializers.Serializer):
    fit = FitConfigSerializer()
    simulation = SimulationConfigSerializer()


class EvaluationReportSerializer(serializers.Serializer):
    simulation = serializers.IntegerField()
    fit = serializers.IntegerField()
    dataset = serializers.IntegerField()
    generated_at = serializers.DateTimeField()
    status = serializers.ChoiceField(choices=["final", "partial"])
    dimensions = ReportDimensionSerializer(many=True)
    runtime = RuntimeStatsSerializer()
    policy_summary = PolicyHeadlineSerializer(many=True)
    config = EvaluationReportConfigSerializer()


# Python keywords cannot be class attributes: inject the contract names after class creation.
PortfolioLoanRowSerializer._declared_fields["class"] = serializers.ChoiceField(
    choices=["rescued", "upsized", "repriced_down", "repriced_up", "declined", "unchanged"]
)
SentinelTimelineReferenceSerializer._declared_fields = {
    "from": serializers.CharField(help_text="First epoch of the reference window (YYYY-MM)"),
    **SentinelTimelineReferenceSerializer._declared_fields,
}


class PaginatedLoanRecordSerializer(serializers.Serializer):
    count = serializers.IntegerField()
    next = serializers.URLField(allow_null=True)
    previous = serializers.URLField(allow_null=True)
    results = LoanRecordSerializer(many=True)
