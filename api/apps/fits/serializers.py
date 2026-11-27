"""types.ts ``Fit``, ``FitCreate`` and ``PortfolioRequest``."""

from __future__ import annotations

from bayesian_credit.config import POLICY_EFFECT_FEATURES, FrameworkConfig, SegmentationConfig
from bayesian_credit.data.segmentation import DIMENSIONS
from bayesian_credit.exceptions import ConfigurationError
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from apps.common import schemas as S
from apps.common.mixins import BUILD_STATUSES
from apps.common.serializers import JobSerializer
from apps.common.utils import epoch_start, parse_epoch
from apps.datasets.models import Dataset

from .models import Fit
from .services import contract_config

SECTION_KEYS = {
    "arm_space": {
        "amounts",
        "rates",
        "terms",
        "include_null",
        "max_payment_to_income",
        "rate_corridor",
        "max_amount_to_income",
        "max_total_dti",
    },
    "hierarchy": {
        "kappa_group",
        "kappa_leaf",
        "alpha0",
        "beta0",
        "kappa_min",
        "kappa_max",
        "policy_effect",
        "policy_effect_cfg",
    },
    "recovery": {"min_obs", "basis", "max_concentration", "eps", "term_level"},
    "mctm": {"structure", "prior_strength", "chargeoff_lag_months"},
    "reward": {"funding_rate", "risk_aversion", "lendable_threshold", "exposure_basis", "prepayment_adjustment"},
}
#: Lending guardrails of the arm space: positive fractions, or null to switch one off.
GUARDRAIL_KEYS = ("max_payment_to_income", "rate_corridor", "max_amount_to_income", "max_total_dti")


def _number(value, *, integer: bool = False) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if integer and out != int(out):
        return None
    return out


def apply_exclusions(attrs: dict, ds: Dataset, errors: dict[str, list[str]]) -> tuple[list[str] | None, dict]:
    """Segmentation dimensions and hierarchy block consistent with the dataset's excluded
    variables: explicit dimensions derived from an excluded column are rejected, the
    default ones are dropped silently; without ``annual_inc`` the PTI policy-effect feature
    is dropped (every row would be invalid and the effect would fall back to no tilt)."""
    excluded = set((ds.params or {}).get("exclude_variables") or [])
    dims = (attrs.get("segmentation") or {}).get("dimensions") or None
    hierarchy = dict(attrs.get("hierarchy") or {})
    if not excluded:
        return dims, hierarchy
    order = (attrs.get("segmentation") or {}).get("refinement_order")
    if isinstance(order, list):
        gone = [d for d in order if d in DIMENSIONS and DIMENSIONS[d][0] in excluded]
        if gone:
            errors.setdefault("segmentation", []).append(
                f"refinement_order: {', '.join(gone)} derive(s) from variables excluded from dataset #{ds.pk}."
            )
    if dims:
        gone = [d for d in dims if d in DIMENSIONS and DIMENSIONS[d][0] in excluded]
        if gone:
            errors.setdefault("segmentation", []).append(
                f"{', '.join(gone)} derive(s) from variables excluded from dataset #{ds.pk} "
                f"({', '.join(sorted({DIMENSIONS[d][0] for d in gone}))})."
            )
    else:
        dims = [d for d in SegmentationConfig().dimensions if DIMENSIONS[d][0] not in excluded]
        if not dims:
            errors.setdefault("segmentation", []).append(
                f"Every default dimension derives from a variable excluded from dataset #{ds.pk}; choose dimensions."
            )
    if "annual_inc" in excluded and isinstance(hierarchy.get("policy_effect_cfg", {}), dict):
        pe = dict(hierarchy.get("policy_effect_cfg") or {})
        features = pe.get("features")
        if features is None:
            pe["features"] = [f for f in POLICY_EFFECT_FEATURES if f != "pti"]
            hierarchy["policy_effect_cfg"] = pe
        elif "pti" in features:
            errors.setdefault("hierarchy.policy_effect_cfg", []).append(
                f"pti needs annual_inc, which is excluded from dataset #{ds.pk}."
            )
    return dims, hierarchy


def segmentation_block(dims: list[str] | None, ds: Dataset, seg: dict | None = None) -> dict | None:
    """FrameworkConfig ``segmentation`` section for a dataset: the chosen dimensions, the
    refinement order a breaker ``resegment`` walks (the request's, else the default without the
    dimensions whose source variable the dataset excluded, so it never refines on a constant
    ``"unknown"`` column) and ``max_dimensions`` = dimensions + ``max_refinements`` when the
    request caps it (``None`` = all defaults)."""
    seg = seg or {}
    excluded = set((ds.params or {}).get("exclude_variables") or [])
    order = seg.get("refinement_order")
    cap = seg.get("max_refinements")
    if not dims and not excluded and order is None and cap is None:
        return None
    block: dict = {"dimensions": list(dims or SegmentationConfig().dimensions)}
    if order is not None:
        block["refinement_order"] = list(order)
    elif excluded:
        block["refinement_order"] = [
            d for d in SegmentationConfig().refinement_order if DIMENSIONS[d][0] not in excluded
        ]
    if cap is not None:
        block["max_dimensions"] = len(block["dimensions"]) + int(cap)
    return block


SEGMENTATION_KEYS = {"dimensions", "refinement_order", "max_refinements"}


def _distinct_dimensions(value, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(d in DIMENSIONS for d in value)
        and len(set(value)) == len(value)
    )


def validate_sections(attrs: dict) -> dict[str, list[str]]:
    """Field-level checks with dotted keys (``hierarchy.kappa_group``) — docs/API.md §1.2, §6.1."""
    errors: dict[str, list[str]] = {}

    def err(key: str, msg: str) -> None:
        errors.setdefault(key, []).append(msg)

    for section, keys in SECTION_KEYS.items():
        block = attrs.get(section)
        if block is None:
            continue
        if not isinstance(block, dict):
            err(section, "Must be an object.")
            continue
        unknown = sorted(set(block) - keys)
        if unknown:
            err(section, f"unknown key(s): {', '.join(unknown)}")
    seg = attrs.get("segmentation")
    if seg is not None:
        if not isinstance(seg, dict) or set(seg) - SEGMENTATION_KEYS:
            unknown = sorted(set(seg) - SEGMENTATION_KEYS) if isinstance(seg, dict) else []
            err("segmentation", f"unknown key(s): {', '.join(unknown)}" if unknown else "Must be an object.")
        else:
            dims = seg.get("dimensions", ...)
            if dims is not ... and not _distinct_dimensions(dims, allow_empty=False):
                err("segmentation", "Choose one or more distinct supported dimensions.")
            if "refinement_order" in seg and not _distinct_dimensions(seg["refinement_order"], allow_empty=True):
                err("segmentation", "refinement_order must be distinct supported dimensions.")
            cap = seg.get("max_refinements")
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap <= 10):
                err("segmentation", "max_refinements must be an integer in [0, 10] or null.")
    arm = attrs.get("arm_space") or {}
    if "amounts" in arm and not (
        isinstance(arm["amounts"], list) and arm["amounts"] and all((_number(x) or 0) > 0 for x in arm["amounts"])
    ):
        err("arm_space", "amounts must be positive numbers.")
    if "rates" in arm and not (
        isinstance(arm["rates"], list) and arm["rates"] and all(0 < (_number(x) or 0) < 1 for x in arm["rates"])
    ):
        err("arm_space", "rates are fractions in (0, 1), e.g. 0.13.")
    if "terms" in arm and not (
        isinstance(arm["terms"], list)
        and arm["terms"]
        and all((_number(x, integer=True) or 0) > 0 for x in arm["terms"])
    ):
        err("arm_space", "terms must be positive integers (months).")
    if "include_null" in arm and not isinstance(arm["include_null"], bool):
        err("arm_space", "include_null must be true or false.")
    for key in GUARDRAIL_KEYS:
        if key in arm and arm[key] is not None and not 0 < (_number(arm[key]) or 0) <= 10:
            err(f"arm_space.{key}", "Must be a positive fraction, or null to switch the guardrail off.")
    hier = attrs.get("hierarchy") or {}
    for key in ("kappa_group", "kappa_leaf"):
        if key in hier and not (hier[key] == "auto" or (_number(hier[key]) or 0) > 0):
            err(f"hierarchy.{key}", 'Must be > 0 or "auto".')
    for key in ("alpha0", "beta0", "kappa_min", "kappa_max"):
        if key in hier and not (_number(hier[key]) or 0) > 0:
            err(f"hierarchy.{key}", "Must be > 0.")
    if "policy_effect" in hier and not isinstance(hier["policy_effect"], bool):
        err("hierarchy.policy_effect", "Must be true or false.")
    if "policy_effect_cfg" in hier and not isinstance(hier["policy_effect_cfg"], dict):
        err("hierarchy.policy_effect_cfg", "Must be an object (partial PolicyEffectConfig).")
    rec = attrs.get("recovery") or {}
    if "min_obs" in rec and not (_number(rec["min_obs"], integer=True) or 0) >= 1:
        err("recovery.min_obs", "Must be an integer >= 1.")
    if "basis" in rec and rec["basis"] not in ("principal", "post_chargeoff"):
        err("recovery.basis", 'Must be "principal" or "post_chargeoff".')
    if "term_level" in rec and not isinstance(rec["term_level"], bool):
        err("recovery.term_level", "Must be true or false.")
    mctm = attrs.get("mctm") or {}
    if "structure" in mctm and mctm["structure"] not in ("proposal", "extended"):
        err("mctm.structure", 'Must be "proposal" or "extended".')
    if "prior_strength" in mctm and (_number(mctm["prior_strength"]) is None or _number(mctm["prior_strength"]) < 0):
        err("mctm.prior_strength", "Must be >= 0.")
    if "chargeoff_lag_months" in mctm and not (_number(mctm["chargeoff_lag_months"], integer=True) or 0) >= 1:
        err("mctm.chargeoff_lag_months", "Must be an integer >= 1.")
    reward = attrs.get("reward") or {}
    if (
        "funding_rate" in reward
        and not 0 <= (_number(reward["funding_rate"]) if _number(reward["funding_rate"]) is not None else -1) <= 0.5
    ):
        err("reward.funding_rate", "Annual fraction in [0, 0.5].")
    if "risk_aversion" in reward and (_number(reward["risk_aversion"]) is None or _number(reward["risk_aversion"]) < 0):
        err("reward.risk_aversion", "Must be >= 0.")
    if (
        "lendable_threshold" in reward
        and not 0
        <= (_number(reward["lendable_threshold"]) if _number(reward["lendable_threshold"]) is not None else -1)
        <= 1
    ):
        err("reward.lendable_threshold", "Must be in [0, 1].")
    if "exposure_basis" in reward and reward["exposure_basis"] not in ("full", "outstanding"):
        err("reward.exposure_basis", 'Must be "full" or "outstanding".')
    if "prepayment_adjustment" in reward and reward["prepayment_adjustment"] not in ("empirical", None):
        err("reward.prepayment_adjustment", 'Must be "empirical" or null.')
    return errors


class FitCreateSerializer(serializers.Serializer):
    """POST /fits/ body (types.ts ``FitCreate``): every config block is optional and partial."""

    dataset = serializers.IntegerField()
    name = serializers.CharField(max_length=200)
    segmentation = serializers.JSONField(
        required=False,
        help_text="{dimensions: SegmentationDimension[], refinement_order?: SegmentationDimension[], "
        "max_refinements?: int} (refinement_order / max_refinements govern a breaker resegment)",
    )
    arm_space = serializers.JSONField(required=False, help_text="Partial ArmSpaceConfig")
    hierarchy = serializers.JSONField(required=False, help_text="Partial HierarchyConfig (kappa_* may be 'auto')")
    recovery = serializers.JSONField(required=False, help_text="Partial RecoveryConfig")
    mctm = serializers.JSONField(required=False, help_text="Partial MctmConfig")
    reward = serializers.JSONField(required=False, help_text="Partial RewardConfig")
    train_end = serializers.CharField(required=False, help_text="YYYY-MM (YYYY-MM-DD accepted); default 2015-01")

    def validate_name(self, value: str) -> str:
        value = value.strip()
        if not value:
            raise serializers.ValidationError("This field may not be blank.")
        return value

    def validate(self, attrs):
        try:
            ds = Dataset.objects.select_related("job").get(pk=attrs["dataset"])
        except Dataset.DoesNotExist:
            raise serializers.ValidationError({"dataset": [f"dataset {attrs['dataset']} does not exist"]}) from None
        errors = validate_sections(attrs)
        dims, hierarchy = apply_exclusions(attrs, ds, errors) if not errors else (None, {})
        try:
            train_end = parse_epoch(attrs.get("train_end") or "2015-01", "train_end")
        except serializers.ValidationError as exc:
            errors.update(exc.detail)
            train_end = None
        if train_end and ds.issue_min and ds.issue_max and not (ds.issue_min < train_end <= ds.issue_max):
            errors.setdefault("train_end", []).append(
                f"Must be a YYYY-MM inside the dataset issue range ({ds.issue_min}..{ds.issue_max}]."
            )
        if errors:
            raise serializers.ValidationError(errors)
        rec = dict(attrs.get("recovery") or {})
        mctm = dict(attrs.get("mctm") or {})
        data = {"snapshot_date": ds.snapshot_date}  # censor where the dataset's file was observed
        if "basis" in rec:
            data["recovery_basis"] = rec["basis"]
        if "chargeoff_lag_months" in mctm:
            data["chargeoff_lag_months"] = mctm["chargeoff_lag_months"]
        raw = {
            "hierarchy": hierarchy,
            "recovery": rec,
            "mctm": mctm,
            "reward": attrs.get("reward") or {},
            "arms": attrs.get("arm_space") or {},
            "data": data,
            "train_end": epoch_start(train_end),
        }
        seg = segmentation_block(dims, ds, attrs.get("segmentation"))
        if seg:
            raw["segmentation"] = seg
        try:
            framework = FrameworkConfig.from_dict(raw)
            framework.validate()
        except ConfigurationError as exc:
            raise serializers.ValidationError({"non_field_errors": [str(exc)], "code": "ConfigurationError"}) from None
        except (TypeError, ValueError) as exc:
            raise serializers.ValidationError({"non_field_errors": [f"invalid configuration: {exc}"]}) from None
        attrs["dataset_obj"] = ds
        attrs["framework"] = framework
        attrs["train_end_epoch"] = train_end
        attrs["config"] = contract_config(framework, train_end)
        return attrs


class FitSerializer(serializers.ModelSerializer):
    """types.ts ``Fit``."""

    dataset_name = serializers.CharField(source="dataset.name", read_only=True)
    status = serializers.ChoiceField(source="current_status", choices=list(BUILD_STATUSES), read_only=True)
    config = serializers.SerializerMethodField()
    metrics = serializers.SerializerMethodField()
    job = JobSerializer(read_only=True, allow_null=True)

    class Meta:
        model = Fit
        fields = [
            "id",
            "name",
            "dataset",
            "dataset_name",
            "status",
            "config",
            "metrics",
            "n_groups",
            "n_arms",
            "created_at",
            "job",
        ]
        read_only_fields = fields

    @extend_schema_field(S.FitConfigSerializer)
    def get_config(self, obj: Fit):
        return obj.config

    @extend_schema_field(S.FitMetricsSerializer(allow_null=True))
    def get_metrics(self, obj: Fit):
        return obj.metrics if obj.current_status == "ready" else None


class PortfolioRequestSerializer(serializers.Serializer):
    """POST /fits/{id}/shadow/portfolio/ body."""

    max_loans = serializers.IntegerField(min_value=100, max_value=500_000, required=False, default=5000)
    seed = serializers.IntegerField(required=False, default=42)
    filters = S.PortfolioFiltersSerializer(required=False)

    def validate_filters(self, value):
        value = dict(value or {})
        for key in ("issue_from", "issue_to"):
            if value.get(key):
                value[key] = parse_epoch(value[key], f"filters.{key}")
        return value


class ShadowEvaluateSerializer(serializers.Serializer):
    """POST /fits/{id}/shadow/evaluate/ body: ``{loan_id}`` or ``{context, historical_policy?}``."""

    loan_id = serializers.IntegerField(required=False)
    context = serializers.DictField(required=False, help_text="PRE_DECISION columns only (others → 400 LeakageError)")
    historical_policy = S.LendingPolicySerializer(required=False, allow_null=True)

    def validate(self, attrs):
        if attrs.get("loan_id") is None and attrs.get("context") is None:
            raise serializers.ValidationError({"non_field_errors": ["Provide either loan_id or context."]})
        return attrs
