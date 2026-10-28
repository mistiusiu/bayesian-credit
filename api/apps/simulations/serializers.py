"""types.ts ``SimulationCreate`` (validation) and ``Simulation`` (documentation)."""

from __future__ import annotations

from bayesian_credit.config import FrameworkConfig
from bayesian_credit.data.schema import PRE_DECISION_COLUMNS
from bayesian_credit.exceptions import ConfigurationError
from bayesian_credit.simulation.policies import parse_policy_specs
from rest_framework import serializers

from apps.common import schemas as S
from apps.common.serializers import JobSerializer
from apps.common.utils import parse_epoch, to_epoch
from apps.fits.models import Fit

from .services import default_psi_settings

FEEDBACK_MODES = ["simulated", "replay", "hybrid"]
BREAKER_MODES = ["pause", "auto_ack", "off"]


class PsiSettingsInputSerializer(serializers.Serializer):
    """Partial ``PsiSettings`` (core-style flat names ``watch_threshold`` / ``breaker_threshold`` accepted)."""

    features = serializers.ListField(child=serializers.CharField(), required=False, allow_empty=False)
    window_months = serializers.IntegerField(required=False, min_value=1, max_value=36)
    thresholds = serializers.DictField(child=serializers.FloatField(), required=False)
    watch_threshold = serializers.FloatField(required=False)
    breaker_threshold = serializers.FloatField(required=False)
    aggregate = serializers.ChoiceField(choices=["max", "mean"], required=False)
    recovery_readings = serializers.IntegerField(required=False, min_value=0, max_value=60)
    fallback = serializers.ChoiceField(choices=["champion_snapshot", "decline", "historical"], required=False)
    auto_action = serializers.ChoiceField(choices=["resume", "resegment", "rollback"], required=False)
    auto_ack_epochs = serializers.IntegerField(required=False, min_value=0, max_value=60)
    n_bins = serializers.IntegerField(required=False, min_value=2, max_value=50)


class SimulationCreateSerializer(serializers.Serializer):
    """POST /simulations/ body."""

    fit = serializers.IntegerField()
    name = serializers.CharField(max_length=200)
    policies = serializers.ListField(child=serializers.CharField(help_text=S.POLICY_HELP), allow_empty=False)
    feedback_mode = serializers.ChoiceField(choices=FEEDBACK_MODES, required=False, default="simulated")
    start = serializers.CharField(
        required=False, help_text="YYYY-MM (YYYY-MM-DD accepted); default: the fit's train_end"
    )
    end = serializers.CharField(required=False, help_text="YYYY-MM (YYYY-MM-DD accepted); default: last dataset epoch")
    max_loans = serializers.IntegerField(required=False, allow_null=True, min_value=100, default=50_000)
    seed = serializers.IntegerField(required=False, default=42)
    interim_months = serializers.IntegerField(required=False, min_value=0, max_value=60, default=12)
    interim_weight = serializers.FloatField(required=False, min_value=0.0, max_value=1.0, default=0.5)
    psi = PsiSettingsInputSerializer(required=False)
    breaker_mode = serializers.ChoiceField(choices=BREAKER_MODES, required=False, default="auto_ack")
    checkpoint_every = serializers.IntegerField(required=False, min_value=1, max_value=120, default=6)
    hindsight_sample = serializers.ChoiceField(
        choices=["elapsed", "resolved"],
        required=False,
        default="elapsed",
        help_text="Loans the hindsight yardstick is fitted on: resolved loans whose term elapsed by the snapshot "
        "(unbiased, default) or every resolved loan (biased towards early defaults)",
    )

    def validate_name(self, value: str) -> str:
        value = value.strip()
        if not value:
            raise serializers.ValidationError("This field may not be blank.")
        return value

    def validate_policies(self, value: list[str]) -> list[str]:
        try:
            specs = parse_policy_specs(value)
        except ConfigurationError as exc:
            raise serializers.ValidationError(str(exc)) from None
        return [s.label for s in specs]

    def validate(self, attrs):
        try:
            fit = Fit.objects.select_related("dataset", "job").get(pk=attrs["fit"])
        except Fit.DoesNotExist:
            raise serializers.ValidationError({"fit": [f"fit {attrs['fit']} does not exist"]}) from None
        ds = fit.dataset
        errors: dict[str, list[str]] = {}
        start = parse_epoch(attrs.get("start") or fit.train_end, "start")
        end = parse_epoch(attrs.get("end") or ds.issue_max or "2018-12", "end")
        if start > end:
            errors.setdefault("start", []).append("start must not be after end")
        if ds.issue_min and start <= ds.issue_min:
            errors.setdefault("start", []).append(f"start must leave a reference window after {ds.issue_min}")
        if ds.issue_max and end > ds.issue_max:
            errors.setdefault("end", []).append(f"end must not exceed the last dataset epoch {ds.issue_max}")
        framework = FrameworkConfig.from_dict(fit.framework or {})
        psi = default_psi_settings(framework)
        given = dict(attrs.get("psi") or {})
        thresholds = dict(psi["thresholds"])
        thresholds.update({k: v for k, v in (given.pop("thresholds", None) or {}).items() if k in ("watch", "breaker")})
        if "watch_threshold" in given:
            thresholds["watch"] = given.pop("watch_threshold")
        if "breaker_threshold" in given:
            thresholds["breaker"] = given.pop("breaker_threshold")
        psi.update(given)
        psi["thresholds"] = thresholds
        if not 0 < thresholds["watch"] < thresholds["breaker"]:
            errors.setdefault("psi", []).append("Need 0 < thresholds.watch < thresholds.breaker.")
        unknown = [f for f in psi["features"] if f not in PRE_DECISION_COLUMNS]
        if unknown:
            errors.setdefault("psi", []).append(
                f"PSI features must be pre-decision columns; unknown: {', '.join(unknown)}"
            )
        if errors:
            raise serializers.ValidationError(errors)
        attrs["fit_obj"] = fit
        attrs["config"] = {
            "policies": attrs["policies"],
            "feedback_mode": attrs.get("feedback_mode", "simulated"),
            "start": to_epoch(start),
            "end": to_epoch(end),
            "max_loans": attrs.get("max_loans"),
            "seed": attrs.get("seed", 42),
            "interim_months": attrs.get("interim_months", 12),
            "interim_weight": attrs.get("interim_weight", 0.5),
            "psi": psi,
            "breaker_mode": attrs.get("breaker_mode", "auto_ack"),
            "checkpoint_every": attrs.get("checkpoint_every", 6),
            "hindsight_sample": attrs.get("hindsight_sample", "elapsed"),
        }
        return attrs


class SimulationSerializer(serializers.Serializer):
    """types.ts ``Simulation`` (documentation; responses are built by ``services.simulation_payload``)."""

    id = serializers.IntegerField()
    name = serializers.CharField()
    fit = serializers.IntegerField()
    fit_name = serializers.CharField()
    dataset = serializers.IntegerField()
    status = serializers.ChoiceField(
        choices=["queued", "running", "paused", "awaiting_review", "interrupted", "completed", "failed", "cancelled"]
    )
    progress = serializers.FloatField()
    config = S.SimulationConfigSerializer()
    epochs_total = serializers.IntegerField()
    epochs_done = serializers.IntegerField()
    current_epoch = serializers.CharField(allow_null=True)
    breaker_state = serializers.ChoiceField(choices=["closed", "open", "half_open"])
    sentinel_level = serializers.ChoiceField(choices=["normal", "watch", "breaker"])
    psi_aggregate = serializers.FloatField(allow_null=True)
    pending_reviews = serializers.IntegerField()
    last_checkpoint_epoch = serializers.CharField(allow_null=True)
    restarts = serializers.IntegerField()
    worker = S.SimulationWorkerSerializer()
    last_recovery = S.RecoveryInfoSerializer(allow_null=True)
    headline = S.PolicyHeadlineSerializer(many=True)
    uplift_vs_historical = serializers.FloatField(allow_null=True)
    created_at = serializers.DateTimeField()
    started_at = serializers.DateTimeField(allow_null=True)
    finished_at = serializers.DateTimeField(allow_null=True)
    job = JobSerializer(allow_null=True)
