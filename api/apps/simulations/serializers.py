"""types.ts ``SimulationCreate`` (validation) and ``Simulation`` (documentation)."""

from __future__ import annotations

from bayesian_credit.config import FrameworkConfig
from bayesian_credit.data.schema import PRE_DECISION_COLUMNS
from bayesian_credit.exceptions import ConfigurationError
from bayesian_credit.simulation.policies import parse_policy_specs
from rest_framework import serializers

from apps.common import schemas as S
from apps.common.serializers import JobSerializer
from apps.common.utils import epoch_start, parse_epoch, to_epoch
from apps.datasets.models import Dataset
from apps.fits.models import Fit
from apps.fits.serializers import apply_exclusions, segmentation_block, validate_sections

from .services import default_psi_settings

FEEDBACK_MODES = ["simulated", "replay", "hybrid"]
#: First streamed month of a fit-less cold start (the loans before it build the replay world).
COLD_START_DEFAULT = "2015-01"
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
    """POST /simulations/ body.

    A warm run names a ``fit`` (its trained model warm-starts the policies). A cold start
    (``cold_start: true``) runs the policies without an offline model; it names a ``fit`` to
    reuse that fit's configuration and replay world, or just a ``dataset``: the replay world
    is then built from the loans issued before ``start`` with the default configuration and
    the given ``segmentation`` (default: the dataset's default dimensions)."""

    fit = serializers.IntegerField(required=False, allow_null=True)
    dataset = serializers.IntegerField(
        required=False, allow_null=True, help_text="Cold start only: replay this dataset without a fit"
    )
    cold_start = serializers.BooleanField(
        required=False,
        default=False,
        help_text="Policies start from a flat prior and learn online only (no risk-model warm start)",
    )
    segmentation = serializers.JSONField(
        required=False,
        allow_null=True,
        help_text="Fit-less cold start only: {dimensions: SegmentationDimension[]} of the replay world and the "
        "policies' groups (default: the dataset's default dimensions)",
    )
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

    def _sources(self, attrs) -> tuple[Fit | None, Dataset, FrameworkConfig]:
        """(fit, dataset, framework) of the run; a fit-less cold start gets the default
        configuration with the dataset's snapshot and variable selection."""
        cold = bool(attrs.get("cold_start"))
        if attrs.get("fit") is not None:
            try:
                fit = Fit.objects.select_related("dataset", "job").get(pk=attrs["fit"])
            except Fit.DoesNotExist:
                raise serializers.ValidationError({"fit": [f"fit {attrs['fit']} does not exist"]}) from None
            if attrs.get("segmentation") is not None:
                raise serializers.ValidationError(
                    {"segmentation": [f"The run uses fit {fit.pk}'s segmentation; omit segmentation or the fit."]}
                )
            if attrs.get("dataset") is not None and attrs["dataset"] != fit.dataset_id:
                raise serializers.ValidationError(
                    {"dataset": [f"fit {fit.pk} was trained on dataset {fit.dataset_id}; omit dataset or match it"]}
                )
            return fit, fit.dataset, FrameworkConfig.from_dict(fit.framework or {})
        if not cold:
            raise serializers.ValidationError(
                {"fit": ["A fit is required unless cold_start is true (then a dataset is enough)."]}
            )
        if attrs.get("dataset") is None:
            raise serializers.ValidationError({"dataset": ["A cold start without a fit needs a dataset."]})
        try:
            ds = Dataset.objects.select_related("job").get(pk=attrs["dataset"])
        except Dataset.DoesNotExist:
            raise serializers.ValidationError({"dataset": [f"dataset {attrs['dataset']} does not exist"]}) from None
        given = {"segmentation": attrs["segmentation"]} if attrs.get("segmentation") is not None else {}
        errors = validate_sections(given)
        dims, hierarchy = apply_exclusions(given, ds, errors) if not errors else (None, {})
        if errors:
            raise serializers.ValidationError(errors)
        raw: dict = {"hierarchy": hierarchy, "data": {"snapshot_date": ds.snapshot_date}}
        seg = segmentation_block(dims, ds)
        if seg:
            raw["segmentation"] = seg
        return None, ds, FrameworkConfig.from_dict(raw)

    def validate(self, attrs):
        fit, ds, framework = self._sources(attrs)
        errors: dict[str, list[str]] = {}
        start = parse_epoch(attrs.get("start") or (fit.train_end if fit is not None else COLD_START_DEFAULT), "start")
        end = parse_epoch(attrs.get("end") or ds.issue_max or "2018-12", "end")
        if start > end:
            errors.setdefault("start", []).append("start must not be after end")
        if ds.issue_min and start <= ds.issue_min:
            errors.setdefault("start", []).append(f"start must leave a reference window after {ds.issue_min}")
        if ds.issue_max and end > ds.issue_max:
            errors.setdefault("end", []).append(f"end must not exceed the last dataset epoch {ds.issue_max}")
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
        excluded = set((ds.params or {}).get("exclude_variables") or [])
        if excluded and "features" in given:
            gone = [f for f in psi["features"] if f in excluded]
            if gone:
                errors.setdefault("psi", []).append(
                    f"PSI features excluded from dataset #{ds.pk}: {', '.join(gone)}."
                )
        elif excluded:  # the fit's defaults: monitor what is left
            psi["features"] = [f for f in psi["features"] if f not in excluded]
            if not psi["features"]:
                errors.setdefault("psi", []).append(
                    f"Every default PSI feature is excluded from dataset #{ds.pk}; choose features."
                )
        if errors:
            raise serializers.ValidationError(errors)
        if fit is None:
            # the replay world's reference is fitted on the loans issued before the stream
            framework.train_end = epoch_start(start)
        attrs["fit_obj"] = fit
        attrs["dataset_obj"] = ds
        attrs["framework"] = framework.to_dict()
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
            "cold_start": bool(attrs.get("cold_start")),
        }
        return attrs


class SimulationSerializer(serializers.Serializer):
    """types.ts ``Simulation`` (documentation; responses are built by ``services.simulation_payload``)."""

    id = serializers.IntegerField()
    name = serializers.CharField()
    fit = serializers.IntegerField(allow_null=True, help_text="null for a cold start without a fit")
    fit_name = serializers.CharField(allow_null=True)
    dataset = serializers.IntegerField()
    dataset_name = serializers.CharField()
    cold_start = serializers.BooleanField()
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
