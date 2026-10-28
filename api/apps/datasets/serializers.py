"""types.ts ``Dataset``, ``DatasetCreate``, ``VariableRow``, ``DatasetSummary``, ``LoanRecord``."""

from __future__ import annotations

from pathlib import Path

from bayesian_credit.data.schema import NUMERIC_COLUMNS, PRE_DECISION_COLUMNS
from bayesian_credit.data.synthetic import DEFAULT_SHIFT, DEFAULT_SHIFT_AT
from django.conf import settings
from rest_framework import serializers

from apps.common.mixins import BUILD_STATUSES
from apps.common.serializers import JobSerializer
from apps.common.utils import parse_epoch

from .models import Dataset

SOURCES = ["synthetic", "lending_club"]
SHIFTABLE = (set(PRE_DECISION_COLUMNS) & set(NUMERIC_COLUMNS)) - {"installment", "fico_range_high"}
SHIFT_EXTRA = {"grade_mix", "latent"}


class DistributionShiftSerializer(serializers.Serializer):
    at = serializers.CharField(help_text="First affected issue month (YYYY-MM).")
    deltas = serializers.DictField(child=serializers.FloatField(), help_text="feature -> additive shift")


class DatasetCreateSerializer(serializers.Serializer):
    """POST /datasets/ body (types.ts ``DatasetCreate``)."""

    name = serializers.CharField(max_length=200)
    source = serializers.ChoiceField(choices=SOURCES)
    path = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    n = serializers.IntegerField(required=False, allow_null=True, min_value=1000, max_value=2_500_000)
    seed = serializers.IntegerField(required=False, allow_null=True, min_value=0)
    max_rows = serializers.IntegerField(required=False, allow_null=True, min_value=1000)
    sample_fraction = serializers.FloatField(required=False, allow_null=True)
    issue_from = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    issue_to = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    shift = serializers.JSONField(required=False, allow_null=True)
    shift_at = serializers.CharField(required=False, allow_null=True, write_only=True, help_text="alias of shift.at")

    def validate_name(self, value: str) -> str:
        value = value.strip()
        if not value:
            raise serializers.ValidationError("This field may not be blank.")
        return value

    def validate_sample_fraction(self, value):
        if value is not None and not 0 < value <= 1:
            raise serializers.ValidationError("Must be in (0, 1].")
        return value

    def validate(self, attrs):
        source = attrs["source"]
        synthetic = source == "synthetic"
        errors: dict[str, list[str]] = {}
        issue_from = attrs.get("issue_from") or None
        issue_to = attrs.get("issue_to") or None
        try:
            issue_from = parse_epoch(issue_from, "issue_from") if issue_from else None
            issue_to = parse_epoch(issue_to, "issue_to") if issue_to else None
        except serializers.ValidationError as exc:
            raise serializers.ValidationError(exc.detail) from None
        # docs/API.md §5.1: both sources default to 2012-01 .. 2018-12 (inclusive)
        issue_from = issue_from or "2012-01"
        issue_to = issue_to or "2018-12"
        if synthetic and issue_to > "2019-03":
            errors.setdefault("issue_to", []).append("Synthetic loans cannot be issued after the 2019-03 snapshot.")
        if not synthetic and not ("2007-06" <= issue_from and issue_to <= "2018-12"):
            errors.setdefault("issue_from", []).append("Lending Club issue dates span 2007-06 to 2018-12.")
        if issue_from > issue_to:
            errors.setdefault("issue_to", []).append("Must not precede issue_from.")

        path = None
        if not synthetic:
            raw = (attrs.get("path") or "").strip()
            candidate = Path(raw).expanduser() if raw else Path(settings.LENDING_CLUB_PATH)
            if not candidate.is_absolute():
                candidate = Path(settings.DATA_ROOT) / candidate
            candidate = candidate.resolve()
            data_root = Path(settings.DATA_ROOT).resolve()
            if candidate != Path(settings.LENDING_CLUB_PATH).resolve() and data_root not in candidate.parents:
                errors.setdefault("path", []).append("The file must live under the server's DATA_ROOT.")
            elif not candidate.is_file():
                errors.setdefault("path", []).append(
                    f"File not found: {candidate}. Run scripts/download_data.sh or use source='synthetic'."
                )
            path = str(candidate)

        shift = None
        raw_shift = attrs.get("shift")
        shift_at = attrs.get("shift_at")
        if synthetic and (raw_shift or shift_at):
            if raw_shift is not None and not isinstance(raw_shift, dict):
                errors.setdefault("shift", []).append("Must be an object {at, deltas}.")
            else:
                raw_shift = dict(raw_shift or {})
                if "at" in raw_shift or "deltas" in raw_shift:
                    at, deltas = raw_shift.get("at") or shift_at, raw_shift.get("deltas") or {}
                else:  # core style: {"fico_range_low": -25, ...} plus top-level shift_at
                    at, deltas = shift_at, raw_shift
                at = at or DEFAULT_SHIFT_AT
                try:
                    at = parse_epoch(at, "shift")
                except serializers.ValidationError:
                    errors.setdefault("shift", []).append("shift.at must be YYYY-MM.")
                    at = None
                if at and not (issue_from < at <= issue_to):
                    errors.setdefault("shift", []).append("shift.at must be a YYYY-MM inside the issue range.")
                clean_deltas: dict[str, float] = {}
                if not isinstance(deltas, dict):
                    errors.setdefault("shift", []).append("shift.deltas must be an object of numbers.")
                    deltas = {}
                for key, value in deltas.items():
                    if key not in SHIFTABLE and key not in SHIFT_EXTRA:
                        errors.setdefault("shift", []).append(
                            f"{key!r} cannot be shifted (numeric pre-decision columns only)."
                        )
                        continue
                    try:
                        clean_deltas[key] = float(value)
                    except (TypeError, ValueError):
                        errors.setdefault("shift", []).append(f"shift.deltas.{key} must be a number.")
                shift = {"at": at, "deltas": clean_deltas or dict(DEFAULT_SHIFT)}
        if errors:
            raise serializers.ValidationError(errors)

        return {
            "name": attrs["name"],
            "source": source,
            "path": path,
            "n": (attrs.get("n") or 200_000) if synthetic else None,
            "seed": attrs.get("seed") if attrs.get("seed") is not None else 42,
            "max_rows": None if synthetic else attrs.get("max_rows"),
            "sample_fraction": attrs.get("sample_fraction"),
            "issue_from": issue_from,
            "issue_to": issue_to,
            "shift": shift,
        }


class DatasetSerializer(serializers.ModelSerializer):
    """types.ts ``Dataset``."""

    source = serializers.ChoiceField(choices=SOURCES, read_only=True)
    status = serializers.ChoiceField(source="current_status", choices=list(BUILD_STATUSES), read_only=True)
    params = DatasetCreateSerializer(read_only=True)
    job = JobSerializer(read_only=True, allow_null=True)

    class Meta:
        model = Dataset
        fields = [
            "id",
            "name",
            "source",
            "status",
            "params",
            "n_loans",
            "n_resolved",
            "issue_min",
            "issue_max",
            "resolved_share",
            "default_rate",
            "status_counts",
            "created_at",
            "job",
        ]
        read_only_fields = fields

    def to_representation(self, instance: Dataset):
        data = super().to_representation(instance)
        data["params"] = dict(instance.params or {})
        if data["status"] != "ready":
            for key in ("n_loans", "n_resolved", "issue_min", "issue_max", "resolved_share", "default_rate"):
                data[key] = None
            data["status_counts"] = {}
        return data


class VariableRowSerializer(serializers.Serializer):
    name = serializers.CharField()
    stage = serializers.ChoiceField(choices=["pre_decision", "policy", "disbursement", "interim", "terminal"])
    primary_use = serializers.CharField()
    kind = serializers.ChoiceField(choices=["numeric", "categorical", "date"])
    present = serializers.BooleanField()
    dtype = serializers.CharField(allow_null=True)
    null_share = serializers.FloatField(allow_null=True)
