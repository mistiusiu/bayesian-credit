"""Cheap contract check: every property of the main ``types.ts`` interfaces appears in the
corresponding payload served by the API (extra keys are allowed, missing keys are not).

Property names are extracted with a regex: the top-level members of ``export interface X
{ ... }`` (``extends`` parents merged). Payloads come from the session-wide ``e2e`` run.
"""

from __future__ import annotations

import re

import pytest

from conftest import TYPES_TS

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(not TYPES_TS.is_file(), reason="frontend/src/api/types.ts not available"),
]

_INTERFACE = re.compile(r"export interface (\w+)(?:<[^>]*>)?(?:\s+extends\s+(\w+))?\s*\{")
_MEMBER = re.compile(r"^\s*(?:readonly\s+)?(\w+)\??\s*:", re.M)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def interfaces() -> dict[str, set[str]]:
    src = _strip_comments(TYPES_TS.read_text(encoding="utf-8"))
    raw: dict[str, tuple[str | None, set[str]]] = {}
    for m in _INTERFACE.finditer(src):
        i, depth = m.end(), 1
        top_level = []
        while depth and i < len(src):
            ch = src[i]
            if ch in "{(<[":
                depth += 1
            elif ch in "})>]":
                depth -= 1
            if depth == 1 or (depth == 0 and ch == "}"):
                top_level.append(ch if depth == 1 else "")
            else:
                top_level.append(" ")
            i += 1
        body = "".join(top_level)
        keys = set(_MEMBER.findall(body))
        raw[m.group(1)] = (m.group(2), keys)

    def resolve(name: str) -> set[str]:
        parent, keys = raw[name]
        return keys | (resolve(parent) if parent in raw else set())

    return {name: resolve(name) for name in raw}


@pytest.fixture(scope="module")
def ts_interfaces():
    found = interfaces()
    assert {
        "Dataset",
        "Fit",
        "Simulation",
        "Job",
        "Review",
        "TimeseriesRow",
        "ShadowReport",
        "Mctm",
        "HierarchyRow",
    } <= set(found)
    return found


def _check(ts_interfaces, name: str, payload: dict) -> None:
    expected = ts_interfaces[name]
    missing = sorted(expected - set(payload))
    assert not missing, f"{name} payload is missing {missing}"


def test_parser_reads_nested_members_correctly(ts_interfaces):
    assert "results" in ts_interfaces["Paginated"]
    sim = ts_interfaces["Simulation"]
    assert {"worker", "last_recovery", "headline"} <= sim and "alive" not in sim  # nested keys are not top-level
    assert ts_interfaces["FitArmRow"] >= ts_interfaces["ArmRow"]  # extends merged


def test_main_interfaces(ts_interfaces, e2e):
    _check(ts_interfaces, "Dataset", e2e["dataset"])
    _check(ts_interfaces, "Job", e2e["dataset"]["job"])
    _check(ts_interfaces, "Fit", e2e["fit"])
    _check(ts_interfaces, "FitMetrics", e2e["fit"]["metrics"])
    _check(ts_interfaces, "Simulation", e2e["simulation"])
    _check(ts_interfaces, "TimeseriesRow", e2e["timeseries"]["rows"][0])
    _check(ts_interfaces, "ShadowReport", e2e["shadow_loan"])
    _check(ts_interfaces, "ShadowReport", e2e["shadow_context"])
    _check(ts_interfaces, "Mctm", e2e["mctm"])
    _check(ts_interfaces, "HierarchyRow", e2e["groups"][0])
    _check(ts_interfaces, "HierarchyRow", e2e["leaves"][0])
    _check(ts_interfaces, "Review", e2e["reviews"]["results"][0])


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("Health", ("health",)),
        ("VariableRow", ("variables", 0)),
        ("DatasetSummary", ("summary",)),
        ("NumericFeatureSummary", ("summary", "features", 0)),
        ("MonthlyArrival", ("summary", "monthly_arrivals", 0)),
        ("Paginated", ("loans",)),
        ("LoanRecord", ("loans", "results", 0)),
        ("BorrowerContext", ("loans", "results", 0, "context")),
        ("LoanOutcome", ("loans", "results", 0, "outcome")),
        ("FitConfig", ("fit", "config")),
        ("RiskEstimate", ("risk",)),
        ("RewardBreakdown", ("risk", "reward")),
        ("Calibration", ("calibration",)),
        ("RecoveryFit", ("calibration", "recovery")),
        ("ReliabilityBin", ("calibration", "bins", 0)),
        ("RocPoint", ("calibration", "roc", 0)),
        ("MctmTrajectoryRow", ("mctm", "trajectory", 0)),
        ("PDefaultPoint", ("mctm", "p_default_curve", 0)),
        ("FitArms", ("arms",)),
        ("FitArmRow", ("arms", "arms", 0)),
        ("ShadowArmRow", ("shadow_loan", "arms", 1)),
        ("LoanOutcome", ("shadow_loan", "outcome")),
        ("PortfolioResult", ("portfolio",)),
        ("PortfolioSummary", ("portfolio", "summary")),
        ("PortfolioGradeRow", ("portfolio", "by_grade", 0)),
        ("InclusionRow", ("portfolio", "inclusion", 0)),
        ("PortfolioLoanRow", ("portfolio", "top_loans", 0)),
        ("PortfolioRun", ("portfolio_runs", 0)),
        ("SimulationConfig", ("simulation", "config")),
        ("PsiSettings", ("simulation", "config", "psi")),
        ("PolicyHeadline", ("simulation", "headline", 0)),
        ("Timeseries", ("timeseries",)),
        ("SimArms", ("sim_arms",)),
        ("SimArmRow", ("sim_arms", "arms", 5)),
        ("PosteriorHistory", ("posteriors",)),
        ("PosteriorSnapshotRow", ("posteriors", "rows", 0)),
        ("SimpsonPanel", ("simpson",)),
        ("SimpsonGroupRow", ("simpson", "groups", 0)),
        ("EventPage", ("events",)),
        ("SimulationEvent", ("events", "results", 0)),
        ("SentinelTimeline", ("sentinel",)),
        ("PsiPoint", ("sentinel", "features_psi", 0)),
        ("PsiAggregatePoint", ("sentinel", "aggregate", 0)),
        ("PsiBucketTable", ("buckets",)),
        ("PsiBucket", ("buckets", "buckets", 0)),
        ("EvaluationReport", ("report",)),
        ("ReportDimension", ("report", "dimensions", 0)),
        ("ReportMetric", ("report", "dimensions", 0, "metrics", 0)),
        ("RuntimeStats", ("report", "runtime")),
    ],
)
def test_secondary_interfaces(ts_interfaces, e2e, name, path):
    node = e2e
    for key in path:
        node = node[key]
    _check(ts_interfaces, name, node)


def test_breaker_transitions_shape(ts_interfaces, e2e):
    transitions = e2e["sentinel"]["transitions"]
    if not transitions:
        pytest.skip("this run had no breaker transition")
    _check(ts_interfaces, "BreakerTransition", transitions[0])
