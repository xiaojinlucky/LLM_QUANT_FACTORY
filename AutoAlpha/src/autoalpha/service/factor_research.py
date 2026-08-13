"""Bounded, single-researcher factor discovery MVP.

The module deliberately keeps orchestration small.  The existing DSL,
``PriceVolumeEvaluator``, ``ServiceStore`` and ``FactorRegistry`` remain the
authoritative implementations for expression safety, evaluation, persistence
and versioned factor artifacts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autoalpha.dsl.expression import Expression, FactorDefinition, canonical_family, field
from autoalpha.registry.store import FactorRegistry
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.openai_client import CompatibleChatClient, GeneratedProposal
from autoalpha.service.store import ServiceStore

KEEP = "KEEP"
INVALID = "INVALID"
DUPLICATE = "DUPLICATE"
TRAIN_FAILED = "TRAIN_FAILED"
VALIDATION_FAILED = "VALIDATION_FAILED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
REJECTED = "REJECTED"

_REJECTED_STATUSES = frozenset(
    {INVALID, DUPLICATE, TRAIN_FAILED, VALIDATION_FAILED, BUDGET_EXHAUSTED}
)
_DEFAULT_CORRELATION_THRESHOLD = 0.85


class FactorResearcher(Protocol):
    """The one LLM Researcher (or a deterministic test double) used by the MVP."""

    def propose_batch(
        self,
        research_direction: str,
        candidate_count: int,
        round_number: int,
        context: dict[str, Any],
    ) -> Sequence[Mapping[str, Any]]: ...

    def repair(
        self,
        proposal: Mapping[str, Any],
        feedback: str,
        context: dict[str, Any],
    ) -> Mapping[str, Any] | None: ...

    def conclude(self, summary: Mapping[str, Any]) -> str: ...


@dataclass(frozen=True)
class FactorResearchBudget:
    candidate_count: int = 4
    rounds: int = 3
    maximum_repairs_per_candidate: int = 1
    maximum_llm_calls: int | None = None
    maximum_candidate_evaluations: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.candidate_count <= 100:
            raise ValueError("candidate_count must be between 1 and 100")
        if not 1 <= self.rounds <= 20:
            raise ValueError("rounds must be between 1 and 20")
        if self.maximum_repairs_per_candidate != 1:
            raise ValueError("MVP supports exactly one repair per candidate")
        default_calls = (
            self.candidate_count * (1 + self.maximum_repairs_per_candidate) + self.rounds + 1
        )
        default_evaluations = self.candidate_count * (1 + self.maximum_repairs_per_candidate)
        if self.maximum_llm_calls is not None and self.maximum_llm_calls < 1:
            raise ValueError("maximum_llm_calls must be positive")
        if (
            self.maximum_candidate_evaluations is not None
            and self.maximum_candidate_evaluations < 1
        ):
            raise ValueError("maximum_candidate_evaluations must be positive")
        object.__setattr__(
            self,
            "maximum_llm_calls",
            self.maximum_llm_calls if self.maximum_llm_calls is not None else default_calls,
        )
        object.__setattr__(
            self,
            "maximum_candidate_evaluations",
            self.maximum_candidate_evaluations
            if self.maximum_candidate_evaluations is not None
            else default_evaluations,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_count": self.candidate_count,
            "rounds": self.rounds,
            "maximum_repairs_per_candidate": self.maximum_repairs_per_candidate,
            "maximum_llm_calls": int(self.maximum_llm_calls or 0),
            "maximum_candidate_evaluations": int(self.maximum_candidate_evaluations or 0),
        }


@dataclass
class _RunBudget:
    llm_calls: int = 0
    candidate_evaluations: int = 0
    repair_counts: dict[str, int] = dataclass_field(default_factory=dict)
    exhausted: list[str] = dataclass_field(default_factory=list)


@dataclass
class _CandidateOutcome:
    ordinal: int
    round_number: int
    status: str
    reason: str
    proposal: dict[str, Any]
    candidate_id: str
    factor: FactorDefinition | None = None
    train_metrics: dict[str, Any] = dataclass_field(default_factory=dict)
    validation_metrics: dict[str, Any] = dataclass_field(default_factory=dict)
    core_metrics: dict[str, Any] = dataclass_field(default_factory=dict)
    repair_count: int = 0
    parent_factor_id: str | None = None
    duplicate_of: str | None = None
    registry_artifact_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "ordinal": self.ordinal,
                "round": self.round_number,
                "status": self.status,
                "reason": self.reason,
                "candidate_id": self.candidate_id,
                "proposal": self.proposal,
                "train_metrics": self.train_metrics,
                "validation_metrics": self.validation_metrics,
                "core_metrics": self.core_metrics,
                "repair_count": self.repair_count,
                "parent_factor_id": self.parent_factor_id,
                "duplicate_of": self.duplicate_of,
                "registry_artifact_hash": self.registry_artifact_hash,
            }
        )


class CompatibleChatResearcher:
    """Adapt the existing OpenAI-compatible client to the MVP's one role."""

    def __init__(self, client: CompatibleChatClient) -> None:
        self.client = client

    def propose_batch(
        self,
        research_direction: str,
        candidate_count: int,
        round_number: int,
        context: dict[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        memories = [
            {"kind": "mvp_research_direction", "content": research_direction},
            {"kind": "mvp_context", "content": context},
        ]
        proposals = _run_async(
            self.client.propose_batch(
                memories,
                round_number,
                batch_size=min(candidate_count, 5),
                data_context={"research_direction": research_direction, **context},
            )
        )
        return [proposal.raw for proposal in proposals]

    def repair(
        self,
        proposal: Mapping[str, Any],
        feedback: str,
        context: dict[str, Any],
    ) -> Mapping[str, Any] | None:
        raw = dict(proposal)
        try:
            factor = _factor_from_raw(raw)
        except (TypeError, ValueError, KeyError):
            factor = FactorDefinition(
                name="repair_placeholder",
                family="research",
                hypothesis="Repair the invalid candidate contract.",
                expression=field("close"),
            )
        generated = GeneratedProposal(
            factor=factor,
            change=str(raw.get("change", "")),
            expected=str(raw.get("expected", "")),
            raw=raw,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            prompt_hash="",
            response_hash="",
        )
        repaired = _run_async(
            self.client.repair(
                generated,
                feedback,
                [{"kind": "mvp_context", "content": context}],
                int(context.get("round", 1)),
                data_context={"research_direction": context.get("research_direction", "")},
            )
        )
        return repaired.raw

    def conclude(self, summary: Mapping[str, Any]) -> str:
        outcome = _run_async(
            self.client.analyze(
                role="Researcher",
                system_prompt=(
                    "You are the single Researcher for a bounded factor-research MVP. "
                    "Return one JSON object with a short conclusion string. State only "
                    "what the deterministic candidate counts and public metrics support; "
                    "do not claim production readiness or hidden-test evidence."
                ),
                context=dict(summary),
                required_keys={"conclusion"},
            )
        )
        return str(outcome.artifact.get("conclusion", "")).strip()


def run_factor_research(
    research_direction: str,
    candidate_count: int = 4,
    rounds: int = 3,
    *,
    researcher: FactorResearcher | Any | None = None,
    evaluator: PriceVolumeEvaluator | Any | None = None,
    store: ServiceStore | Any | None = None,
    registry: FactorRegistry | None = None,
    output_dir: Path | None = None,
    maximum_repairs_per_candidate: int = 1,
    maximum_llm_calls: int | None = None,
    maximum_candidate_evaluations: int | None = None,
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    """Run the bounded Alpha Factor Discovery MVP and write two summaries.

    The public three-argument form resolves configured dependencies. Tests and
    local demos may inject deterministic researcher/evaluator/store objects.
    No result from this function is a production promotion or broker action.
    """

    direction = str(research_direction).strip()
    if not direction:
        raise ValueError("research_direction cannot be empty")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    budget = FactorResearchBudget(
        candidate_count=candidate_count,
        rounds=rounds,
        maximum_repairs_per_candidate=maximum_repairs_per_candidate,
        maximum_llm_calls=maximum_llm_calls,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
    )
    run_id = _run_id()
    runtime_root = _runtime_root()
    resolved_store = store or ServiceStore(runtime_root / "autoalpha.sqlite3")
    resolved_evaluator = evaluator or _default_evaluator()
    resolved_researcher = researcher or _default_researcher(resolved_store)
    resolved_registry = registry or FactorRegistry(runtime_root / "factor-registry")
    artifact_dir = Path(output_dir or runtime_root / "factor-research" / run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_budget = _RunBudget()
    candidates: list[_CandidateOutcome] = []
    kept_factors: list[FactorDefinition] = []
    seen_hashes = _existing_expression_hashes(resolved_store)
    existing_factors = _existing_factors(resolved_store)
    generation_errors: list[str] = []
    ordinal = 0
    candidate_generation_complete = False

    for round_number in range(1, budget.rounds + 1):
        if len(candidates) >= budget.candidate_count:
            candidate_generation_complete = True
            break
        if run_budget.llm_calls >= int(budget.maximum_llm_calls or 0):
            run_budget.exhausted.append("maximum_llm_calls")
            break
        remaining = budget.candidate_count - len(candidates)
        context = _research_context(
            direction,
            round_number,
            existing_factors,
            candidates,
            resolved_evaluator,
        )
        try:
            run_budget.llm_calls += 1
            raw_batch = _propose_batch(
                resolved_researcher,
                direction,
                remaining,
                round_number,
                context,
            )
        except Exception as error:  # isolate a failed generation round
            generation_errors.append(f"round_{round_number}:{type(error).__name__}")
            continue
        if not raw_batch:
            generation_errors.append(f"round_{round_number}:empty_proposal_batch")
            continue
        for raw in raw_batch[:remaining]:
            ordinal += 1
            outcome = _process_candidate(
                raw,
                ordinal=ordinal,
                round_number=round_number,
                run_id=run_id,
                direction=direction,
                researcher=resolved_researcher,
                evaluator=resolved_evaluator,
                store=resolved_store,
                registry=resolved_registry,
                source_task_id=f"factor-research:{run_id}",
                seen_hashes=seen_hashes,
                existing_factors=[*existing_factors, *kept_factors],
                budget=budget,
                run_budget=run_budget,
                correlation_threshold=correlation_threshold,
                context=context,
            )
            candidates.append(outcome)
            if outcome.status == KEEP and outcome.factor is not None:
                kept_factors.append(outcome.factor)
            if len(candidates) >= budget.candidate_count:
                break
        if len(candidates) >= budget.candidate_count:
            candidate_generation_complete = True
            break

    if len(candidates) >= budget.candidate_count:
        candidate_generation_complete = True
    if not candidate_generation_complete and len(candidates) < budget.candidate_count:
        run_budget.exhausted.append("candidate_generation_rounds")

    portfolio = _evaluate_simple_portfolio(resolved_evaluator, kept_factors)
    if portfolio.get("status") == "FAILED":
        generation_errors.append("simple_portfolio:evaluation_failed")

    counts = Counter(outcome.status for outcome in candidates)
    rejected_categories = {
        status: counts[status] for status in sorted(counts) if status in _REJECTED_STATUSES
    }
    top_factors = [
        outcome.to_dict()
        for outcome in sorted(
            (item for item in candidates if item.status == KEEP),
            key=_candidate_rank_key,
            reverse=True,
        )[:5]
    ]
    summary = {
        "run_id": run_id,
        "status": (
            "COMPLETED"
            if candidate_generation_complete and not run_budget.exhausted and not generation_errors
            else "PARTIAL_COMPLETED"
        ),
        "research_direction": direction,
        "budgets": budget.to_dict(),
        "usage": {
            "llm_calls": run_budget.llm_calls,
            "candidate_evaluations": run_budget.candidate_evaluations,
            "repairs": sum(run_budget.repair_counts.values()),
            "exhausted": sorted(set(run_budget.exhausted)),
        },
        "counts": {
            "requested": budget.candidate_count,
            "proposed": len(candidates),
            "keep": counts[KEEP],
            "rejected": sum(counts[status] for status in _REJECTED_STATUSES),
            "by_status": dict(sorted(counts.items())),
        },
        "rejected_categories": rejected_categories,
        "candidates": [outcome.to_dict() for outcome in candidates],
        "top_factors": top_factors,
        "simple_portfolio": portfolio,
        "generation_errors": generation_errors,
        "production_promotion": {
            "allowed": False,
            "reason": "MVP research admission is separate from production promotion",
        },
    }
    summary["llm_conclusion"] = _llm_conclusion(
        resolved_researcher,
        summary,
        run_budget=run_budget,
        maximum_llm_calls=int(budget.maximum_llm_calls or 0),
    )
    summary["usage"]["llm_calls"] = run_budget.llm_calls
    summary["usage"]["repairs"] = sum(run_budget.repair_counts.values())
    summary["usage"]["exhausted"] = sorted(set(run_budget.exhausted))
    if run_budget.exhausted or generation_errors:
        summary["status"] = "PARTIAL_COMPLETED"
    summary["artifacts"] = {
        "json": str(artifact_dir / "research_summary.json"),
        "markdown": str(artifact_dir / "research_summary.md"),
    }
    safe_summary = _json_safe(summary)
    (artifact_dir / "research_summary.json").write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "research_summary.md").write_text(
        _markdown_summary(safe_summary),
        encoding="utf-8",
    )
    return safe_summary


def _process_candidate(
    raw: Any,
    *,
    ordinal: int,
    round_number: int,
    run_id: str,
    direction: str,
    researcher: Any,
    evaluator: Any,
    store: Any,
    registry: FactorRegistry,
    source_task_id: str,
    seen_hashes: set[str],
    existing_factors: list[FactorDefinition],
    budget: FactorResearchBudget,
    run_budget: _RunBudget,
    correlation_threshold: float,
    context: dict[str, Any],
) -> _CandidateOutcome:
    proposal = _safe_proposal_dict(raw)
    candidate_key = f"{run_id}-candidate-{ordinal:02d}"
    _begin_iteration(store, run_id, ordinal)
    repair_count = 0
    parent_factor_id: str | None = None
    while True:
        try:
            factor = _factor_from_raw(proposal)
            candidate_key = factor.factor_id
        except (KeyError, TypeError, ValueError) as error:
            outcome = _CandidateOutcome(
                ordinal,
                round_number,
                INVALID,
                _reason_code(error, INVALID),
                proposal,
                candidate_key,
                repair_count=repair_count,
                parent_factor_id=parent_factor_id,
            )
            repaired = _try_repair(
                outcome,
                researcher=researcher,
                direction=direction,
                context=context,
                budget=budget,
                run_budget=run_budget,
            )
            if repaired is None:
                return _finish_candidate(store, run_id, ordinal, outcome)
            if outcome.factor is not None and parent_factor_id is None:
                parent_factor_id = outcome.factor.factor_id
            repair_count = outcome.repair_count
            proposal = _safe_proposal_dict(repaired)
            continue

        normalized_proposal = _normalized_proposal(proposal, factor)
        outcome = _CandidateOutcome(
            ordinal,
            round_number,
            KEEP,
            "candidate_not_processed",
            normalized_proposal,
            candidate_key,
            factor=factor,
            repair_count=repair_count,
            parent_factor_id=parent_factor_id,
        )
        if factor.expression.expression_hash in seen_hashes:
            outcome.status = DUPLICATE
            outcome.reason = "canonical_expression_duplicate"
            outcome.duplicate_of = factor.factor_id
            return _finish_candidate(store, run_id, ordinal, outcome)
        seen_hashes.add(factor.expression.expression_hash)

        try:
            semantic_result = _validate_expression(evaluator, factor)
        except Exception as error:
            outcome.status = INVALID
            outcome.reason = _reason_code(error, INVALID)
            repaired = _try_repair(
                outcome,
                researcher=researcher,
                direction=direction,
                context=context,
                budget=budget,
                run_budget=run_budget,
            )
            if repaired is None:
                return _finish_candidate(store, run_id, ordinal, outcome)
            if outcome.factor is not None and parent_factor_id is None:
                parent_factor_id = outcome.factor.factor_id
            repair_count = outcome.repair_count
            proposal = _safe_proposal_dict(repaired)
            continue

        if run_budget.candidate_evaluations >= int(budget.maximum_candidate_evaluations or 0):
            run_budget.exhausted.append("maximum_candidate_evaluations")
            outcome.status = BUDGET_EXHAUSTED
            outcome.reason = "maximum_candidate_evaluations"
            return _finish_candidate(store, run_id, ordinal, outcome)

        run_budget.candidate_evaluations += 1
        try:
            train_metrics, validation_metrics = _evaluate_stages(evaluator, factor)
        except _TrainEvaluationError as error:
            outcome.status = TRAIN_FAILED
            outcome.reason = error.code
            repaired = _try_repair(
                outcome,
                researcher=researcher,
                direction=direction,
                context=context,
                budget=budget,
                run_budget=run_budget,
            )
            if repaired is None:
                return _finish_candidate(store, run_id, ordinal, outcome)
            if outcome.factor is not None and parent_factor_id is None:
                parent_factor_id = outcome.factor.factor_id
            repair_count = outcome.repair_count
            proposal = _safe_proposal_dict(repaired)
            continue
        except Exception as error:
            outcome.status = VALIDATION_FAILED
            outcome.reason = _reason_code(error, VALIDATION_FAILED)
            repaired = _try_repair(
                outcome,
                researcher=researcher,
                direction=direction,
                context=context,
                budget=budget,
                run_budget=run_budget,
            )
            if repaired is None:
                return _finish_candidate(store, run_id, ordinal, outcome)
            if outcome.factor is not None and parent_factor_id is None:
                parent_factor_id = outcome.factor.factor_id
            repair_count = outcome.repair_count
            proposal = _safe_proposal_dict(repaired)
            continue

        outcome.train_metrics = _json_safe(train_metrics)
        outcome.validation_metrics = _json_safe(validation_metrics)
        outcome.core_metrics = _core_validation_metrics(train_metrics, validation_metrics)
        passed, reason = _validation_passed(outcome.core_metrics)
        if not passed:
            outcome.status = VALIDATION_FAILED
            outcome.reason = reason
            repaired = _try_repair(
                outcome,
                researcher=researcher,
                direction=direction,
                context=context,
                budget=budget,
                run_budget=run_budget,
            )
            if repaired is None:
                return _finish_candidate(store, run_id, ordinal, outcome)
            if outcome.factor is not None and parent_factor_id is None:
                parent_factor_id = outcome.factor.factor_id
            repair_count = outcome.repair_count
            proposal = _safe_proposal_dict(repaired)
            continue

        behavior_duplicate = _behavior_duplicate(
            evaluator,
            factor,
            existing_factors,
            threshold=correlation_threshold,
        )
        if behavior_duplicate is not None:
            outcome.status = DUPLICATE
            outcome.reason = "behavior_correlation_duplicate"
            outcome.duplicate_of = behavior_duplicate
            return _finish_candidate(store, run_id, ordinal, outcome)

        outcome.status = KEEP
        outcome.reason = "validation_core_metrics_passed"
        if semantic_result is not None:
            outcome.core_metrics["semantic_lookback"] = getattr(semantic_result, "lookback", None)
        _persist_keep(
            outcome,
            store=store,
            registry=registry,
            source_iteration=ordinal,
            source_task_id=source_task_id,
            run_id=run_id,
            direction=direction,
        )
        return _finish_candidate(store, run_id, ordinal, outcome)


def _try_repair(
    outcome: _CandidateOutcome,
    *,
    researcher: Any,
    direction: str,
    context: dict[str, Any],
    budget: FactorResearchBudget,
    run_budget: _RunBudget,
) -> Mapping[str, Any] | None:
    if outcome.status == DUPLICATE or outcome.repair_count >= budget.maximum_repairs_per_candidate:
        return None
    if run_budget.llm_calls >= int(budget.maximum_llm_calls or 0):
        run_budget.exhausted.append("maximum_llm_calls")
        return None
    repair = getattr(researcher, "repair", None)
    if not callable(repair):
        return None
    run_budget.llm_calls += 1
    run_budget.repair_counts[outcome.candidate_id] = outcome.repair_count + 1
    outcome.repair_count += 1
    feedback = outcome.status
    try:
        repaired = repair(
            outcome.proposal,
            feedback,
            {**context, "research_direction": direction, "feedback": feedback},
        )
    except Exception:
        return None
    if not isinstance(repaired, Mapping):
        return None
    return repaired


def _persist_keep(
    outcome: _CandidateOutcome,
    *,
    store: Any,
    registry: FactorRegistry,
    source_iteration: int,
    source_task_id: str,
    run_id: str,
    direction: str,
) -> None:
    assert outcome.factor is not None
    metrics = {
        "research_run_id": run_id,
        "research_direction": direction,
        "research_status": KEEP,
        "research_train_metrics": outcome.train_metrics,
        "research_validation_metrics": outcome.validation_metrics,
        "research_core_metrics": outcome.core_metrics,
        "expression_hash": outcome.factor.expression.expression_hash,
        "fields": sorted(_expression_fields(outcome.factor.expression)),
        "parent_factor_id": outcome.parent_factor_id,
        "admission": "MVP_RESEARCH_ONLY",
    }
    store.upsert_factor_pool(
        factor_id=outcome.factor.factor_id,
        source_iteration=source_iteration,
        source_task_id=source_task_id,
        proposal=outcome.proposal,
        metrics=_json_safe(metrics),
        status=KEEP,
        status_reason=outcome.reason,
    )
    store.upsert_factor_knowledge(
        factor_id=outcome.factor.factor_id,
        canonical_mechanism=outcome.factor.family,
        mechanism_summary=outcome.factor.hypothesis,
        tags=["mvp_factor_research", canonical_family(direction)],
        review={"status": KEEP, "metrics": outcome.core_metrics},
        falsification={"feedback_policy": "categorical_only", "research_run_id": run_id},
        related_factors=[],
    )
    try:
        card = registry.publish(
            outcome.factor,
            data_dependencies=tuple(_expression_fields(outcome.factor.expression)),
            data_lag_days=int(outcome.core_metrics.get("semantic_lookback") or 0),
            applicable_regimes=("public_validation",),
            failure_modes=(),
            owner="mvp_factor_research",
            experiment_id=run_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        outcome.reason += "; registry_archive_failed"
        outcome.core_metrics["registry_error"] = type(error).__name__
    else:
        outcome.registry_artifact_hash = card.artifact_hash


def _finish_candidate(
    store: Any, run_id: str, ordinal: int, outcome: _CandidateOutcome
) -> _CandidateOutcome:
    metrics = {
        "research_status": outcome.status,
        "research_reason": outcome.reason,
        "repair_count": outcome.repair_count,
        "duplicate_of": outcome.duplicate_of,
        "train_metrics": outcome.train_metrics,
        "validation_metrics": outcome.validation_metrics,
        "core_metrics": outcome.core_metrics,
        "parent_factor_id": outcome.parent_factor_id,
    }
    if outcome.status != KEEP:
        _persist_rejected_history(store, run_id, ordinal, outcome, metrics)
    finish = getattr(store, "finish_iteration", None)
    if callable(finish):
        finish(
            run_id,
            ordinal,
            status="COMPLETED",
            candidate_id=outcome.candidate_id,
            proposal=outcome.proposal,
            metrics=_json_safe(metrics),
            decision=outcome.status,
            error=None if outcome.status not in _REJECTED_STATUSES else outcome.reason,
        )
    remember = getattr(store, "remember", None)
    if callable(remember):
        remember(
            run_id, ordinal, "factor_research", json.dumps(_json_safe(metrics), sort_keys=True)
        )
    return outcome


def _persist_rejected_history(
    store: Any,
    run_id: str,
    ordinal: int,
    outcome: _CandidateOutcome,
    metrics: dict[str, Any],
) -> None:
    """Keep a small rejected-candidate record without polluting KEEP factors."""

    upsert = getattr(store, "upsert_factor_pool", None)
    if not callable(upsert):
        return
    candidate_hash = hashlib.sha256(
        json.dumps(
            _json_safe(outcome.proposal),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    rejected_id = f"R_{run_id}_{ordinal:02d}"
    rejected_metrics = {
        **metrics,
        "candidate_id": outcome.candidate_id,
        "candidate_hash": candidate_hash,
        "admission": "MVP_RESEARCH_REJECTED_HISTORY",
    }
    try:
        upsert(
            factor_id=rejected_id,
            source_iteration=ordinal,
            source_task_id=f"factor-research:{run_id}",
            proposal=outcome.proposal,
            metrics=_json_safe(rejected_metrics),
            status=REJECTED,
            status_reason=f"{outcome.status}:{outcome.reason}",
        )
    except Exception as error:  # rejected history must not destroy the run
        outcome.reason += f"; rejected_history_persist_failed:{type(error).__name__}"


def _begin_iteration(store: Any, run_id: str, ordinal: int) -> None:
    begin = getattr(store, "begin_iteration", None)
    if callable(begin):
        begin(run_id, ordinal)


def _validate_expression(evaluator: Any, factor: FactorDefinition) -> Any:
    validator = getattr(evaluator, "validator", None)
    if validator is not None and callable(getattr(validator, "validate", None)):
        return validator.validate(factor.expression)
    validate = getattr(evaluator, "validate_expression", None)
    if callable(validate):
        return validate(factor.expression)
    return None


class _TrainEvaluationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _evaluate_stages(
    evaluator: Any, factor: FactorDefinition
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight = getattr(evaluator, "preflight", None)
    if callable(preflight):
        result = preflight(factor)
        passed = getattr(result, "passed", None)
        if passed is False or (isinstance(result, Mapping) and result.get("passed") is False):
            raise _TrainEvaluationError("signal_preflight_failed")

    train_method = getattr(evaluator, "evaluate_train", None)
    validation_method = getattr(evaluator, "evaluate_validation", None)
    if callable(train_method) and callable(validation_method):
        train = _result_metrics(train_method(factor))
        validation = _result_metrics(validation_method(factor))
        return train, validation

    evaluate = getattr(evaluator, "evaluate", None)
    if not callable(evaluate):
        raise _TrainEvaluationError("evaluator_missing")
    try:
        metrics = _result_metrics(evaluate(factor))
    except _TrainEvaluationError:
        raise
    except Exception as error:
        raise RuntimeError("validation_evaluation_failed") from error
    train = metrics.get("train_metrics") or metrics.get("exploration_metrics") or {}
    validation = metrics.get("validation_metrics") or metrics
    if not isinstance(train, Mapping):
        train = {}
    if not isinstance(validation, Mapping):
        validation = metrics
    if not train:
        train = {
            key: metrics[key]
            for key in ("sharpe_ratio", "simple_annual_return", "max_drawdown")
            if key in metrics
        }
    return dict(train), dict(validation)


def _result_metrics(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, Mapping):
        return dict(metrics)
    raise TypeError("evaluator result must expose a metrics mapping")


def _core_validation_metrics(
    train: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    train_sharpe = _metric(train, "sharpe", "sharpe_ratio", "long_only_sharpe_ratio")
    validation_sharpe = _metric(
        validation,
        "sharpe",
        "sharpe_ratio",
        "long_only_sharpe_ratio",
        "portfolio_sharpe_ratio",
    )
    train_return = _metric(
        train, "simple_annual_return", "annual_return", "long_only_simple_annual_return"
    )
    validation_return = _metric(
        validation,
        "simple_annual_return",
        "annual_return",
        "long_only_simple_annual_return",
        "portfolio_simple_annual_return",
    )
    train_sign = _sign(train_sharpe if train_sharpe != 0 else train_return)
    validation_sign = _sign(validation_sharpe if validation_sharpe != 0 else validation_return)
    return {
        "coverage": _metric(validation, "coverage", "long_only_coverage", "portfolio_coverage"),
        "rank_ic": _metric(
            validation, "rank_ic_mean", "rank_ic", "long_only_active_information_ratio"
        ),
        "sharpe": validation_sharpe,
        "annual_return": validation_return,
        "max_drawdown": _metric(
            validation,
            "max_drawdown",
            "long_only_max_drawdown",
            "portfolio_max_drawdown",
        ),
        "turnover": _metric(
            validation,
            "annual_turnover",
            "long_only_annual_turnover",
            "portfolio_annual_turnover",
        ),
        "train_sharpe": train_sharpe,
        "validation_sharpe": validation_sharpe,
        "train_annual_return": train_return,
        "validation_annual_return": validation_return,
        "sign_consistency": bool(train_sign != 0 and train_sign == validation_sign),
        "explicit_validation_passed": validation.get("passed", validation.get("validation_passed")),
        "bankrupt": bool(validation.get("long_only_bankrupt", validation.get("bankrupt", False))),
        "observations": int(
            validation.get(
                "backtest_observations",
                validation.get(
                    "long_only_backtest_observations", validation.get("observations", 0)
                ),
            )
            or 0
        ),
    }


def _validation_passed(metrics: Mapping[str, Any]) -> tuple[bool, str]:
    explicit = metrics.get("explicit_validation_passed")
    if explicit is False:
        return False, "validation_gate_failed"
    numeric_names = ("coverage", "sharpe", "annual_return", "max_drawdown", "turnover")
    if any(not math.isfinite(float(metrics.get(name, 0.0))) for name in numeric_names):
        return False, "non_finite_core_metrics"
    if metrics.get("observations", 0) < 1:
        return False, "no_validation_observations"
    if float(metrics.get("coverage", 0.0)) < 0.80:
        return False, "coverage_below_mvp_threshold"
    if float(metrics.get("sharpe", 0.0)) <= 0.0 or float(metrics.get("annual_return", 0.0)) <= 0.0:
        return False, "non_positive_validation_return"
    if not bool(metrics.get("sign_consistency")):
        return False, "train_validation_sign_inconsistent"
    if bool(metrics.get("bankrupt")):
        return False, "validation_capital_failure"
    return True, "validation_core_metrics_passed"


def _behavior_duplicate(
    evaluator: Any,
    factor: FactorDefinition,
    references: list[FactorDefinition],
    *,
    threshold: float,
) -> str | None:
    references = [item for item in references if item.factor_id != factor.factor_id]
    if not references:
        return None
    method = getattr(evaluator, "library_signal_correlation", None)
    if callable(method):
        try:
            diagnostic = method(factor, references, sample_stride=5, max_references=20)
        except Exception:
            return None
        value = float(diagnostic.get("library_signal_correlation_max", 0.0))
        return (
            str(diagnostic.get("library_signal_correlation_peer")) if value >= threshold else None
        )
    method = getattr(evaluator, "signal_correlation", None)
    if callable(method):
        for reference in references:
            try:
                value = abs(float(method(factor, reference)))
            except Exception:
                continue
            if value >= threshold:
                return reference.factor_id
    return None


def _evaluate_simple_portfolio(evaluator: Any, factors: list[FactorDefinition]) -> dict[str, Any]:
    if not factors:
        return {"status": "NOT_EVALUATED", "reason": "no_KEEP_factors"}
    selected = factors[:5]
    evaluate = getattr(evaluator, "evaluate_portfolio", None)
    if not callable(evaluate):
        return {"status": "FAILED", "reason": "evaluator_missing_portfolio_method"}
    try:
        try:
            result = evaluate(selected, weights=[1.0] * len(selected), bootstrap_samples=0)
        except TypeError:
            result = evaluate(selected, weights=[1.0] * len(selected))
        metrics = _result_metrics(result)
        correlations = getattr(result, "factor_correlations", {})
        return _json_safe(
            {
                "status": "EVALUATED",
                "method": "equal_weight_top_5",
                "factor_ids": [factor.factor_id for factor in selected],
                "weights": {factor.factor_id: 1.0 / len(selected) for factor in selected},
                "metrics": metrics,
                "factor_correlations": correlations,
            }
        )
    except Exception as error:
        return {"status": "FAILED", "reason": type(error).__name__}


def _existing_factors(store: Any) -> list[FactorDefinition]:
    records = getattr(store, "factor_pool", lambda **_: [])(limit=5000)
    result: list[FactorDefinition] = []
    for record in records:
        if str(record.get("status", "")) in {
            REJECTED,
            INVALID,
            DUPLICATE,
            TRAIN_FAILED,
            VALIDATION_FAILED,
            BUDGET_EXHAUSTED,
        }:
            continue
        try:
            result.append(_factor_from_raw(record["proposal"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _existing_expression_hashes(store: Any) -> set[str]:
    hashes: set[str] = set()
    records = getattr(store, "factor_pool", lambda **_: [])(limit=5000)
    for record in records:
        try:
            hashes.add(_factor_from_raw(record["proposal"]).expression.expression_hash)
        except (KeyError, TypeError, ValueError):
            factor_id = record.get("factor_id")
            if factor_id:
                hashes.add(str(factor_id))
    return hashes


def _research_context(
    direction: str,
    round_number: int,
    existing_factors: list[FactorDefinition],
    candidates: list[_CandidateOutcome],
    evaluator: Any,
) -> dict[str, Any]:
    fields = getattr(evaluator, "factor_fields", None)
    return {
        "research_direction": direction,
        "round": round_number,
        "available_factor_fields": sorted(fields)
        if fields
        else ["close", "adj_close", "amount", "vol"],
        "existing_factors": [
            {
                "factor_id": factor.factor_id,
                "name": factor.name,
                "family": factor.family,
                "expression": factor.expression.to_dict(),
            }
            for factor in existing_factors[-40:]
        ],
        "rejected_categories": [
            item.status for item in candidates if item.status in _REJECTED_STATUSES
        ],
    }


def _propose_batch(
    researcher: Any,
    direction: str,
    remaining: int,
    round_number: int,
    context: dict[str, Any],
) -> list[Mapping[str, Any]]:
    method = getattr(researcher, "propose_batch", None)
    if callable(method):
        value = method(direction, min(remaining, 5), round_number, context)
        if isinstance(value, Mapping):
            return [value]
        return [item for item in value if isinstance(item, Mapping)]
    method = getattr(researcher, "propose", None)
    if not callable(method):
        raise TypeError("researcher must implement propose_batch or propose")
    value = method(direction, round_number, context)
    return [value] if isinstance(value, Mapping) else []


def _proposal_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("proposal must be a JSON object")
    return _json_safe(dict(raw))


def _safe_proposal_dict(raw: Any) -> dict[str, Any]:
    try:
        return _proposal_dict(raw)
    except (TypeError, ValueError):
        return {"invalid_proposal": str(raw)[:500]}


def _factor_from_raw(raw: Mapping[str, Any]) -> FactorDefinition:
    expression = Expression.from_dict(raw["expression"])
    return FactorDefinition(
        name=str(raw.get("name", "")).strip(),
        family=canonical_family(str(raw.get("family", "research"))),
        hypothesis=str(raw.get("hypothesis", "")).strip(),
        expression=expression,
        expected_direction=_expected_direction(raw.get("expected_direction", 1)),
    )


def _normalized_proposal(raw: Mapping[str, Any], factor: FactorDefinition) -> dict[str, Any]:
    proposal = dict(raw)
    proposal.update(
        {
            "name": factor.name,
            "family": factor.family,
            "hypothesis": factor.hypothesis,
            "expected_direction": factor.expected_direction,
            "expression": factor.expression.to_dict(),
            "expression_hash": factor.expression.expression_hash,
            "fields": sorted(_expression_fields(factor.expression)),
        }
    )
    return _json_safe(proposal)


def _expected_direction(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
    if value in {1, "+1", "1", "positive", "long", "up", "higher", "bullish"}:
        return 1
    if value in {-1, -1.0, "-1", "negative", "short", "down", "lower", "bearish"}:
        return -1
    raise ValueError("expected_direction must be 1 or -1")


def _metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
    return 0.0


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _candidate_rank_key(outcome: _CandidateOutcome) -> tuple[float, float, float]:
    metrics = outcome.core_metrics
    return (
        float(metrics.get("sharpe", -math.inf)),
        float(metrics.get("annual_return", -math.inf)),
        float(metrics.get("coverage", 0.0)),
    )


def _reason_code(error: Exception, fallback: str) -> str:
    message = str(error).lower()
    if (
        "unknown" in message
        or "requires" in message
        or "expression" in message
        or "field" in message
    ):
        return "dsl_contract_or_semantics_failed"
    return fallback.lower()


def _expression_fields(expression: Expression) -> set[str]:
    fields: set[str] = set()
    if expression.operator == "field":
        fields.add(str(expression.parameter("name")))
    for argument in expression.arguments:
        fields.update(_expression_fields(argument))
    return fields


def _run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"factor-research-{timestamp}-{uuid.uuid4().hex[:8]}"


def _runtime_root() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return Path(os.getenv("AUTOALPHA_RUNTIME", project_root / "runtime-full-llm"))


def _default_evaluator() -> PriceVolumeEvaluator:
    project_root = Path(__file__).resolve().parents[3]
    data_path = Path(os.getenv("AUTOALPHA_DATA_PATH", project_root / "data"))
    config_path = Path(os.getenv("AUTOALPHA_CONFIG", project_root / "config/research.toml"))
    return PriceVolumeEvaluator(data_path, config_path)


def _default_researcher(store: Any) -> CompatibleChatResearcher:
    settings = store.settings() if callable(getattr(store, "settings", None)) else {}
    base_url = settings.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = settings.get("model") or os.getenv("OPENAI_MODEL", "gpt-5")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AUTOALPHA_API_KEY")
    if not api_key:
        raise RuntimeError("LLM Researcher is not configured: set OPENAI_API_KEY")
    return CompatibleChatResearcher(
        CompatibleChatClient(base_url=base_url, api_key=api_key, model=model)
    )


def _llm_conclusion(
    researcher: Any,
    summary: Mapping[str, Any],
    *,
    run_budget: _RunBudget,
    maximum_llm_calls: int,
) -> str:
    conclude = getattr(researcher, "conclude", None)
    if not callable(conclude) or run_budget.llm_calls >= maximum_llm_calls:
        if run_budget.llm_calls >= maximum_llm_calls:
            run_budget.exhausted.append("maximum_llm_calls")
        return "本轮结论由确定性门禁和公开评价证据构成，未作生产晋级判断。"
    try:
        run_budget.llm_calls += 1
        result = conclude(
            {
                "research_direction": summary["research_direction"],
                "counts": summary["counts"],
                "top_factors": summary["top_factors"],
                "simple_portfolio": summary["simple_portfolio"],
            }
        )
    except Exception:
        return "本轮结论由确定性门禁和公开评价证据构成，未作生产晋级判断。"
    return str(result).strip() or "本轮没有可供总结的 LLM 结论。"


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# 因子研究摘要",
        "",
        f"- 运行状态：`{summary['status']}`",
        f"- 研究方向：{summary['research_direction']}",
        (
            f"- 候选：{counts['proposed']} / {counts['requested']}；"
            f"KEEP：{counts['keep']}；拒绝：{counts['rejected']}"
        ),
        (
            f"- LLM 调用：{summary['usage']['llm_calls']}；"
            f"候选评价：{summary['usage']['candidate_evaluations']}；"
            f"修复：{summary['usage']['repairs']}"
        ),
        "",
        "## 候选结果",
        "",
        "| 候选 | 状态 | 验证 Sharpe | 验证年化 | 覆盖率 | 原因 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for candidate in summary["candidates"]:
        metrics = candidate.get("core_metrics", {})
        lines.append(
            (
                "| {name} | `{status}` | {sharpe:.3f} | {annual:.2%} | {coverage:.2%} | {reason} |"
            ).format(
                name=str(candidate.get("proposal", {}).get("name", candidate["candidate_id"])),
                status=candidate["status"],
                sharpe=float(metrics.get("sharpe", 0.0)),
                annual=float(metrics.get("annual_return", 0.0)),
                coverage=float(metrics.get("coverage", 0.0)),
                reason=candidate["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## 简单组合",
            "",
            "```json",
            json.dumps(summary["simple_portfolio"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    lines.extend(["## 研究者结论", "", str(summary["llm_conclusion"]), ""])
    lines.extend(
        [
            "## 约束",
            "",
            "本摘要只记录公开研究证据；MVP 不触发隐藏测试、生产晋级、券商连接或实盘下单。",
            "",
        ]
    )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("CompatibleChatResearcher must run from a synchronous context")


__all__ = [
    "BUDGET_EXHAUSTED",
    "CompatibleChatResearcher",
    "DUPLICATE",
    "FactorResearchBudget",
    "FactorResearcher",
    "INVALID",
    "KEEP",
    "TRAIN_FAILED",
    "VALIDATION_FAILED",
    "run_factor_research",
]
