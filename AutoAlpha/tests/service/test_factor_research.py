from __future__ import annotations

import json
from types import SimpleNamespace

from autoalpha.dsl.expression import Expression, FactorDefinition, field
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.registry.store import FactorRegistry
from autoalpha.service.factor_research import (
    DUPLICATE,
    INVALID,
    KEEP,
    REJECTED,
    VALIDATION_FAILED,
    run_factor_research,
)
from autoalpha.service.store import ServiceStore


def _proposal(name: str, expression: dict, *, expected_direction: int = 1) -> dict:
    return {
        "name": name,
        "family": "test",
        "hypothesis": f"{name} has a falsifiable return hypothesis.",
        "expected_direction": expected_direction,
        "expression": expression,
    }


class FakeResearcher:
    def __init__(self, proposals: list[dict]) -> None:
        self.proposals = proposals
        self.repair_feedback: list[str] = []

    def propose_batch(
        self,
        research_direction: str,
        candidate_count: int,
        round_number: int,
        context: dict,
    ) -> list[dict]:
        return self.proposals[:candidate_count]

    def repair(self, proposal: dict, feedback: str, context: dict) -> None:
        self.repair_feedback.append(feedback)
        return None

    def conclude(self, summary: dict) -> str:
        return "四个候选经过确定性 DSL、公开评价和去重门禁。"


class FakeEvaluator:
    def __init__(self) -> None:
        self.validator = SemanticValidator(
            [
                FieldDefinition("close", "price"),
                FieldDefinition("amount", "cny"),
            ]
        )
        self.portfolio_calls: list[list[str]] = []

    def evaluate(self, factor):  # noqa: ANN001
        failed = factor.name == "D_validation_failed"
        return SimpleNamespace(
            metrics={
                "exploration_metrics": {
                    "observations": 100,
                    "sharpe": 0.40,
                    "simple_annual_return": 0.04,
                    "max_drawdown": -0.12,
                },
                "passed": not failed,
                "observations": 100,
                "coverage": 0.92,
                "rank_ic_mean": 0.03,
                "long_only_sharpe_ratio": -0.10 if failed else 1.10,
                "long_only_simple_annual_return": -0.02 if failed else 0.12,
                "long_only_max_drawdown": -0.20,
                "long_only_annual_turnover": 0.40,
            }
        )

    def evaluate_portfolio(self, factors, *, weights, bootstrap_samples=0):  # noqa: ANN001
        self.portfolio_calls.append([factor.factor_id for factor in factors])
        return SimpleNamespace(
            metrics={
                "portfolio_sharpe_ratio": 1.25,
                "portfolio_simple_annual_return": 0.14,
                "portfolio_max_drawdown": -0.18,
                "portfolio_annual_turnover": 0.45,
                "portfolio_coverage": 0.91,
            },
            factor_correlations={},
        )


def test_factor_research_mvp_keeps_only_valid_non_duplicate_candidates(tmp_path) -> None:
    expression_a = field("close").to_dict()
    proposals = [
        _proposal("A_keep", expression_a),
        _proposal("B_duplicate", expression_a),
        _proposal(
            "C_invalid_dsl",
            Expression.from_dict(
                {"operator": "future_operator", "arguments": [], "parameters": {}}
            ).to_dict(),
        ),
        _proposal("D_validation_failed", field("amount").to_dict()),
    ]
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    evaluator = FakeEvaluator()
    researcher = FakeResearcher(proposals)
    summary = run_factor_research(
        "短期反转",
        candidate_count=4,
        rounds=1,
        researcher=researcher,
        evaluator=evaluator,
        store=store,
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    assert summary["status"] == "COMPLETED"
    assert summary["counts"]["by_status"] == {
        DUPLICATE: 1,
        INVALID: 1,
        KEEP: 1,
        VALIDATION_FAILED: 1,
    }
    assert summary["rejected_categories"] == {
        DUPLICATE: 1,
        INVALID: 1,
        VALIDATION_FAILED: 1,
    }
    assert summary["simple_portfolio"]["status"] == "EVALUATED"
    assert len(evaluator.portfolio_calls) == 1

    pool = store.factor_pool()
    keep_pool = [item for item in pool if item["status"] == KEEP]
    rejected_pool = [item for item in pool if item["status"] == REJECTED]
    assert len(keep_pool) == 1
    assert len(rejected_pool) == 3
    assert all(item["metrics"]["candidate_hash"] for item in rejected_pool)
    assert keep_pool[0]["proposal"]["expression_hash"]
    assert keep_pool[0]["proposal"]["fields"] == ["close"]
    factor_id = keep_pool[0]["factor_id"]
    assert store.factor_knowledge(factor_id) is not None
    assert len(FactorRegistry(tmp_path / "factor-registry").versions(factor_id)) == 1

    json_path = tmp_path / "summary" / "research_summary.json"
    markdown_path = tmp_path / "summary" / "research_summary.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts"]["keep"] == 1
    assert "确定性 DSL" in markdown_path.read_text(encoding="utf-8")
    assert researcher.repair_feedback
    assert all(
        '"status"' in feedback and '"reason"' in feedback
        for feedback in researcher.repair_feedback
    )


def test_factor_research_mvp_generates_candidates_per_round(tmp_path) -> None:
    proposals = [
        _proposal(
            f"round_{index:02d}",
            Expression.from_dict(
                {
                    "operator": "rolling_mean",
                    "arguments": [field("close").to_dict()],
                    "parameters": {"window": index},
                }
            ).to_dict(),
        )
        for index in range(1, 13)
    ]

    class RoundResearcher(FakeResearcher):
        def __init__(self, values: list[dict]) -> None:
            super().__init__(values)
            self.rounds: list[int] = []

        def propose_batch(
            self,
            research_direction: str,
            candidate_count: int,
            round_number: int,
            context: dict,
        ) -> list[dict]:
            self.rounds.append(round_number)
            start = (round_number - 1) * candidate_count
            return self.proposals[start : start + candidate_count]

    researcher = RoundResearcher(proposals)
    summary = run_factor_research(
        "多轮候选",
        candidates_per_round=4,
        rounds=3,
        researcher=researcher,
        evaluator=FakeEvaluator(),
        store=ServiceStore(tmp_path / "autoalpha.sqlite3"),
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    assert researcher.rounds == [1, 2, 3]
    assert len(summary["candidates"]) == 12
    assert summary["counts"]["requested"] == 12
    assert summary["budgets"]["total_candidates"] == 12


def test_factor_research_mvp_repair_is_bounded(tmp_path) -> None:
    class RepairResearcher(FakeResearcher):
        def repair(self, proposal: dict, feedback: str, context: dict) -> dict:
            return {"name": "still_invalid", "hypothesis": "missing expression"}

    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    summary = run_factor_research(
        "测试修复预算",
        candidate_count=1,
        rounds=1,
        researcher=RepairResearcher([{"name": "invalid", "hypothesis": "x"}]),
        evaluator=FakeEvaluator(),
        store=store,
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    assert summary["status"] == "COMPLETED"
    assert summary["usage"]["repairs"] == 1
    assert summary["candidates"][0]["repair_count"] == 1
    assert summary["candidates"][0]["status"] == INVALID


def test_factor_research_mvp_persists_repaired_factor_parent(tmp_path) -> None:
    class RepairingResearcher(FakeResearcher):
        def repair(self, proposal: dict, feedback: str, context: dict) -> dict:
            return _proposal("repaired_keep", field("close").to_dict())

    summary = run_factor_research(
        "测试修复谱系",
        candidate_count=1,
        rounds=1,
        researcher=RepairingResearcher(
            [_proposal("D_validation_failed", field("amount").to_dict())]
        ),
        evaluator=FakeEvaluator(),
        store=ServiceStore(tmp_path / "autoalpha.sqlite3"),
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    candidate = summary["candidates"][0]
    assert candidate["status"] == KEEP
    assert candidate["repair_count"] == 1
    assert candidate["parent_factor_id"]
    assert candidate["proposal"]["expression_hash"]
    assert candidate["proposal"]["fields"] == ["close"]


def test_factor_research_mvp_marks_summary_partial_when_llm_budget_is_exhausted(
    tmp_path,
) -> None:
    summary = run_factor_research(
        "测试结论预算",
        candidate_count=1,
        rounds=1,
        researcher=FakeResearcher([_proposal("budget_keep", field("close").to_dict())]),
        evaluator=FakeEvaluator(),
        store=ServiceStore(tmp_path / "autoalpha.sqlite3"),
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
        maximum_llm_calls=1,
    )

    assert summary["status"] == "PARTIAL_COMPLETED"
    assert summary["usage"]["llm_calls"] == 1
    assert summary["usage"]["exhausted"] == ["maximum_llm_calls"]


def test_factor_research_mvp_ranks_simple_portfolio_by_validation_metrics(tmp_path) -> None:
    class RankingEvaluator(FakeEvaluator):
        def evaluate(self, factor):  # noqa: ANN001
            result = super().evaluate(factor)
            rank = int(factor.name.rsplit("_", 1)[-1])
            result.metrics["exploration_metrics"]["sharpe"] = rank / 10
            result.metrics["exploration_metrics"]["simple_annual_return"] = rank / 100
            result.metrics["long_only_sharpe_ratio"] = rank / 10
            result.metrics["long_only_simple_annual_return"] = rank / 100
            return result

    proposals = [
        _proposal(
            f"rank_{rank}",
            Expression.from_dict(
                {
                    "operator": "rolling_mean",
                    "arguments": [field("close").to_dict()],
                    "parameters": {"window": rank},
                }
            ).to_dict(),
        )
        for rank in range(1, 6)
    ]
    evaluator = RankingEvaluator()
    run_factor_research(
        "排序组合",
        candidate_count=5,
        rounds=1,
        researcher=FakeResearcher(proposals),
        evaluator=evaluator,
        store=ServiceStore(tmp_path / "autoalpha.sqlite3"),
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    expected_ids = [
        FactorDefinition(
            name=proposal["name"],
            family=proposal["family"],
            hypothesis=proposal["hypothesis"],
            expression=Expression.from_dict(proposal["expression"]),
        ).factor_id
        for proposal in reversed(proposals)
    ][:5]
    assert evaluator.portfolio_calls == [expected_ids]


def test_factor_research_mvp_keeps_non_finite_metrics_out_of_markdown(tmp_path) -> None:
    class NaNEvaluator(FakeEvaluator):
        def evaluate(self, factor):  # noqa: ANN001
            result = super().evaluate(factor)
            result.metrics["coverage"] = float("nan")
            return result

    summary = run_factor_research(
        "非有限指标",
        candidate_count=1,
        rounds=1,
        researcher=FakeResearcher([_proposal("nan_candidate", field("close").to_dict())]),
        evaluator=NaNEvaluator(),
        store=ServiceStore(tmp_path / "autoalpha.sqlite3"),
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    assert summary["candidates"][0]["reason"] == "non_finite_core_metrics"
    markdown = (tmp_path / "summary" / "research_summary.md").read_text(encoding="utf-8")
    assert "| nan_candidate | `VALIDATION_FAILED` | 1.100 | 12.00% | 0.00% |" in markdown


def test_factor_research_mvp_isolates_candidate_persistence_failure(tmp_path) -> None:
    class ExplodingStore(ServiceStore):
        def upsert_factor_pool(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("simulated_factor_pool_failure")

    summary = run_factor_research(
        "持久化隔离",
        candidate_count=1,
        rounds=1,
        researcher=FakeResearcher([_proposal("persist_failure", field("close").to_dict())]),
        evaluator=FakeEvaluator(),
        store=ExplodingStore(tmp_path / "autoalpha.sqlite3"),
        registry=FactorRegistry(tmp_path / "factor-registry"),
        output_dir=tmp_path / "summary",
    )

    assert summary["status"] == "PARTIAL_COMPLETED"
    assert summary["candidates"][0]["status"] == "TRAIN_FAILED"
    assert summary["generation_errors"] == ["candidate_1:RuntimeError"]
    assert (tmp_path / "summary" / "research_summary.json").exists()


def test_factor_research_mvp_rejected_history_does_not_block_a_later_run(tmp_path) -> None:
    proposal = _proposal(
        "retry_invalid",
        {"operator": "future_operator", "arguments": [], "parameters": {}},
    )
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    for run_number in (1, 2):
        summary = run_factor_research(
            f"重试拒绝候选 {run_number}",
            candidate_count=1,
            rounds=1,
            researcher=FakeResearcher([proposal]),
            evaluator=FakeEvaluator(),
            store=store,
            registry=FactorRegistry(tmp_path / f"factor-registry-{run_number}"),
            output_dir=tmp_path / f"summary-{run_number}",
        )
        assert summary["candidates"][0]["status"] == INVALID
