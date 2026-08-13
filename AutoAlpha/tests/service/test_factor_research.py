from __future__ import annotations

import json
from types import SimpleNamespace

from autoalpha.dsl.expression import Expression, field
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.registry.store import FactorRegistry
from autoalpha.service.factor_research import (
    DUPLICATE,
    INVALID,
    KEEP,
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

    def propose_batch(
        self,
        research_direction: str,
        candidate_count: int,
        round_number: int,
        context: dict,
    ) -> list[dict]:
        return self.proposals[:candidate_count]

    def repair(self, proposal: dict, feedback: str, context: dict) -> None:
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
    summary = run_factor_research(
        "短期反转",
        candidate_count=4,
        rounds=3,
        researcher=FakeResearcher(proposals),
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
    assert len(pool) == 1
    assert pool[0]["status"] == KEEP
    factor_id = pool[0]["factor_id"]
    assert store.factor_knowledge(factor_id) is not None
    assert len(FactorRegistry(tmp_path / "factor-registry").versions(factor_id)) == 1

    json_path = tmp_path / "summary" / "research_summary.json"
    markdown_path = tmp_path / "summary" / "research_summary.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts"]["keep"] == 1
    assert "确定性 DSL" in markdown_path.read_text(encoding="utf-8")


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
