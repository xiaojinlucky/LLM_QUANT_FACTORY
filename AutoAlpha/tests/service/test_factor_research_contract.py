from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from autoalpha.config import ResearchConfig
from autoalpha.dsl.expression import field
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.service import app as service_app
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.factor_research import TRAIN_FAILED
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.research_protocol import default_task_protocol, protocol_fingerprint
from autoalpha.service.store import ServiceStore
from autoalpha.service.system_jobs import SystemJobRunner


def _proposal(name: str, expression: dict | None = None) -> dict:
    return {
        "name": name,
        "family": "test",
        "hypothesis": f"{name} has a falsifiable return hypothesis.",
        "expected_direction": 1,
        "expression": expression or field("close").to_dict(),
    }


class ContractResearcher:
    def __init__(self, proposals: list[dict]) -> None:
        self.proposals = proposals
        self.rounds: list[int] = []

    def propose_batch(
        self,
        research_direction: str,
        candidate_count: int,
        round_number: int,
        context: dict,
    ) -> list[dict]:
        self.rounds.append(round_number)
        return self.proposals[:candidate_count]

    def repair(self, proposal: dict, feedback: str, context: dict) -> None:
        return None

    def conclude(self, summary: dict) -> str:
        return "只根据确定性门禁和探索期、公开验证期证据总结。"


class ContractEvaluator:
    def __init__(self, *, data_path: str | None = None, fail_names: set[str] | None = None) -> None:
        self.data_path = data_path
        self.fail_names = fail_names or set()
        self.factor_fields = ("close", "amount")
        self.validator = SemanticValidator(
            [FieldDefinition("close", "price"), FieldDefinition("amount", "cny")]
        )

    def validate_expression(self, expression):  # noqa: ANN001
        return SimpleNamespace(lookback=1)

    def evaluate(self, factor):  # noqa: ANN001
        if factor.name in self.fail_names:
            raise RuntimeError("candidate evaluation failed")
        return SimpleNamespace(
            metrics={
                "train_metrics": {
                    "observations": 100,
                    "sharpe": 0.4,
                    "simple_annual_return": 0.04,
                },
                "validation_metrics": {
                    "observations": 100,
                    "coverage": 0.92,
                    "rank_ic_mean": 0.03,
                    "long_only_sharpe_ratio": 1.1,
                    "long_only_simple_annual_return": 0.12,
                    "long_only_max_drawdown": -0.2,
                    "long_only_annual_turnover": 0.4,
                },
            }
        )

    def evaluate_portfolio(self, factors, *, weights, bootstrap_samples=0):  # noqa: ANN001
        return SimpleNamespace(
            metrics={"portfolio_sharpe_ratio": 1.2},
            factor_correlations={},
        )


def _task(store: ServiceStore, task_id: str = "task-factor") -> dict:
    base = ResearchConfig.from_toml(Path("config/research.toml"))
    protocol = default_task_protocol("2010-01-04", "2026-07-16", base)
    return store.create_research_task(
        task_id=task_id,
        name="Factor Research Task",
        market="CN_A",
        data_path="C:/private/panel",
        data_start="2010-01-04",
        data_end="2026-07-16",
        snapshot_hash="snapshot-task",
        status="READY",
        protocol=protocol,
        protocol_hash=protocol_fingerprint(protocol),
    )


class FakeResearchTaskManager:
    def __init__(self, store: ServiceStore, evaluator: ContractEvaluator) -> None:
        self.store = store
        self.evaluator = evaluator
        self.calls: list[str] = []
        base = ResearchConfig.from_toml(Path("config/research.toml"))
        self.config = base

    def factor_research_context(self, task_id: str) -> dict:
        self.calls.append(task_id)
        task = self.store.research_task(task_id)
        assert task is not None
        protocol = task["protocol"]
        config = SimpleNamespace(
            splits=SimpleNamespace(
                train=SimpleNamespace(
                    start=date.fromisoformat(protocol["exploration_start"]),
                    end=date.fromisoformat(protocol["exploration_end"]),
                ),
                validation=SimpleNamespace(
                    start=date.fromisoformat(protocol["validation_start"]),
                    end=date.fromisoformat(protocol["validation_end"]),
                ),
            )
        )
        self.evaluator.data_path = task["data_path"]
        return {
            "task": task,
            "readiness": {
                "research_evidence_tier": "REGIME_SLICE_ONLY",
                "public_range": {
                    "start": protocol["exploration_start"],
                    "end": protocol["validation_end"],
                },
            },
            "config": config,
            "evaluator": self.evaluator,
        }


def _runner(
    store: ServiceStore,
    manager: FakeResearchTaskManager,
    runtime_root: Path,
    *,
    builder=None,  # noqa: ANN001
) -> SystemJobRunner:
    return SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=runtime_root,
        factor_library_builder=builder,
        research_task_manager=manager,
    )


def _enqueue_factor_job(
    store: ServiceStore,
    *,
    candidates_per_round: int = 1,
    rounds: int = 1,
    job_id: str = "job-factor-research",
) -> None:
    store.enqueue_system_job(
        job_id=job_id,
        queue="factor-research",
        job_type="factor_research",
        payload={
            "research_task_id": "task-factor",
            "research_direction": "短期反转",
            "candidates_per_round": candidates_per_round,
            "rounds": rounds,
        },
        progress_total=candidates_per_round * rounds,
        max_attempts=1,
    )


def test_factor_research_system_job_completes_and_uses_task_context(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    evaluator = ContractEvaluator()
    manager = FakeResearchTaskManager(store, evaluator)
    researcher = ContractResearcher([_proposal("keep")])
    real_run = __import__(
        "autoalpha.service.system_jobs", fromlist=["run_factor_research"]
    ).run_factor_research

    def run_with_researcher(direction: str, **kwargs):  # noqa: ANN001
        return real_run(direction, researcher=researcher, **kwargs)

    monkeypatch.setattr(
        "autoalpha.service.system_jobs.run_factor_research", run_with_researcher
    )
    _enqueue_factor_job(store)

    runner = _runner(store, manager, tmp_path / "runtime")
    result = runner.run_next(queue="factor-research")

    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert manager.calls == ["task-factor"]
    assert evaluator.data_path == "C:/private/panel"
    assert result["job"]["result"]["research_task_id"] == "task-factor"
    assert result["job"]["result"]["production_promotion"]["allowed"] is False
    assert result["job"]["result"]["artifacts"] == [
        {
            "artifact_id": (
                f"factor-research/{result['job']['result']['run_id']}/research_summary.json"
            ),
            "name": "research_summary.json",
            "media_type": "application/json",
        },
        {
            "artifact_id": (
                f"factor-research/{result['job']['result']['run_id']}/research_summary.md"
            ),
            "name": "research_summary.md",
            "media_type": "text/markdown",
        },
    ]


def test_factor_research_public_context_excludes_holdout_from_get_and_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    task = _task(store)
    hidden_dates = {
        task["protocol"]["holdout_start"],
        task["protocol"]["holdout_end"],
    }
    manager = FakeResearchTaskManager(store, ContractEvaluator())
    researcher = ContractResearcher([_proposal("safe")])
    module = __import__("autoalpha.service.system_jobs", fromlist=["run_factor_research"])
    real_run = module.run_factor_research

    def run_with_researcher(direction: str, **kwargs):  # noqa: ANN001
        return real_run(direction, researcher=researcher, **kwargs)

    monkeypatch.setattr(module, "run_factor_research", run_with_researcher)
    runtime_root = tmp_path / "runtime"
    _enqueue_factor_job(store)
    result = _runner(store, manager, runtime_root).run_next(queue="factor-research")
    assert result["job"]["status"] == "COMPLETED"

    monkeypatch.setattr(service_app, "store", store)
    monkeypatch.setenv("AUTOALPHA_SYSTEM_JOB_WORKER_ENABLED", "false")
    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get(
            f"/api/factor-research/runs/{result['job']['job_id']}"
        )

    assert response.status_code == 200
    payload_text = json.dumps(response.json(), ensure_ascii=False)
    artifact_path = next((runtime_root / "factor-research").glob("*/research_summary.json"))
    artifact_text = artifact_path.read_text(encoding="utf-8")
    for text in ("holdout_start", "holdout_end", "hidden_test_range", *hidden_dates):
        assert text not in payload_text
        assert text not in artifact_text
    context = response.json()["factor_research"]["result"]["research_context"]
    assert set(context) == {
        "research_task_id",
        "market",
        "snapshot",
        "protocol_hash",
        "evidence_tier",
        "exploration",
        "public_validation",
        "minimum_folds",
    }


def test_single_candidate_failure_does_not_crash_factor_research_job(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    manager = FakeResearchTaskManager(
        store, ContractEvaluator(fail_names={"bad_candidate"})
    )
    researcher = ContractResearcher([_proposal("bad_candidate")])
    module = __import__("autoalpha.service.system_jobs", fromlist=["run_factor_research"])
    real_run = module.run_factor_research

    def run_with_researcher(direction: str, **kwargs):  # noqa: ANN001
        return real_run(direction, researcher=researcher, **kwargs)

    monkeypatch.setattr(module, "run_factor_research", run_with_researcher)
    _enqueue_factor_job(store)

    result = _runner(store, manager, tmp_path / "runtime").run_next(
        queue="factor-research"
    )

    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["status_counts"][TRAIN_FAILED] == 1


def test_factor_research_system_job_preserves_partial_completed_status(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    manager = FakeResearchTaskManager(store, ContractEvaluator())
    researcher = ContractResearcher([_proposal("partial")])
    module = __import__("autoalpha.service.system_jobs", fromlist=["run_factor_research"])
    real_run = module.run_factor_research

    def run_with_budget(direction: str, **kwargs):  # noqa: ANN001
        return real_run(
            direction,
            researcher=researcher,
            maximum_llm_calls=1,
            **kwargs,
        )

    monkeypatch.setattr(module, "run_factor_research", run_with_budget)
    _enqueue_factor_job(store)

    result = _runner(store, manager, tmp_path / "runtime").run_next(
        queue="factor-research"
    )

    assert result["job"]["status"] == "PARTIAL_COMPLETED"
    assert result["job"]["result"]["status"] == "PARTIAL_COMPLETED"
    assert result["job"]["progress_current"] == 1
    assert result["job"]["progress_total"] == 1


def test_keep_enqueues_existing_factor_library_refresh_and_materializes_keep(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    manager = FakeResearchTaskManager(store, ContractEvaluator())
    researcher = ContractResearcher([_proposal("keep")])
    module = __import__("autoalpha.service.system_jobs", fromlist=["run_factor_research"])
    real_run = module.run_factor_research

    def run_with_researcher(direction: str, **kwargs):  # noqa: ANN001
        return real_run(direction, researcher=researcher, **kwargs)

    monkeypatch.setattr(module, "run_factor_research", run_with_researcher)

    def builder() -> dict:
        keeps = [item for item in store.factor_pool() if item["status"] == "KEEP"]
        return {
            "summary": {"factor_count": len(keeps)},
            "factors": [{"factor_id": item["factor_id"]} for item in keeps],
            "research_tasks": [{"task_id": "task-factor"}],
            "data": {},
            "knowledge_integrity": {"protocol": "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1"},
        }

    _enqueue_factor_job(store)
    runner = _runner(store, manager, tmp_path / "runtime", builder=builder)
    factor_result = runner.run_next(queue="factor-research")
    refresh_job = next(
        job
        for job in store.system_jobs(queue="system")
        if job["job_type"] == "factor_library_refresh"
    )

    assert factor_result["job"]["result"]["factor_library_refresh"]["queued"] is True
    assert refresh_job["payload"]["research_task_id"] == "task-factor"
    refresh_result = runner.run_next(queue="system")
    snapshot = store.materialized_snapshot("factor_library")
    keep_id = next(item["factor_id"] for item in store.factor_pool() if item["status"] == "KEEP")

    assert refresh_result["job"]["status"] == "COMPLETED"
    assert snapshot is not None
    assert keep_id in {item["factor_id"] for item in snapshot["payload"]["factors"]}


def test_keep_after_running_factor_library_refresh_enqueues_successor(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    running = store.enqueue_system_job(
        job_id="job-factor-library-running",
        queue="system",
        job_type="factor_library_refresh",
        payload={"source": "previous-factor-research"},
        resource_group="sqlite-writer",
        max_workers=1,
        progress_total=1,
    )
    store.update_system_job(
        running["job_id"],
        status="RUNNING",
        lease_owner="other-worker",
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    manager = FakeResearchTaskManager(store, ContractEvaluator())
    researcher = ContractResearcher([_proposal("keep-after-running")])
    module = __import__("autoalpha.service.system_jobs", fromlist=["run_factor_research"])
    real_run = module.run_factor_research

    def run_with_researcher(direction: str, **kwargs):  # noqa: ANN001
        return real_run(direction, researcher=researcher, **kwargs)

    monkeypatch.setattr(module, "run_factor_research", run_with_researcher)

    def builder() -> dict:
        keeps = [item for item in store.factor_pool() if item["status"] == "KEEP"]
        return {
            "summary": {"factor_count": len(keeps)},
            "factors": [{"factor_id": item["factor_id"]} for item in keeps],
            "research_tasks": [{"task_id": "task-factor"}],
            "data": {},
            "knowledge_integrity": {"protocol": "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1"},
        }

    _enqueue_factor_job(store)
    factor_result = _runner(
        store, manager, tmp_path / "runtime", builder=builder
    ).run_next(queue="factor-research")
    refresh_jobs = [
        job
        for job in store.system_jobs(queue="system")
        if job["job_type"] == "factor_library_refresh"
    ]
    successor = next(job for job in refresh_jobs if job["job_id"] != running["job_id"])

    assert factor_result["job"]["result"]["factor_library_refresh"] == {
        "queued": True,
        "deduplicated": False,
        "job_id": successor["job_id"],
        "queue": "system",
    }
    assert successor["status"] == "QUEUED"
    assert successor["job_id"] != running["job_id"]


def test_factor_research_pause_resume_is_explicitly_unsupported(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    _enqueue_factor_job(store)

    with pytest.raises(RuntimeError, match="does not support pause/resume"):
        store.command_system_job("job-factor-research", command="pause")


def test_factor_research_post_contract_uses_task_id_and_separate_queue(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store)
    monkeypatch.setattr(service_app, "store", store)
    monkeypatch.setenv("AUTOALPHA_SYSTEM_JOB_WORKER_ENABLED", "false")
    monkeypatch.setattr(
        service_app.research_manager,
        "readiness",
        lambda task_id: {"runnable": True, "blockers": []},
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post(
            "/api/factor-research/runs",
            json={
                "research_task_id": "task-factor",
                "research_direction": "短期反转",
                "candidates_per_round": 4,
                "rounds": 3,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job"]["queue"] == "factor-research"
    assert payload["job"]["job_type"] == "factor_research"
    assert payload["progress"] == {"current": 0, "total": 12}
    assert payload["request"] == {
        "research_task_id": "task-factor",
        "research_direction": "短期反转",
        "candidates_per_round": 4,
        "rounds": 3,
    }
    assert "data_path" not in json.dumps(payload, ensure_ascii=False)


def test_factor_research_get_contract_allow_lists_artifacts_and_forbids_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.enqueue_system_job(
        job_id="job-factor-get",
        queue="factor-research",
        job_type="factor_research",
        payload={
            "research_task_id": "task-factor",
            "research_direction": "短期反转",
            "candidates_per_round": 1,
            "rounds": 1,
        },
        progress_total=1,
    )
    store.update_system_job(
        "job-factor-get",
        status="COMPLETED",
        progress_current=1,
        error="failed at C:/private/panel with sk-provider-secret",
        result={
            "run_id": "factor-research-run",
            "status": "COMPLETED",
            "research_direction": "短期反转",
            "candidates": [
                {
                    "status": "KEEP",
                    "reason": "ok",
                    "proposal": {
                        **_proposal("safe"),
                        "api_key": "sk-provider-secret",
                    },
                }
            ],
            "counts": {"requested": 1, "proposed": 1, "keep": 1, "by_status": {"KEEP": 1}},
            "simple_portfolio": {"status": "NOT_EVALUATED"},
            "llm_conclusion": "ok",
            "artifacts": {
                "json": "C:/private/research_summary.json",
                "markdown": "C:/private/research_summary.md",
            },
            "production_promotion": {"allowed": True},
        },
        lease_owner=None,
        lease_expires_at=None,
        heartbeat_at=None,
        finished_at="2026-08-14T00:00:00+00:00",
    )
    monkeypatch.setattr(service_app, "store", store)
    monkeypatch.setenv("AUTOALPHA_SYSTEM_JOB_WORKER_ENABLED", "false")

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/factor-research/runs/job-factor-get")

    assert response.status_code == 200
    payload = response.json()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "C:/private" not in encoded
    assert "sk-provider-secret" not in encoded
    assert (
        payload["factor_research"]["result"]["production_promotion"]["allowed"]
        is False
    )
    assert payload["factor_research"]["artifacts"] == [
        {
            "artifact_id": "factor-research/factor-research-run/research_summary.json",
            "name": "research_summary.json",
            "media_type": "application/json",
        },
        {
            "artifact_id": "factor-research/factor-research-run/research_summary.md",
            "name": "research_summary.md",
            "media_type": "text/markdown",
        },
    ]
