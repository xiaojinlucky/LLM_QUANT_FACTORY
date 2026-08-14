from pathlib import Path

STATIC = Path(__file__).parents[2] / "src" / "autoalpha" / "service" / "static"
APPLICATION = STATIC.parent / "app.py"
PAGE = STATIC / "factor_research.html"
SCRIPT = STATIC / "factor_research.js"


def test_factor_research_page_is_served_and_research_task_links_to_bound_context() -> None:
    application = APPLICATION.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    research_tasks = (STATIC / "research_tasks.js").read_text(encoding="utf-8")

    assert '@app.get("/factor-research", include_in_schema=False)' in application
    assert 'FileResponse(PACKAGE_ROOT / "static/factor_research.html")' in application
    assert 'id="researchDirection"' in page
    assert 'id="candidatesPerRound">4' in page
    assert 'id="rounds">3' in page
    assert 'factor-research?research_task_id=' in research_tasks


def test_factor_research_page_uses_frozen_request_and_cancel_contract() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "research_task_id: appState.taskId" in script
    assert "research_direction: direction" in script
    assert "candidates_per_round: 4" in script
    assert "rounds: 3" in script
    assert "/api/factor-research/runs" in script
    assert "/api/factor-research/runs/${encodeURIComponent(appState.jobId)}" in script
    assert "/api/jobs/${encodeURIComponent(appState.jobId)}/cancel" in script
    assert "actor: \"local-operator\"" in script
    assert "factor_library_refresh" in script


def test_running_state_does_not_render_candidate_evidence_until_terminal_result() -> None:
    page = PAGE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'id="resultsGrid" hidden' in page
    assert 'id="candidateRows"' in page
    assert 'document.getElementById("terminalState").hidden = active' in script
    assert 'document.getElementById("resultsGrid").hidden = !showCandidates' in script
    assert '候选证据将在研究结束后显示' in page
    assert "KEEP count" not in script
    assert "正在验证" not in script
    assert "AI 正在思考" not in script


def test_candidate_table_and_evidence_share_core_metrics_and_exploration_allowlist() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    exploration = script.split("function evidenceExploration", 1)[1].split(
        "function evidencePublicValidation", 1
    )[0]

    for key in ("rank_ic", "sharpe", "annual_return", "max_drawdown", "turnover", "coverage"):
        assert f"candidate.core_metrics?.{key}" in script or "core[key]" in script
    assert "formatMetric(candidate.core_metrics?.rank_ic" in script
    assert "formatMetric(candidate.core_metrics?.coverage" in script
    assert "metrics.sharpe" in exploration
    assert "metrics.simple_annual_return" in exploration
    assert "metrics.max_drawdown" in exploration
    assert "rank_ic" not in exploration
    assert "coverage" not in exploration
    assert "turnover" not in exploration


def test_dynamic_research_content_is_rendered_without_raw_inner_html() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert ".innerHTML" not in script
    assert "textContent" in script
    assert "createElement" in script
    assert "formatExpression" in script
    assert "research_context" not in script
    assert "holdout_start" not in script
    assert "holdout_end" not in script
    assert "production_promotion" not in script


def test_terminal_status_and_refresh_fallbacks_are_explicit() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "未知状态 · ${raw}" in script
    assert 'PARTIAL_COMPLETED: "部分完成"' in script
    assert 'NO_KEEP: "无保留候选"' in script
    assert '"因子库刷新中"' in script
    assert '"打开因子库继续复核"' in script
    assert '"因子库刷新失败"' in script
    assert '"打开现有因子库"' in script
