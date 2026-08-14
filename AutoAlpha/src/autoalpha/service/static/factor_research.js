const FACTOR_RESEARCH_STATUS_LABELS = Object.freeze({
  QUEUED: "排队中",
  RUNNING: "运行中",
  CANCEL_REQUESTED: "取消请求中",
  CANCELLED: "已取消",
  COMPLETED: "已完成",
  PARTIAL_COMPLETED: "部分完成",
  FAILED: "失败",
  NO_KEEP: "无保留候选",
  KEEP: "保留",
  INVALID: "表达式无效",
  DUPLICATE: "重复候选",
  VALIDATION_FAILED: "公开验证未通过",
  TRAIN_FAILED: "探索期失败",
  BUDGET_EXHAUSTED: "研究预算耗尽",
  REJECTED: "未通过研究准入",
});

const FACTOR_RESEARCH_STATUS_TONES = Object.freeze({
  KEEP: "keep",
  RUNNING: "running",
  QUEUED: "running",
  CANCEL_REQUESTED: "running",
  INVALID: "reject",
  DUPLICATE: "reject",
  VALIDATION_FAILED: "reject",
  TRAIN_FAILED: "error",
  FAILED: "error",
  CANCELLED: "neutral",
  COMPLETED: "neutral",
  PARTIAL_COMPLETED: "neutral",
  NO_KEEP: "neutral",
});

const CORE_METRICS = Object.freeze([
  ["Rank IC", "rank_ic", "rankIc"],
  ["Sharpe", "sharpe", "sharpe"],
  ["年化收益", "annual_return", "annualPercent"],
  ["最大回撤", "max_drawdown", "drawdownPercent"],
  ["换手", "turnover", "turnover"],
  ["覆盖率", "coverage", "coverage"],
]);

const appState = {
  task: null,
  taskId: null,
  jobId: null,
  jobView: null,
  selectedCandidateIndex: null,
  refreshJob: null,
  pollTimer: null,
  refreshPollTimer: null,
  cancelling: false,
};

document.addEventListener("DOMContentLoaded", () => {
  bindFactorResearchControls();
  initializeFactorResearch();
});

function bindFactorResearchControls() {
  document.getElementById("launchForm").addEventListener("submit", startFactorResearch);
  document.getElementById("cancelRun").addEventListener("click", cancelFactorResearch);
  document.getElementById("closeEvidence").addEventListener("click", closeEvidence);
  document.getElementById("roundFilter").addEventListener("change", renderCandidateTable);
  document.getElementById("statusFilter").addEventListener("change", renderCandidateTable);
  document.getElementById("candidateSearch").addEventListener("input", renderCandidateTable);
}

async function initializeFactorResearch() {
  const params = new URLSearchParams(window.location.search);
  appState.taskId = params.get("research_task_id") || "";
  appState.jobId = params.get("job_id") || "";
  if (!appState.taskId && !appState.jobId) {
    showEntryBlocked();
    return;
  }
  try {
    if (appState.jobId) {
      await loadFactorResearchRun();
    } else {
      if (!appState.taskId) throw new Error("因子研究运行没有关联 ResearchTask");
      appState.task = await api(`/api/research-tasks/${encodeURIComponent(appState.taskId)}`);
      renderTaskContext();
      updateTaskLinks();
      renderLaunchState();
    }
  } catch (error) {
    showEntryError(error);
  }
}

async function startFactorResearch(event) {
  event.preventDefault();
  if (!appState.taskId || appState.jobId) return;
  const directionNode = document.getElementById("researchDirection");
  const direction = directionNode.value.trim();
  if (!direction) {
    directionNode.focus();
    showToast("请输入研究方向", true);
    return;
  }
  const button = document.getElementById("startResearch");
  button.disabled = true;
  try {
    const response = await api("/api/factor-research/runs", {
      method: "POST",
      body: JSON.stringify({
        research_task_id: appState.taskId,
        research_direction: direction,
        candidates_per_round: 4,
        rounds: 3,
      }),
    });
    appState.jobId = response.job?.job_id || "";
    if (!appState.jobId) throw new Error("因子研究作业未返回 job_id");
    history.replaceState(null, "", factorResearchUrl(appState.taskId, appState.jobId));
    appState.jobView = response;
    appState.cancelling = false;
    renderRunState();
    scheduleRunPoll();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function loadFactorResearchRun() {
  if (!appState.jobId) return;
  try {
    appState.jobView = await api(`/api/factor-research/runs/${encodeURIComponent(appState.jobId)}`);
    const runTaskId = String(appState.jobView.request?.research_task_id || "");
    if (!runTaskId) throw new Error("因子研究运行没有关联 ResearchTask");
    const urlTaskId = String(appState.taskId || "");
    if (urlTaskId !== runTaskId) {
      appState.taskId = runTaskId;
      history.replaceState(null, "", factorResearchUrl(appState.taskId, appState.jobId));
    }
    if (!appState.task || String(appState.task.task_id || "") !== runTaskId) {
      appState.taskId = runTaskId;
      appState.task = await api(`/api/research-tasks/${encodeURIComponent(appState.taskId)}`);
      renderTaskContext();
      updateTaskLinks();
    }
    renderRunState();
    if (isRunActive()) scheduleRunPoll();
  } catch (error) {
    showEntryError(error);
  }
}

function scheduleRunPoll() {
  window.clearTimeout(appState.pollTimer);
  if (!isRunActive()) return;
  appState.pollTimer = window.setTimeout(async () => {
    await loadFactorResearchRun();
  }, 1600);
}

function isRunActive() {
  const status = String(appState.jobView?.job?.status || "");
  return ["QUEUED", "RUNNING", "CANCEL_REQUESTED"].includes(status);
}

async function cancelFactorResearch() {
  if (!appState.jobId || appState.cancelling || !isRunActive()) return;
  appState.cancelling = true;
  const button = document.getElementById("cancelRun");
  button.disabled = true;
  try {
    const response = await api(`/api/jobs/${encodeURIComponent(appState.jobId)}/cancel`, {
      method: "POST",
      body: JSON.stringify({ actor: "local-operator", reason: "factor research UI cancel" }),
    });
    if (appState.jobView?.job) appState.jobView.job.status = response.status || "CANCEL_REQUESTED";
    renderRunState();
    scheduleRunPoll();
  } catch (error) {
    appState.cancelling = false;
    button.disabled = false;
    showToast(error.message, true);
  }
}

function renderTaskContext() {
  const task = appState.task;
  if (!task) return;
  document.getElementById("contextPanel").hidden = false;
  setText("taskIdentity", task.name || task.task_id || "研究任务");
  document.getElementById("taskIdentity").title = task.task_id || "";
  setText("contextTaskName", task.name || task.task_id || "研究任务");
  setText("contextTaskId", task.task_id || "—");
  setText("contextMarket", marketLabel(task.market));
  setText("contextSnapshot", task.snapshot_hash || "—");
  setText("contextTaskStatus", taskStatusLabel(task.status));
  setStatusTone(document.getElementById("contextTaskStatus"), task.status);
  const runnable = task.readiness?.runnable;
  setText("contextReadiness", runnable === false ? "暂不可运行" : "任务上下文已绑定");
  const startButton = document.getElementById("startResearch");
  startButton.disabled = runnable === false;
  setText("entryState", runnable === false ? "当前 ResearchTask 尚未满足研究条件" : "研究会绑定当前 ResearchTask");
}

function updateTaskLinks() {
  const href = appState.taskId
    ? `/research-tasks/${encodeURIComponent(appState.taskId)}`
    : "/research-tasks";
  document.getElementById("returnTaskLink").href = href;
  document.getElementById("runTaskLink").href = href;
}

function renderLaunchState() {
  document.getElementById("entryPanel").hidden = false;
  document.getElementById("runPanel").hidden = true;
  document.getElementById("entryBlocked").hidden = true;
  setText("candidateBudget", 4 * 3);
}

function renderRunState() {
  const view = appState.jobView;
  if (!view) return;
  document.getElementById("entryPanel").hidden = true;
  document.getElementById("entryBlocked").hidden = true;
  document.getElementById("runPanel").hidden = false;
  const job = view.job || {};
  const request = view.request || {};
  const status = String(job.status || view.factor_research?.status || "UNKNOWN");
  const progress = view.progress || {
    current: job.progress_current || 0,
    total: job.progress_total || 0,
  };
  const current = finiteCount(progress.current);
  const total = finiteCount(progress.total);
  const result = view.factor_research?.result || null;
  setText("runId", view.factor_research?.run_id || appState.jobId || "—");
  setText("runDirection", request.research_direction || "—");
  setText("requestDirection", request.research_direction || "—");
  setText("requestBudget", budgetLabel(request.candidates_per_round, request.rounds));
  setText("runStatus", statusLabel(status));
  setStatusTone(document.getElementById("runStatus"), status);
  setText("runningLifecycle", statusLabel(status));
  setText("progressValue", `${current} / ${total}`);
  const progressBar = document.getElementById("progressBar");
  progressBar.style.width = progressWidth(current, total);
  const active = isRunActive();
  document.getElementById("runningState").hidden = !active;
  document.getElementById("cancelRun").hidden = !active;
  document.getElementById("cancelRun").disabled = appState.cancelling || status === "CANCEL_REQUESTED";
  document.getElementById("terminalState").hidden = active;
  if (!active) renderTerminalState(status, result, job.error);
}

function renderTerminalState(status, result, error) {
  const hasResult = Boolean(result && typeof result === "object");
  const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
  const resultStatus = String(result?.status || status);
  const terminalSupportsEvidence = hasResult && [
    "COMPLETED",
    "PARTIAL_COMPLETED",
    "CANCELLED",
    "NO_KEEP",
  ].includes(status) || ["COMPLETED", "PARTIAL_COMPLETED", "NO_KEEP"].includes(resultStatus);
  const keepCount = finiteCount(result?.counts?.keep);
  const noKeep = terminalSupportsEvidence && (resultStatus === "NO_KEEP" || keepCount === 0);
  setText("terminalSummary", "");
  const summary = document.getElementById("terminalSummary");
  summary.replaceChildren(...terminalSummaryItems(result, candidates, status));
  document.getElementById("admissionNote").hidden = !terminalSupportsEvidence;
  document.getElementById("noKeepNote").hidden = !noKeep;
  const errorNode = document.getElementById("terminalError");
  const shouldShowError = status === "FAILED" || (!hasResult && Boolean(error));
  errorNode.hidden = !shouldShowError;
  if (shouldShowError) errorNode.textContent = safeError(error || `研究运行状态：${statusLabel(status)}`);

  const showCandidates = terminalSupportsEvidence && candidates.length > 0;
  document.getElementById("resultsToolbar").hidden = !showCandidates;
  document.getElementById("resultsGrid").hidden = !showCandidates;
  document.getElementById("emptyResult").hidden = showCandidates || shouldShowError;
  if (showCandidates) {
    prepareCandidateFilters(candidates);
    if (!candidateIndexExists(appState.selectedCandidateIndex, candidates)) {
      appState.selectedCandidateIndex = 0;
    }
    renderCandidateTable();
    renderEvidence();
  } else {
    document.getElementById("candidateRows").replaceChildren();
    document.getElementById("evidencePanel").hidden = true;
    document.getElementById("resultsGrid").classList.remove("has-evidence");
    if (!shouldShowError) document.getElementById("emptyResult").hidden = false;
  }
  if (hasResult) loadFactorLibraryRefresh(result);
}

function terminalSummaryItems(result, candidates, status) {
  const counts = result?.counts || {};
  const statusCounts = result?.status_counts || {};
  const items = [
    summaryItem("状态", statusLabel(status), status),
    summaryItem("候选", firstFinite(counts.proposed, candidates.length)),
    summaryItem("保留", firstFinite(counts.keep, statusCounts.KEEP, 0), "KEEP"),
  ];
  const order = ["INVALID", "DUPLICATE", "VALIDATION_FAILED", "TRAIN_FAILED"];
  order.forEach(raw => {
    if (Object.prototype.hasOwnProperty.call(statusCounts, raw)) {
      items.push(summaryItem(statusLabel(raw), statusCounts[raw], raw));
    }
  });
  return items;
}

function summaryItem(label, value, raw = "") {
  const nodeValue = element("strong", "", displayValue(value));
  const node = element("span", "fr-summary-item");
  node.append(element("span", "", `${label} `), nodeValue);
  if (raw) node.append(element("code", "", raw));
  return node;
}

function prepareCandidateFilters(candidates) {
  const rounds = [...new Set(candidates.map(candidate => finiteCount(candidate.round)).filter(value => value !== null))];
  rounds.sort((a, b) => a - b);
  const roundFilter = document.getElementById("roundFilter");
  const previousRound = roundFilter.value;
  roundFilter.replaceChildren(element("option", "", "全部轮次"));
  roundFilter.firstChild.value = "all";
  rounds.forEach(round => {
    const option = element("option", "", `第 ${round} 轮`);
    option.value = String(round);
    roundFilter.append(option);
  });
  roundFilter.value = rounds.some(round => String(round) === previousRound) ? previousRound : "all";

  const statuses = [];
  candidates.forEach(candidate => {
    const status = String(candidate.status || "UNKNOWN");
    if (!statuses.includes(status)) statuses.push(status);
  });
  const statusFilter = document.getElementById("statusFilter");
  const previousStatus = statusFilter.value;
  statusFilter.replaceChildren(element("option", "", "全部状态"));
  statusFilter.firstChild.value = "all";
  statuses.forEach(status => {
    const option = element("option", "", statusLabel(status));
    option.value = status;
    statusFilter.append(option);
  });
  statusFilter.value = statuses.includes(previousStatus) ? previousStatus : "all";
}

function renderCandidateTable() {
  const candidateEntries = currentCandidateEntries();
  const round = document.getElementById("roundFilter").value;
  const status = document.getElementById("statusFilter").value;
  const query = document.getElementById("candidateSearch").value.trim().toLowerCase();
  const filtered = candidateEntries.filter(({ candidate }) => {
    const candidateRound = String(finiteCount(candidate.round) ?? "");
    const candidateStatus = String(candidate.status || "UNKNOWN");
    const factorName = String(candidate.factor_name || candidate.normalized_candidate?.name || "");
    return (round === "all" || candidateRound === round)
      && (status === "all" || candidateStatus === status)
      && (!query || factorName.toLowerCase().includes(query));
  });
  setText("candidateCountLabel", `${filtered.length} / ${candidateEntries.length}`);
  const body = document.getElementById("candidateRows");
  if (!filtered.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 10;
    cell.className = "table-empty";
    cell.textContent = "没有符合条件的候选";
    row.append(cell);
    body.replaceChildren(row);
    return;
  }
  body.replaceChildren(...filtered.map(({ candidate, index }) => candidateRow(candidate, index)));
}

function candidateRow(candidate, index) {
  const candidateId = String(candidate.candidate_id || "");
  const row = document.createElement("tr");
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.setAttribute("aria-label", `查看候选 ${candidate.factor_name || candidateId}`);
  row.classList.toggle("selected", index === appState.selectedCandidateIndex);
  row.addEventListener("click", () => selectCandidate(index));
  row.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectCandidate(index);
    }
  });
  const statusCell = document.createElement("td");
  statusCell.className = "fr-status-cell";
  statusCell.append(statusPill(candidate.status), element("code", "fr-status-code", String(candidate.status || "UNKNOWN")));
  const factorCell = document.createElement("td");
  factorCell.className = "factor-cell";
  factorCell.append(
    element("strong", "", candidate.factor_name || candidate.normalized_candidate?.name || "—"),
    element("code", "", candidateId || "—"),
  );
  row.append(
    statusCell,
    factorCell,
    textCell(`R${displayValue(finiteCount(candidate.round))}`),
    textCell(displayValue(finiteCount(candidate.repair_count))),
    numericCell(formatMetric(candidate.core_metrics?.rank_ic, "rankIc")),
    numericCell(formatMetric(candidate.core_metrics?.sharpe, "sharpe")),
    numericCell(formatMetric(candidate.core_metrics?.annual_return, "annualPercent")),
    numericCell(formatMetric(candidate.core_metrics?.max_drawdown, "drawdownPercent")),
    numericCell(formatMetric(candidate.core_metrics?.turnover, "turnover")),
    numericCell(formatMetric(candidate.core_metrics?.coverage, "coverage")),
  );
  return row;
}

function selectCandidate(index) {
  if (!candidateIndexExists(index, currentCandidates())) return;
  appState.selectedCandidateIndex = index;
  renderCandidateTable();
  renderEvidence();
  if (window.innerWidth <= 860) document.getElementById("evidencePanel").scrollIntoView({ block: "start", behavior: "smooth" });
}

function closeEvidence() {
  appState.selectedCandidateIndex = null;
  document.getElementById("evidencePanel").hidden = true;
  document.getElementById("resultsGrid").classList.remove("has-evidence");
  renderCandidateTable();
}

function renderEvidence() {
  const candidates = currentCandidates();
  const candidate = candidateIndexExists(appState.selectedCandidateIndex, candidates)
    ? candidates[appState.selectedCandidateIndex]
    : null;
  const panel = document.getElementById("evidencePanel");
  if (!candidate) {
    panel.hidden = true;
    document.getElementById("resultsGrid").classList.remove("has-evidence");
    return;
  }
  panel.hidden = false;
  document.getElementById("resultsGrid").classList.add("has-evidence");
  setText("evidenceTitle", candidate.factor_name || candidate.normalized_candidate?.name || candidate.candidate_id || "候选证据");
  const spine = document.getElementById("evidenceSpine");
  spine.replaceChildren(
    evidenceStructure(candidate),
    evidenceExploration(candidate),
    evidencePublicValidation(candidate),
    evidenceDecision(candidate),
    evidenceAuxiliary(),
  );
}

function evidenceStructure(candidate) {
  const normalized = candidate.normalized_candidate || {};
  const expression = candidate.expression ?? normalized.expression;
  const fields = Array.isArray(candidate.fields) ? candidate.fields : (Array.isArray(normalized.fields) ? normalized.fields : []);
  const section = evidenceSection("候选结构");
  section.append(definitionList([
    ["因子名", candidate.factor_name || normalized.name || "—", "text"],
    ["假设", candidate.hypothesis || normalized.hypothesis || "—", "text"],
    ["表达式", formatExpression(expression), "code text"],
    ["字段", fieldList(fields), "node"],
  ]));
  return section;
}

function evidenceExploration(candidate) {
  const metrics = candidate.exploration_metrics || {};
  const section = evidenceSection("探索期");
  section.append(definitionList([
    ["时间范围", dateRange(metrics.start, metrics.end), "text"],
    ["观测数", formatCount(metrics.observations), ""],
    ["Sharpe", formatMetric(metrics.sharpe, "sharpe"), ""],
    ["年化收益", formatMetric(metrics.simple_annual_return, "annualPercent"), ""],
    ["最大回撤", formatMetric(metrics.max_drawdown, "drawdownPercent"), ""],
  ]));
  return section;
}

function evidencePublicValidation(candidate) {
  const core = candidate.core_metrics || {};
  const publicMetrics = candidate.public_validation_metrics || {};
  const section = evidenceSection("公开验证期");
  const rows = CORE_METRICS.map(([label, key, kind]) => [label, formatMetric(core[key], kind), ""]);
  rows.push(["观测数", formatCount(firstFinite(publicMetrics.observations, core.observations)), ""]);
  section.append(definitionList(rows));
  return section;
}

function evidenceDecision(candidate) {
  const core = candidate.core_metrics || {};
  const section = evidenceSection("关系与判定");
  section.append(definitionList([
    ["候选状态", statusLabel(candidate.status), "text"],
    ["修复次数", displayValue(finiteCount(candidate.repair_count)), ""],
    ["父候选", displayValue(candidate.parent_lineage), "code text"],
    ["重复候选", displayValue(candidate.duplicate_peer), "code text"],
    ["拒绝原因", displayValue(candidate.rejection_reason), "text"],
    ["符号一致性", booleanLabel(core.sign_consistency), "text"],
    ["破产状态", booleanLabel(core.bankrupt, "否", "是"), "text"],
  ]));
  return section;
}

function evidenceAuxiliary() {
  const result = appState.jobView?.factor_research?.result || {};
  const section = evidenceSection("辅助信息");
  const portfolio = result.simple_portfolio;
  if (portfolio && typeof portfolio === "object" && Object.keys(portfolio).length) {
    section.append(collapsedDetail("简单组合结果", structuredPortfolioText(portfolio)));
  }
  if (String(result.llm_conclusion || "").trim()) {
    section.append(collapsedDetail("AI 研究总结 / 辅助结论", String(result.llm_conclusion)));
  }
  const artifacts = Array.isArray(result.artifacts) ? result.artifacts : [];
  if (artifacts.length) section.append(artifactList(artifacts));
  if (appState.refreshJob?.jobId || result.factor_library_refresh?.job_id) {
    section.append(libraryRefreshNode());
  }
  if (!section.children.length) section.append(element("p", "", "暂无辅助信息"));
  return section;
}

function evidenceSection(title) {
  const section = element("section", "fr-evidence-section");
  section.append(element("h4", "", title));
  return section;
}

function definitionList(rows) {
  const list = element("dl", "fr-definition-list");
  rows.forEach(([label, value, kind]) => {
    const term = element("dt", "", label);
    const detail = element("dd", kind?.includes("text") ? "text-value" : "");
    if (kind?.includes("node") && value instanceof Node) detail.append(value);
    else if (kind?.includes("code")) detail.append(element("code", "", displayValue(value)));
    else detail.textContent = displayValue(value);
    list.append(term, detail);
  });
  return list;
}

function fieldList(fields) {
  const list = element("span", "fr-fields");
  if (!fields.length) return element("span", "", "—");
  fields.forEach(field => list.append(element("span", "fr-field-chip", displayValue(field))));
  return list;
}

function collapsedDetail(title, value) {
  const details = element("details", "fr-detail-toggle");
  const summary = document.createElement("summary");
  summary.textContent = title;
  const body = element("div", "fr-detail-body");
  const pre = document.createElement("pre");
  pre.textContent = value;
  body.append(pre);
  details.append(summary, body);
  return details;
}

function artifactList(artifacts) {
  const wrapper = element("div", "fr-artifact-list");
  artifacts.forEach(artifact => {
    if (!artifact || typeof artifact !== "object") return;
    const item = element("div", "fr-artifact-item");
    item.append(
      element("strong", "", artifact.name || "研究摘要"),
      element("code", "", `${artifact.media_type || ""} · ${artifact.artifact_id || ""}`),
    );
    wrapper.append(item);
  });
  return wrapper;
}

function structuredPortfolioText(portfolio) {
  const allowed = ["status", "method", "factor_ids", "weights", "metrics", "factor_correlations", "reason"];
  const safe = {};
  allowed.forEach(key => {
    if (Object.prototype.hasOwnProperty.call(portfolio, key)) safe[key] = portfolio[key];
  });
  return JSON.stringify(safe, null, 2);
}

async function loadFactorLibraryRefresh(result) {
  const metadata = result?.factor_library_refresh;
  if (!metadata?.job_id) {
    appState.refreshJob = null;
    return;
  }
  const jobId = String(metadata.job_id);
  if (appState.refreshJob?.jobId === jobId) {
    renderEvidence();
    return;
  }
  appState.refreshJob = { jobId, status: "UNKNOWN", retryCount: 0, retryable: true };
  renderEvidence();
  await pollFactorLibraryRefresh();
}

function scheduleRefreshPoll() {
  window.clearTimeout(appState.refreshPollTimer);
  if (!shouldPollRefresh()) return;
  appState.refreshPollTimer = window.setTimeout(async () => {
    await pollFactorLibraryRefresh();
  }, 2200);
}

async function pollFactorLibraryRefresh() {
  const refreshJob = appState.refreshJob;
  if (!refreshJob) return;
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(refreshJob.jobId)}/logs?limit=1`);
    const rawStatus = String(data.job?.status || "UNKNOWN");
    refreshJob.status = ["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "BLOCKED_UNSUPPORTED"].includes(rawStatus)
      ? rawStatus
      : "UNKNOWN";
    refreshJob.retryCount = 0;
    refreshJob.retryable = false;
  } catch (error) {
    refreshJob.status = "UNKNOWN";
    refreshJob.retryCount = (refreshJob.retryCount || 0) + 1;
    refreshJob.retryable = refreshJob.retryCount < 3;
  }
  renderEvidence();
  scheduleRefreshPoll();
}

function shouldPollRefresh() {
  const refreshJob = appState.refreshJob;
  if (!refreshJob) return false;
  if (["QUEUED", "RUNNING"].includes(refreshJob.status)) return true;
  return refreshJob.status === "UNKNOWN" && refreshJob.retryable === true;
}

function libraryRefreshNode() {
  const wrapper = element("div", "fr-library-status");
  const status = appState.refreshJob?.status || "UNKNOWN";
  if (["QUEUED", "RUNNING"].includes(status)) {
    wrapper.append(element("strong", "", "因子库刷新中"), element("div", "", "本次保留因子尚未承诺已出现在因子库。"));
    return wrapper;
  }
  if (status === "COMPLETED") {
    wrapper.append(element("strong", "", "因子库已刷新"));
    const link = element("a", "", "打开因子库继续复核");
    link.href = "/factors";
    wrapper.append(link);
    return wrapper;
  }
  if (["FAILED", "CANCELLED", "BLOCKED_UNSUPPORTED"].includes(status)) {
    wrapper.append(element("strong", "", "因子库刷新失败"));
    const link = element("a", "", "打开现有因子库");
    link.href = "/factors";
    wrapper.append(link);
    return wrapper;
  }
  wrapper.append(element("strong", "", `因子库刷新状态 · ${statusLabel(status)}`));
  return wrapper;
}

function currentCandidates() {
  return Array.isArray(appState.jobView?.factor_research?.result?.candidates)
    ? appState.jobView.factor_research.result.candidates
    : [];
}

function currentCandidateEntries() {
  return currentCandidates().map((candidate, index) => ({ candidate, index }));
}

function candidateIndexExists(index, candidates = currentCandidates()) {
  return Number.isInteger(index) && index >= 0 && index < candidates.length;
}

function statusPill(rawStatus) {
  const raw = String(rawStatus || "UNKNOWN");
  const pill = element("span", `fr-status-pill ${statusTone(raw)}`, statusLabel(raw));
  pill.title = raw;
  return pill;
}

function setStatusTone(node, rawStatus) {
  node.className = `fr-status-pill ${statusTone(rawStatus)}`;
}

function statusLabel(rawStatus) {
  const raw = String(rawStatus || "UNKNOWN");
  return FACTOR_RESEARCH_STATUS_LABELS[raw] || `未知状态 · ${raw}`;
}

function taskStatusLabel(rawStatus) {
  const labels = { READY: "已就绪", DATA_REQUIRED: "需要数据", DRAFT: "草稿", RUNNING: "运行中", STOPPING: "停止中" };
  const raw = String(rawStatus || "UNKNOWN");
  return labels[raw] || `未知任务状态 · ${raw}`;
}

function statusTone(rawStatus) {
  return FACTOR_RESEARCH_STATUS_TONES[String(rawStatus || "UNKNOWN")] || "neutral";
}

function marketLabel(value) {
  return ({ CN_A: "A 股", HK: "港股", US: "美股" })[value] || displayValue(value);
}

function budgetLabel(candidatesPerRound, rounds) {
  const perRound = finiteCount(candidatesPerRound) ?? 4;
  const runCount = finiteCount(rounds) ?? 3;
  return `${perRound} × ${runCount} · 最大候选 ${perRound * runCount}`;
}

function formatMetric(value, kind) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  if (kind === "rankIc") return number.toFixed(3);
  if (kind === "sharpe") return number.toFixed(2);
  if (kind === "turnover") return number.toFixed(2);
  const percent = number * 100;
  const sign = kind === "coverage" ? "" : percent > 0 ? "+" : "";
  return `${sign}${percent.toFixed(1)}%`;
}

function formatCount(value) {
  const number = finiteCount(value);
  return number === null ? "—" : String(number);
}

function dateRange(start, end) {
  if (!start || !end) return "—";
  return `${displayValue(start)} — ${displayValue(end)}`;
}

function formatExpression(expression) {
  if (typeof expression === "string") return expression;
  if (!expression || typeof expression !== "object") return "—";
  return expressionText(expression);
}

function expressionText(expression) {
  const operator = String(expression.operator || expression.op || "expression");
  const parameters = expression.parameters && typeof expression.parameters === "object" ? expression.parameters : {};
  const argumentsList = Array.isArray(expression.arguments) ? expression.arguments.map(expressionText) : [];
  if (operator === "field") return displayValue(parameters.name || parameters.field);
  if (operator === "constant") return displayValue(parameters.value);
  const symbols = { add: "+", subtract: "−", multiply: "×", divide: "÷" };
  if (symbols[operator] && argumentsList.length >= 2) return `(${argumentsList[0]} ${symbols[operator]} ${argumentsList[1]})`;
  const title = operator.replaceAll("_", " ").replace(/(^|\s)\S/g, value => value.toUpperCase()).replaceAll(" ", "");
  const parameterText = Object.entries(parameters)
    .filter(([key]) => !["name", "field"].includes(key))
    .map(([key, value]) => `${key}=${displayValue(value)}`);
  return `${title}(${[...argumentsList, ...parameterText].join(", ")})`;
}

function booleanLabel(value, falseLabel = "—", trueLabel = "是") {
  if (value === true) return trueLabel;
  if (value === false) return falseLabel === "—" ? "否" : falseLabel;
  return "—";
}

function textCell(value) {
  return element("td", "", displayValue(value));
}

function numericCell(value) {
  return element("td", "numeric", value);
}

function firstFinite(...values) {
  for (const value of values) {
    const number = finiteNumber(value);
    if (number !== null) return number;
  }
  return null;
}

function finiteCount(value) {
  const number = finiteNumber(value);
  return number === null ? null : Math.max(0, Math.trunc(number));
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function progressWidth(current, total) {
  if (!total || total < 1) return "0%";
  return `${Math.min(100, Math.max(0, (current / total) * 100))}%`;
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number" && !Number.isFinite(value)) return "—";
  if (Array.isArray(value)) return value.length ? value.map(displayValue).join("、") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function safeError(error) {
  if (typeof error === "string") return error;
  if (error && typeof error.message === "string") return error.message;
  return "研究运行失败，请查看作业状态。";
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = displayValue(value);
}

function element(tag, className = "", value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = displayValue(value);
  return node;
}

function showEntryBlocked() {
  document.getElementById("entryPanel").hidden = true;
  document.getElementById("contextPanel").hidden = true;
  document.getElementById("runPanel").hidden = true;
  document.getElementById("entryBlocked").hidden = false;
}

function showEntryError(error) {
  showEntryBlocked();
  const node = document.getElementById("entryBlocked");
  node.replaceChildren(
    element("h2", "", "无法读取研究任务"),
    element("div", "", safeError(error)),
    (() => {
      const link = element("a", "button", "打开任务总表");
      link.href = "/research-tasks";
      return link;
    })(),
  );
}

function factorResearchUrl(taskId, jobId) {
  const params = new URLSearchParams({ research_task_id: taskId, job_id: jobId });
  return `/factor-research?${params.toString()}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (detail && typeof detail.message === "string") throw new Error(detail.message);
    throw new Error(`请求失败（HTTP ${response.status}）`);
  }
  return response.json();
}

let toastTimer;
function showToast(message, error = false) {
  const node = document.getElementById("toast");
  node.textContent = displayValue(message);
  node.style.background = error ? "#a93430" : "#202a3b";
  node.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => node.classList.remove("show"), 2600);
}
