"""RunStore：runs/<id>/ 落盘 round-trip + 孤儿扫描（纯本地，零网络）。"""
import json

import pytest

from dra.events import EVENT_SCHEMA_VERSION
from dra.runstore import (
    IncompatibleEventSchemaError,
    PlanAttemptStore,
    RunStore,
)


def test_create_append_finalize_roundtrip(tmp_path):
    s = RunStore(tmp_path)
    rid = s.create_run("测试问题", {"clarified_query": "q", "initial_tasks": []}, {})
    assert (tmp_path / rid / "meta.json").exists()
    assert s.get_meta(rid)["status"] == "running"
    assert s.get_meta(rid)["event_schema_version"] == EVENT_SCHEMA_VERSION == 3
    s.append_event(rid, {"type": "scope", "ts": 1.0, "seq": 0})
    s.append_event(rid, {"type": "end", "ts": 2.0, "seq": 1})
    lines = s.read_events_text(rid).strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["seq"] == 0
    (tmp_path / rid / "report.md").write_text("# 报告", encoding="utf-8")
    assert s.read_report(rid) == "# 报告"
    s.finalize(rid, "done", {"wall_s": 1.0})
    m = s.get_meta(rid)
    assert m["status"] == "done" and m["stats"]["wall_s"] == 1.0 and m["finished_at"] is not None


def test_list_runs_sorted_desc(tmp_path):
    s = RunStore(tmp_path)
    a = s.create_run("先", None, {})
    b = s.create_run("后", None, {})
    ids = [m["run_id"] for m in s.list_runs()]
    assert ids.index(b) < ids.index(a)


def test_counted_run_list_reads_only_selected_metadata_and_preserves_replay_status(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    older = store.create_run("较早的旧协议研究", None, {"mode": "simple"})
    newer = store.create_run("较新的研究", None, {"mode": "simple"})
    excluded = store.create_run("别人的研究", None, {"mode": "simple"})
    for run_id, started_at in ((older, 10), (newer, 20), (excluded, 30)):
        meta = store.get_meta(run_id)
        meta["started_at"] = started_at
        if run_id == older:
            meta.pop("event_schema_version")
        (tmp_path / run_id / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    selected = {older, newer}
    reads = []
    real_get_meta = store.get_meta

    def get_meta(run_id):
        reads.append(run_id)
        return real_get_meta(run_id)

    monkeypatch.setattr(store, "get_meta", get_meta)
    metas = store.list_runs(counted=selected.__contains__)
    assert len(reads) == 2 and set(reads) == selected
    assert [meta["run_id"] for meta in metas] == [newer, older]
    assert metas[0]["replay_compatible"] is True and metas[0]["replay_error"] is None
    assert metas[1]["replay_compatible"] is False and "不可回放" in metas[1]["replay_error"]


def test_empty_counted_selection_does_not_read_metadata(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    run_id = store.create_run("不在授权范围的研究", None, {"mode": "simple"})
    (tmp_path / run_id / "meta.json").write_text("{broken", encoding="utf-8")
    reads = []
    real_get_meta = store.get_meta

    def get_meta(run_id):
        reads.append(run_id)
        return real_get_meta(run_id)

    monkeypatch.setattr(store, "get_meta", get_meta)
    counted = set().__contains__
    assert store.list_runs(counted=counted) == []
    assert store.conversations(counted=counted) == []
    assert store.conversation_turns(run_id, counted=counted) == []
    assert reads == []


def test_selected_corrupt_metadata_is_still_an_explicit_error(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run("自己的损坏记录", None, {"mode": "simple"})
    (tmp_path / run_id / "meta.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.conversations(counted={run_id}.__contains__)


def test_counted_list_checks_both_directory_and_metadata_identity(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    selected = store.create_run("授权目录", None, {"mode": "simple"})
    excluded = store.create_run("未授权目录", None, {"mode": "simple"})
    # 目录与 metadata 互换身份：既不能读未授权目录，也不能返回授权目录里的未授权身份。
    for directory, claimed_id in ((selected, excluded), (excluded, selected)):
        meta = store.get_meta(directory)
        meta["run_id"] = claimed_id
        (tmp_path / directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    reads = []
    real_get_meta = store.get_meta

    def get_meta(run_id):
        reads.append(run_id)
        return real_get_meta(run_id)

    monkeypatch.setattr(store, "get_meta", get_meta)
    assert store.list_runs(counted={selected}.__contains__) == []
    assert reads == [selected]


def test_mark_orphans(tmp_path):
    s = RunStore(tmp_path)
    rid = s.create_run("孤儿", None, {})
    assert s.mark_orphans() == 1
    m = s.get_meta(rid)
    assert m["status"] == "failed" and m["fail_reason"] == "server restart"


def test_plan_attempt_store_roundtrip(tmp_path):
    s = PlanAttemptStore(tmp_path)
    pid = s.create(query="芯片对比", kind="plan", config_summary={"planner": "x@y"})
    meta_path = tmp_path / pid / "meta.json"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["status"] == "running"
    s.append_event(pid, {"type": "llm_call", "step": "research_brief", "seq": 0})
    s.append_event(pid, {"type": "step_done", "step": "research_brief", "seq": 1})
    text = (tmp_path / pid / "events.jsonl").read_text(encoding="utf-8")
    assert text is not None and "research_brief" in text
    s.finalize(
        pid, "ok",
        plan={"kind": "plan", "brief": {"goal": "芯片对比", "requirements": []}},
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "ok"
    assert meta["error"] is None
    assert meta["plan"]["brief"]["goal"] == "芯片对比"
    assert "trace" not in meta["plan"]


def test_plan_attempt_store_failed(tmp_path):
    s = PlanAttemptStore(tmp_path)
    pid = s.create(query="q", kind="plan", config_summary={})
    s.finalize(pid, "failed", error="planner 输出缺少有效 initial_tasks")
    meta = json.loads((tmp_path / pid / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert "initial_tasks" in meta["error"]


def test_old_run_event_schema_is_explicitly_incompatible(tmp_path):
    s = RunStore(tmp_path)
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": "old-run", "status": "done", "started_at": 1}),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text('{"type":"scope"}\n', encoding="utf-8")

    [meta] = s.list_runs()
    assert meta["replay_compatible"] is False
    assert "不可回放" in meta["replay_error"]
    with pytest.raises(IncompatibleEventSchemaError, match="不可回放"):
        s.read_events_text("old-run")
