"""按实验版本保存结果；训练期间写暂存目录，成功后发布并保留旧快照。"""

from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import pandas as pd


EXPERIMENT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_id(identity: dict) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_identity(*, input_files, stock_codes, feature_names, preprocessing_version,
                   configuration, implementation_files, runtime_versions, panel_summary):
    """周期不参与标识：同一输入和配置可分批补跑不同周期。"""
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "input_sha256": {name: sha256_file(path) for name, path in sorted(input_files.items())},
        "stock_codes": sorted(set(str(code).zfill(6) for code in stock_codes)),
        "features": list(feature_names),
        "preprocessing_version": preprocessing_version,
        "configuration": dict(configuration),
        "implementation_sha256": {
            name: sha256_file(path) for name, path in sorted(implementation_files.items())
        },
        "runtime_versions": dict(runtime_versions),
        "panel_summary": dict(panel_summary),
    }


class ExperimentRun:
    """隔离单次运行，兼容周期合并，发布失败时回滚当前目录。"""

    def __init__(self, workspace, identity, horizons, *, result_dir=None, model_dir=None,
                 legacy_model_dir=None, identity_refresher=None):
        self.workspace = Path(workspace).resolve()
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.experiment_id = experiment_id(self.identity)
        self.horizons = tuple(sorted(set(int(h) for h in horizons)))
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("实验必须包含有效的正整数预测周期")
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:10]
        self.result_dir = self._checked(result_dir or self.workspace / "results")
        self.model_dir = self._checked(model_dir or self.workspace / "saved_models")
        self.legacy_model_dir = self._checked(legacy_model_dir) if legacy_model_dir else None
        directories = [self.result_dir, self.model_dir]
        if self.legacy_model_dir:
            directories.append(self.legacy_model_dir)
        if any(left.is_relative_to(right) or right.is_relative_to(left)
               for i, left in enumerate(directories) for right in directories[i + 1:]):
            raise ValueError("结果和模型目录必须互不相同，且不能相互嵌套")
        self.staging_dir = self._checked(self.workspace / ".experiment_staging" / self.run_id)
        self.staged_results = self.staging_dir / "results"
        self.staged_models = self.staging_dir / "saved_models"
        self.identity_refresher = identity_refresher
        self.previous = {}
        self.compatible = False
        self.committed = False
        self.history_dir = None
        self._lock = None
        self._result_files = set()
        self._model_files = set()

    def _checked(self, path):
        absolute = Path(os.path.abspath(path))
        resolved = absolute.resolve()
        if resolved == self.workspace or not resolved.is_relative_to(self.workspace):
            raise ValueError(f"输出路径必须位于实验工作目录内部：{resolved}")
        if resolved != absolute:
            raise ValueError(f"输出路径不能通过符号链接或重解析跳转：{absolute}")
        return resolved

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError(f"产物必须使用单层文件名：{name!r}")
        return name

    def _acquire_lock(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        stream = self._checked(self.workspace / ".experiment_run.lock").open("a+b")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("已有训练正在使用当前结果目录，请等待其结束") from error
        self._lock = stream

    def _release_lock(self):
        if self._lock is not None:
            try:
                self._lock.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None

    def _status(self, status, **details):
        self._checked(self.staging_dir / "run_status.json").write_text(
            json.dumps({"status": status, "experiment_id": self.experiment_id,
                        "run_id": self.run_id, "horizons": list(self.horizons), **details},
                       ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def __enter__(self):
        self._acquire_lock()
        try:
            metadata_path = self._checked(self.result_dir / "metadata.json")
            if metadata_path.is_file():
                try:
                    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                    self.previous = previous if isinstance(previous, dict) else {}
                except (ValueError, UnicodeError):
                    self.previous = {}
            self.compatible = (
                self.previous.get("status") == "complete"
                and self.previous.get("experiment_id") == self.experiment_id
                and self.previous.get("experiment_identity") == self.identity
                and isinstance(self.previous.get("artifacts"), dict)
                and all(isinstance(self.previous["artifacts"].get(kind), dict)
                        for kind in ("results", "models"))
            )
            self.staged_results.mkdir(parents=True)
            self.staged_models.mkdir()
            if self.compatible:
                self._copy_previous("results", self.result_dir, self.staged_results, self._result_files)
                self._copy_previous("models", self.model_dir, self.staged_models, self._model_files)
            self._status("running", compatible_previous=self.compatible)
            return self
        except BaseException:
            self._release_lock()
            raise

    def _copy_previous(self, kind, source_dir, target_dir, tracked):
        for name, expected_sha256 in self.previous["artifacts"][kind].items():
            self._name(name)
            source = self._checked(source_dir / name)
            if not source.is_file() or sha256_file(source) != expected_sha256:
                raise RuntimeError(f"当前实验产物缺失或已被修改，不能自动合并：{source}")
            shutil.copy2(source, self._checked(target_dir / name))
            tracked.add(name)

    def write_csv(self, frame, filename):
        self._name(filename)
        if not filename.endswith(".csv"):
            raise ValueError("结果表必须保存为CSV")
        path = self._checked(self.staged_results / filename)
        old = pd.read_csv(path, dtype={"code": str}) if path.is_file() else None
        if old is not None:
            if "horizon" not in old or "experiment_id" not in old:
                raise ValueError(f"旧结果缺少周期或实验标识：{filename}")
            if not old.experiment_id.eq(self.experiment_id).all():
                raise ValueError(f"旧结果包含不同实验，拒绝合并：{filename}")
            old = old[~old.horizon.isin(self.horizons)].copy()
        frame = frame.copy()
        if "horizon" not in frame:
            if not frame.empty:
                raise ValueError(f"结果缺少预测周期：{filename}")
            frame = pd.DataFrame(columns=old.columns if old is not None else ["horizon"])
        if not frame.empty and not frame.horizon.isin(self.horizons).all():
            raise ValueError(f"结果包含本次未运行的周期：{filename}")
        frame["experiment_id"] = self.experiment_id
        frame["run_id"] = self.run_id
        if old is not None and not old.empty:
            frame = old if frame.empty else pd.concat([old, frame], ignore_index=True, sort=False)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        self._result_files.add(filename)

    def model_path(self, filename):
        self._name(filename)
        self._model_files.add(filename)
        return self._checked(self.staged_models / filename)

    def publish(self, metadata):
        if self.committed:
            raise RuntimeError("本次实验已经发布")
        if self.identity_refresher is not None and self.identity_refresher() != self.identity:
            raise RuntimeError("训练期间输入数据、代码或配置发生变化，结果保留在暂存目录，不发布")
        if set(metadata.get("horizons", ())) != set(self.horizons):
            raise ValueError("运行元数据的周期与本次请求不一致")
        previous = self.previous if self.compatible else {}
        current = dict(metadata)
        record = {key: current[key] for key in (
            "run_started_at", "run_finished_at", "elapsed_seconds", "command", "device",
            "gpu_name", "thread_limit") if key in current}
        record.update(run_id=self.run_id, horizons=list(self.horizons))
        current.update({
            "status": "complete", "experiment_id": self.experiment_id,
            "experiment_identity": self.identity, "run_id": self.run_id,
            "horizons": sorted(set(previous.get("horizons", ())) | set(self.horizons)),
            "model_inputs": {**previous.get("model_inputs", {}), **current.get("model_inputs", {})},
            "run_history": [*previous.get("run_history", ()), record],
            "run_started_at": previous.get("run_started_at", current.get("run_started_at")),
            "elapsed_seconds": round(float(previous.get("elapsed_seconds", 0))
                                     + float(current.get("elapsed_seconds", 0)), 2),
            "stock_codes": self.identity["stock_codes"],
            "stock_count": len(self.identity["stock_codes"]),
            "artifacts": {
                "results": {name: sha256_file(self._checked(self.staged_results / name))
                            for name in sorted(self._result_files)},
                "models": {name: sha256_file(self._checked(self.staged_models / name))
                           for name in sorted(self._model_files)},
            },
        })
        encoded = json.dumps(current, ensure_ascii=False, indent=2, default=str)
        for name in ("metadata.json", "run_metadata.json"):
            self._checked(self.staged_results / name).write_text(encoded, encoding="utf-8")
        self._commit()
        return current

    def _move(self, source, target):
        # 每次目录移动前验证解析后的绝对路径，避免越出实验工作目录。
        os.replace(self._checked(source), self._checked(target))

    def _commit(self):
        old_id = self.previous.get("experiment_id", "")
        label = old_id[:16] if isinstance(old_id, str) and len(old_id) == 64 and all(
            char in "0123456789abcdef" for char in old_id
        ) else "legacy"
        history = self._checked(self.workspace / "experiment_history" / label / self.run_id)
        candidates = [(self.model_dir, history / "saved_models"),
                      (self.result_dir, history / "results")]
        if self.legacy_model_dir and label == "legacy":
            candidates.append((self.legacy_model_dir, history / "legacy_models"))
        candidates = [(self._checked(source), self._checked(target))
                      for source, target in candidates if source.exists()]
        if candidates:
            history.mkdir(parents=True)
        moved, published = [], []
        self._status("publishing")
        try:
            for source, target in candidates:
                self._move(source, target)
                moved.append((source, target))
            for source, target in ((self.staged_models, self.model_dir),
                                   (self.staged_results, self.result_dir)):
                target.parent.mkdir(parents=True, exist_ok=True)
                self._move(source, target)
                published.append((source, target))
        except BaseException as error:
            rollback_errors = []
            for source, target in reversed(published):
                try:
                    self._move(target, source)
                except (OSError, ValueError) as rollback_error:
                    rollback_errors.append(str(rollback_error))
            for source, target in reversed(moved):
                try:
                    self._move(target, source)
                except (OSError, ValueError) as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if rollback_errors:
                raise RuntimeError(f"发布失败且回滚未完成，请检查历史目录 {history}：{rollback_errors}") from error
            raise
        self.committed = True
        self.history_dir = history if candidates else None
        with suppress(OSError):
            self._status("complete", history_dir=str(self.history_dir) if self.history_dir else None)

    def __exit__(self, error_type, error, traceback):
        try:
            if not self.committed:
                with suppress(OSError):
                    self._status("failed", error=str(error) if error else "运行未发布")
        finally:
            self._release_lock()
        if error_type is None and not self.committed:
            raise RuntimeError("运行没有成功发布，当前结果未更新")
        return False
