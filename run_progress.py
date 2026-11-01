"""控制台阶段提示与耗时心跳；只记录进度，不修改模型或实验数据。"""

from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from time import perf_counter


HEARTBEAT_SECONDS = 30.0
_PRINT_LOCK = Lock()
_SHANGHAI = timezone(timedelta(hours=8))


def duration_text(seconds):
    seconds = max(0., float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}分{remainder:02d}秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes:02d}分{remainder:02d}秒"


def log_progress(message):
    stamp = datetime.now(_SHANGHAI).strftime("%H:%M:%S")
    with _PRINT_LOCK:
        print(f"[{stamp}] {message}", flush=True)


class progress_stage:
    """阶段开始/结束立即输出；长步骤定期提示仍在等待该步骤返回。"""

    def __init__(self, label, heartbeat_seconds=None):
        self.label = label
        self.heartbeat_seconds = HEARTBEAT_SECONDS if heartbeat_seconds is None else heartbeat_seconds
        if self.heartbeat_seconds <= 0:
            raise ValueError("进度提示间隔必须大于0")
        self.detail = ""
        self._stop = Event()
        self._thread = None

    def __enter__(self):
        self._started = perf_counter()
        log_progress(f"{self.label} | 开始")
        self._thread = Thread(target=self._heartbeat, name="fintech-progress", daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(self.heartbeat_seconds):
            elapsed = duration_text(perf_counter() - self._started)
            log_progress(f"{self.label} | 仍在等待本步骤返回，已用{elapsed}")

    def __exit__(self, error_type, error, traceback):
        self._stop.set()
        self._thread.join()
        elapsed = duration_text(perf_counter() - self._started)
        if error_type is None:
            detail = f" | {self.detail}" if self.detail else ""
            log_progress(f"{self.label} | 完成，耗时{elapsed}{detail}")
        else:
            status = "已中断" if issubclass(error_type, KeyboardInterrupt) else "失败"
            log_progress(f"{self.label} | {status}，耗时{elapsed} | {error_type.__name__}: {error}")
        return False
