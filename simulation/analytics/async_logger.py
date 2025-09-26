# simulation/analytics/async_logger.py
import threading, queue, json, os
from typing import Dict, Any

class AsyncNDJSONLogger:
    def __init__(self, filepath: str, flush_every: int = 1000):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q = queue.Queue(maxsize=100000)
        self.stop_flag = False
        self.flush_every = flush_every
        self._writer = threading.Thread(target=self._run, daemon=True)
        self._writer.start()

    def log(self, obj: Dict[str, Any]):
        try:
            self.q.put_nowait(json.dumps(obj, separators=(",", ":")) + "\n")
        except queue.Full:
            pass  # drop under backpressure for speed; or block if you prefer

    def _run(self):
        f = open(self.filepath, "a", buffering=1)
        n = 0
        while not self.stop_flag or not self.q.empty():
            try:
                line = self.q.get(timeout=0.5)
                f.write(line)
                n += 1
                if n >= self.flush_every:
                    f.flush()
                    n = 0
            except queue.Empty:
                continue
        f.flush()
        f.close()

    def close(self):
        self.stop_flag = True
        self._writer.join(timeout=5.0)