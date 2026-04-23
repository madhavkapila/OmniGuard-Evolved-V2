from __future__ import annotations

import json
import os
import signal
import threading
import time
from typing import Any

import redis
from datasets import load_dataset

from server.payloads import BENIGN_DATASET_ID, MALICIOUS_DATASET_ID, ORACLE_DATASET_ID


class RedisStreamingDataWorker:
    def __init__(
        self,
        redis_url: str,
        max_stream_len: int = 50000,
    ) -> None:
        self.redis_url = redis_url
        self.max_stream_len = max_stream_len
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def _extract_text(self, row: dict[str, Any]) -> str:
        for key in ("text", "content", "payload", "prompt", "instruction", "message"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in row.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _stream_dataset(self, dataset_id: str, stream_key: str, is_malicious: bool) -> None:
        while not self.stop_event.is_set():
            try:
                stream = load_dataset(dataset_id, split="train", streaming=True)
                for row in stream:
                    if self.stop_event.is_set():
                        return
                    text = self._extract_text(row)
                    if not text:
                        continue
                    payload = {
                        "dataset_id": dataset_id,
                        "is_malicious": is_malicious,
                        "payload_raw": text,
                        "created_at": time.time(),
                    }
                    self.client.xadd(
                        stream_key,
                        {"data": json.dumps(payload)},
                        maxlen=self.max_stream_len,
                        approximate=True,
                    )
            except Exception:
                time.sleep(2.0)

    def _stream_oracle(self) -> None:
        stream_key = "omniguard:oracle"
        while not self.stop_event.is_set():
            try:
                stream = load_dataset(ORACLE_DATASET_ID, split="train", streaming=True)
                for row in stream:
                    if self.stop_event.is_set():
                        return
                    text = self._extract_text(row)
                    if not text:
                        continue
                    payload = {
                        "dataset_id": ORACLE_DATASET_ID,
                        "payload_raw": text,
                        "created_at": time.time(),
                    }
                    self.client.xadd(
                        stream_key,
                        {"data": json.dumps(payload)},
                        maxlen=self.max_stream_len,
                        approximate=True,
                    )
            except Exception:
                time.sleep(2.0)

    def run(self) -> None:
        workers = [
            threading.Thread(
                target=self._stream_dataset,
                args=(BENIGN_DATASET_ID, "omniguard:benign", False),
                daemon=True,
                name="stream-benign",
            ),
            threading.Thread(
                target=self._stream_dataset,
                args=(MALICIOUS_DATASET_ID, "omniguard:malicious", True),
                daemon=True,
                name="stream-malicious",
            ),
            threading.Thread(
                target=self._stream_oracle,
                daemon=True,
                name="stream-oracle",
            ),
        ]
        self.threads.extend(workers)
        for thread in self.threads:
            thread.start()

        while not self.stop_event.is_set():
            time.sleep(1.0)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.0)


def main() -> None:
    redis_url = os.getenv("OMNIGUARD_REDIS_URL", "redis://redis_cache:6379/0")
    max_stream_len = int(os.getenv("OMNIGUARD_REDIS_STREAM_MAXLEN", "50000"))
    worker = RedisStreamingDataWorker(redis_url=redis_url, max_stream_len=max_stream_len)

    def _handle_signal(signum, frame) -> None:
        del signum, frame
        worker.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    worker.run()


if __name__ == "__main__":
    main()
