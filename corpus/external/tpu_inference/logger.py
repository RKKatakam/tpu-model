"""Stub of tpu_inference.logger (the real one wraps vllm's logger)."""

import logging


class _Logger(logging.LoggerAdapter):
    _seen = set()

    def warning_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self._seen.add(msg)
            self.warning(msg, *args, **kwargs)

    def info_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self._seen.add(msg)
            self.info(msg, *args, **kwargs)


def init_logger(name):
    return _Logger(logging.getLogger(name), {})
