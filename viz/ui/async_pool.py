"""Asynchronous worker pool for managing client workers in the UI.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import concurrent.futures as futures
import queue
import threading

import ui.utils as ui_utils

TIMEOUT = 60.0

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()


# ------------------------------------------------------------------------------------------------------------
# Worker pool for managing client workers

class MultiThreadWorkerPool:
    def __init__(self, num_workers: int):
        self.task_queue = queue.PriorityQueue()  # priority queue using running task index, so that re-added tasks execute immediately
        self._running_task_idx = 0
        self._task_idx_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self.threads = [threading.Thread(target=self.worker_func, daemon=True) for _ in range(num_workers)]
        for t in self.threads:
            t.start()

    def num_active(self):
        return len(self.threads)

    def submit(self, func: callable, *args, **kwargs) -> futures.Future:
        """
        Add callable, along with args and kwargs, to worker pool.
        Callable will be called with allocated worker as the first parameter.
        Callable might be re-added if worker goes offline, should have no side effects.
        """
        with self._task_idx_lock:
            task_idx = self._running_task_idx
            self._running_task_idx += 1
        f = futures.Future()
        cancel_event = self._get_cancel_event()
        self.task_queue.put((task_idx, f, func, args, kwargs, cancel_event))
        return f

    def _get_cancel_event(self):
        with self._cancel_lock:
            return self._cancel_event

    def _clear_queue(self, q: queue.Queue):
        try:
            while True:
                _ = q.get(block=False)
                q.task_done()
        except queue.Empty:
            pass

    def abort_running(self):
        """Abort running tasks (early out on remote)"""
        with self._cancel_lock:
            self._cancel_event.set()
            self._cancel_event = threading.Event()

    def clear_tasks(self, block=True):
        """Abort running and clear pending tasks."""
        self.abort_running()  # start early-out process for in-flight tasks
        self._clear_queue(self.task_queue)  # pop pending (not in flight)
        if block:
            self.task_queue.join()  # wait for early-out tasks to finish

    def worker_func(self):
        tid = threading.current_thread().ident
        while True:
            task_idx, future, func, args, kwargs, cancel_event = self.task_queue.get()  # blocks indefinitely

            # Don't re-start already running future
            if future.running():
                logger.info('Detected running future (previous server went offline?)')
            else:
                if not future.set_running_or_notify_cancel():
                    self.task_queue.task_done()
                    continue

            try:
                ret = func(*args, cancel_event=cancel_event, **kwargs)
                future.set_result(ret)
            except BaseException as exc:
                logger.error(exc)
                future.set_exception(exc)

            self.task_queue.task_done()

# ------------------------------------------------------------------------------------------------------------
