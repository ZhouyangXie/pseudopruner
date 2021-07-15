import logging
from time import perf_counter
import torch


class CuTimer:
    def __init__(self, device):
        self.device = device
        self.reset(True)

    def checkpoint(self, msg=''):
        if self.enabled:
            torch.cuda.synchronize(self.device)
            if self.start_time:
                self.events.append(
                    (msg, perf_counter() - self.start_time)
                )

            self.start_time = perf_counter()

    def reset(self, enabled=True):
        self.enabled = enabled
        self.start_time = None
        self.events = []

    def log_event(self):
        for msg, elapsed in self.events:
            logging.debug(f'event {msg}: {elapsed:.6f}s')
