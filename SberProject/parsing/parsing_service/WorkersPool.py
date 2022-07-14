import logging
from typing import Callable, Any
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from multiprocessing import Queue
from threading import Thread


class WorkersPool:
    def __init__(self,
                 options: Options,
                 time_to_wait: int,
                 cpu_cores: int,
                 input_queue: Queue) -> None:
        self.workers = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.options = options
        self.time_to_wait = time_to_wait
        self.input_queue = input_queue
        self.worker_queue = Queue()

        num_threads = min(cpu_cores, self.input_queue.qsize())
        self.worker_ids = list(range(num_threads))
        for worker_id in self.worker_ids:
            self.worker_queue.put(worker_id)

    def __func_handler(self,
                       func: Callable[[webdriver,
                                       WebDriverWait,
                                       Queue], None]) -> Any:
        def func_wrapper(input_queue: Queue):
            worker_id = self.worker_queue.get()
            worker = self.workers[worker_id]
            wait = WebDriverWait(worker, self.time_to_wait)
            try:
                func(worker, wait, input_queue)
            except Exception as e:
                self.logger.warning(e)
            finally:
                self.worker_queue.put(worker_id)

        return func_wrapper

    def run(self,
            func: Callable[[webdriver,
                            WebDriverWait,
                            Queue], None]):
        try:
            self.logger.info("Starting drivers...")
            self.workers = {i: webdriver.Firefox(service=Service('/usr/bin/geckodriver'),
                                                 options=self.options
                                                ) for i in self.worker_ids}

            self.logger.info("Starting selenium background processes")
            processes = [Thread(target=self.__func_handler(func),
                                args=(self.input_queue,)
                                ) for _ in self.worker_ids]

            self.logger.info("Starting Queue threads")
            for process in processes:
                process.daemon = True
                process.start()

            self.logger.info("Waiting for Queue listener threads to complete")
            for process in processes:
                process.join()

        finally:
            self.logger.info("Tearing down web workers")
            for worker in self.workers.values():
                worker.quit()
