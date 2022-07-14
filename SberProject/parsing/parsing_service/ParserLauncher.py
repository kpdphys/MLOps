import logging
import os
from typing import Type
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from multiprocessing import Queue
from WorkersPool import WorkersPool
from Parser import Parser
from WeaviateClient import WeaviateClient


class ParserLauncher:
    def __init__(self, parser_classname: Type[Parser]) -> None:
        self.parser = None
        self.logger = logging.getLogger(__name__)
        self.parser_classname = parser_classname
        self.options = self.__form_driver_options()
        self.time_to_wait = 3

        self.cpu_cores = os.getenv("CPU_CORES", default="4")
        try:
            self.cpu_cores = int(self.cpu_cores)
        except:
            self.cpu_cores = 4
        self.logger.info(f"Value of env variable 'CPU_CORES': {self.cpu_cores}")

    def __form_driver_options(self) -> Options:
        options = webdriver.FirefoxOptions()
        options.headless = True
        options.binary_location = r'/opt/firefox/firefox'
        return options

    def __get_links_from_parser(self) -> list[tuple[str, str]]:
        links = []
        driver = None
        try:
            self.logger.info("Starting webdriver...")
            driver = webdriver.Firefox(service=Service('/usr/bin/geckodriver'),
                                       options=self.options)
            wait = WebDriverWait(driver, self.time_to_wait)
            self.logger.info("Webdriver has started.")
            links = links + self.parser.get_links(driver, wait)
        except:
            self.logger.warning("Links can't been obtained!")
        finally:
            links.append(('', ''))
            if driver is not None:
                self.logger.info("Driver is finishing...")
                driver.quit()
            return links

    def __selenium_queue_listener(self,
                                  worker: webdriver,
                                  wait: WebDriverWait,
                                  input_queue: Queue) -> None:

        weaviate_client = WeaviateClient()
        while True:
            current_input = input_queue.get()
            if current_input == ('', ''):
                self.logger.info("STOP encountered, killing worker thread")
                input_queue.put(current_input)
                break
            else:
                self.logger.info(f"Got the item <{current_input}> on the input queue")
                output = self.parser.get_data(worker, wait, current_input)
                weaviate_client.post_data(output)
        return

    def parse(self) -> None:
        input_queue = Queue()
        self.parser = self.parser_classname()

        self.logger.info("Adding links to queue")
        for link in self.__get_links_from_parser():
            input_queue.put(link)

        workers_pool = WorkersPool(self.options,
                                   self.time_to_wait,
                                   self.cpu_cores,
                                   input_queue)
        workers_pool.run(self.__selenium_queue_listener)
