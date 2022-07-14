from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


class Parser(ABC):

    @abstractmethod
    def get_links(self,
                  driver: webdriver,
                  wait: WebDriverWait) -> list[tuple[str, str]]:
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_data(self,
                 worker: webdriver,
                 wait: WebDriverWait,
                 data: tuple[str, str]) -> dict[str, str]:
        raise NotImplementedError("Subclasses should implement this!")