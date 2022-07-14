import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from Parser import Parser


class SberImpressionsParser(Parser):
    def __init__(self) -> None:
        self.url = "https://spasibosberbank.ru/events_category/impressions"
        self.logger = logging.getLogger(__name__)

    def get_links(self,
                  driver: webdriver,
                  wait: WebDriverWait) -> list[tuple[str, str]]:
        self.logger.info(f"Getting ${self.url}...")
        driver.get(self.url)

        self.logger.info("Getting links...")
        elems = driver.find_elements(By.XPATH, "//*[@class='gift_card container-item']")
        self.logger.info(elems)
        return [(elem.text.replace("\n", " - "), elem.get_attribute("href")) for elem in elems]

    def get_data(self,
                 worker: webdriver,
                 wait: WebDriverWait,
                 data: tuple[str, str]) -> dict[str, str]:
        self.logger.info("Getting offer: " + data[0])
        try:
            worker.get(data[1])
            element = wait.until(ec.visibility_of_element_located(
                (By.XPATH, "//*[@class='movie-description__text']")))
            text = data[0] + " " + re.sub(r"\n+", " ", element.text)
        except:
            text = data[0]
        return {"title": data[0], "uri": data[1], "content": text}
