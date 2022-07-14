import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.remote import webelement
from Parser import Parser


class SberOffersParser(Parser):
    def __init__(self) -> None:
        self.url = "https://spasibosberbank.ru/partners"
        self.logger = logging.getLogger(__name__)

    def __expand_all_offers_card(self,
                                 driver: webdriver,
                                 wait: WebDriverWait) -> None:
        while True:
            try:
                element = wait.until(ec.element_to_be_clickable(
                    (By.XPATH,
                     '/html/body/div[1]/div/div/div[1]/div/div[2]/main/div/div[3]/div[2]/button'))
                )
                self.logger.info(f"Element: <{element.text}>")
                driver.execute_script("arguments[0].click();", element)
            except Exception:
                break

    def __links_answer_forming(self,
                               element: webelement) -> tuple[str, str]:
        aux_element = element.find_elements(By.CLASS_NAME, "app-skeleton-wrapper")[0]
        title = aux_element.get_attribute("title")
        subtitle = aux_element.get_attribute("subtitle")
        aux_element = element.find_elements(By.CLASS_NAME, "product-card")[0]
        return title + " - " + subtitle, aux_element.get_attribute("href")

    def get_links(self,
                  driver: webdriver,
                  wait: WebDriverWait) -> list[tuple[str, str]]:
        self.logger.info(f"Getting ${self.url}...")
        driver.get(self.url)
        self.logger.info("Expanding cards...")
        self.__expand_all_offers_card(driver, wait)
        self.logger.info("Getting links...")
        elems = driver.find_elements(By.CLASS_NAME, "partnersProposal__item")
        return [self.__links_answer_forming(elem) for elem in elems]

    def get_data(self,
                 worker: webdriver,
                 wait: WebDriverWait,
                 data: tuple[str, str]) -> dict[str, str]:
        self.logger.info("Getting offer: " + data[0])
        try:
            worker.get(data[1])
            element = wait.until(ec.visibility_of_element_located(
                (By.CLASS_NAME, "proposal-description__description")))
            text = data[0] + " " + re.sub(r"\n+", " ", element.text)
        except:
            text = data[0]
        return {"title": data[0], "uri": data[1], "content": text}
