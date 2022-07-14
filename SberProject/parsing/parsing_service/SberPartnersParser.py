import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from Parser import Parser


class SberPartnersParser(Parser):
    def __init__(self) -> None:
        self.url = "https://spasibosberbank.ru/partners"
        self.logger = logging.getLogger(__name__)

    def __expand_all_partners_card(self,
                                   driver: webdriver,
                                   wait: WebDriverWait) -> None:
        while True:
            try:
                element = wait.until(ec.element_to_be_clickable(
                    (By.XPATH,
                     '/html/body/div[1]/div/div/div[1]/div/div[2]/main/div/div[2]/div[2]/button'))
                )
                self.logger.info(f"Element: <{element.text}>")
                driver.execute_script("arguments[0].click();", element)
            except Exception:
                break

    def get_links(self,
                  driver: webdriver,
                  wait: WebDriverWait) -> list[tuple[str, str]]:
        self.logger.info(f"Getting ${self.url}...")
        driver.get(self.url)
        self.logger.info("Expanding cards...")
        self.__expand_all_partners_card(driver, wait)
        self.logger.info("Getting links...")
        elems = driver.find_elements(By.CLASS_NAME, "card-partner__title")
        return [(elem.text, elem.find_element(By.XPATH, '..') \
                 .get_attribute('href')) for elem in elems]

    def __descr_absence_execution(self,
                                  title: str,
                                  driver: webdriver,
                                  wait: WebDriverWait) -> str:
        self.logger.info(f"Element <{title}> has not text decription. Trying to parse partner\'s site...")
        try:
            element = wait.until(ec.visibility_of_element_located(
                (By.XPATH, "/html/body/div[1]/div/div/div[1]/div/div[2]/main/div[1]/div[2]/div[1]/div[2]/div/div/a")))
            external_url = element.get_attribute('href')
            text = self.__parse_site_with_no_descr(driver, wait, external_url).replace("\n", " ")
        except:
            text = title
        return text

    def __parse_site_with_no_descr(self,
                                   driver: webdriver,
                                   wait: WebDriverWait,
                                   url: str) -> str:
        text = ""
        try:
            driver.get(url)
            element = wait.until(ec.visibility_of_element_located(
                (By.XPATH, "/html/body")))
            text = element.text
        except Exception as e:
            self.logger.warning(e)
        return text

    def get_data(self,
                 worker: webdriver,
                 wait: WebDriverWait,
                 data: tuple[str, str]) -> dict[str, str]:
        self.logger.info("Getting partner: " + data[0])
        try:
            worker.get(data[1])
            element = wait.until(ec.visibility_of_element_located(
                (By.CLASS_NAME, "contants_description")))
            text = re.sub(r"\n+", " ", element.text)
        except:
            text = self.__descr_absence_execution(data[0], worker, wait)
        return {"title": data[0], "uri": data[1], "content": text}
