import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from Parser import Parser


class SberKuponesParser(Parser):
    def __init__(self) -> None:
        self.url = "https://spasibosberbank.ru/coupons/list"
        self.logger = logging.getLogger(__name__)

    def __unpress_active_category(self,
                                  driver: webdriver,
                                  wait: WebDriverWait) -> None:
        while True:
            try:
                element = wait.until(ec.element_to_be_clickable((By.XPATH,
                                                                 '//*[@class="category__name-wrapper active"]')))
                self.logger.info(f"Element: <{element.text}> is active!")
                driver.execute_script("arguments[0].click();", element)
            except:
                break

    def __expand_all_partners_card(self,
                                   driver: webdriver,
                                   wait: WebDriverWait) -> None:
        while True:
            try:
                element = wait.until(ec.element_to_be_clickable(
                    (By.XPATH,
                     '/html/body/div[1]/div/div/div[1]/div/div[2]/main/button'
                     )
                ))
                self.logger.info(f"Element: <{element.text}>")
                driver.execute_script("arguments[0].click();", element)
            except Exception as e:
                break

    def get_links(self,
                  driver: webdriver,
                  wait: WebDriverWait) -> list[tuple[str, str]]:
        self.logger.info(f"Getting ${self.url}...")
        driver.get(self.url)

        # Some buttons can be toggled after loading the page...
        # They should be toggled back
        #self.logger.info("Untoggling active buttons...")
        #self.__unpress_active_category(driver, wait)

        #self.logger.info("Expanding cards...")
        #self.__expand_all_partners_card(driver, wait)

        self.logger.info("Getting links...")
        try:
            elems = driver.find_elements(by=By.CLASS_NAME, value="coupon-card")
            self.logger.info(elems)
            return [(elem.text.replace('\n', ' - ') + " бонусов", elem.get_attribute('href')) for elem in elems]
        except Exception as e:
            self.logger.warning(e)
            return []

    def get_data(self,
                 worker: webdriver,
                 wait: WebDriverWait,
                 data: tuple[str, str]) -> dict[str, str]:
        self.logger.info("Getting partner: " + data[0])
        try:
            worker.get(data[1])
            element = wait.until(ec.visibility_of_element_located(
                (By.CLASS_NAME, "partner-info")))
            element = element.find_elements(by=By.CLASS_NAME, value="v-html")[0]
            text = re.sub(r"\n+", " ", element.text)
        except:
            text = data[0]

        return {"title": data[0], "uri": data[1], "content": text}
