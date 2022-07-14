import logging
import time
from ParserLauncher import ParserLauncher
from SberPartnersParser import SberPartnersParser
from SberOffersParser import SberOffersParser
from SberKuponesParser import SberKuponesParser
from SberImpressionsParser import SberImpressionsParser


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    try:
        logger.info("Parsing has started.")
        logger.info("Parsing partners...")
        ParserLauncher(SberPartnersParser).parse()

        logger.info("Parsing offers...")
        ParserLauncher(SberOffersParser).parse()

        logger.info("Parsing impressions...")
        ParserLauncher(SberImpressionsParser).parse()

        logger.info("Parsing kupones...")
        ParserLauncher(SberKuponesParser).parse()

        logger.info("Parsing has finished. Exiting.")
    except Exception as e:
        logger.error("GLOBAL ERROR!")
        logger.error(e)
        time.sleep(60)


if __name__ == "__main__":
    main()
