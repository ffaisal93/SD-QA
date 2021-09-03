import sys
import logging

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
# fh = logging.FileHandler('my_log_info.log')
sh = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
# fh.setFormatter(formatter)
sh.setFormatter(formatter)
# logger.addHandler(fh)
logger.addHandler(sh)

# def hello_logger():
#     logger.info("Hello info")
#     logger.critical("Hello critical")
#     logger.warning("Hello warning")
#     logger.debug("Hello debug")

if __name__ == "__main__":
    logger.info("Hello info")