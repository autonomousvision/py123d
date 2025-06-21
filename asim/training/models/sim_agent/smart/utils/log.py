import logging
import os
import time


class Logging:
    def make_log_dir(self, dirname="logs"):
        now_dir = os.path.dirname(__file__)
        path = os.path.join(now_dir, dirname)
        path = os.path.normpath(path)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_log_filename(self):
        filename = "{}.log".format(time.strftime("%Y-%m-%d", time.localtime()))
        filename = os.path.join(self.make_log_dir(), filename)
        filename = os.path.normpath(filename)
        return filename

    def log(self, level="DEBUG", name="simagent"):
        logger = logging.getLogger(name)
        level = getattr(logging, level)
        logger.setLevel(level)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = logging.FileHandler(filename=self.get_log_filename(), mode="a", encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s")
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger

    def add_log(self, logger, level="DEBUG"):
        level = getattr(logging, level)
        logger.setLevel(level)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = logging.FileHandler(filename=self.get_log_filename(), mode="a", encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s")
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger
