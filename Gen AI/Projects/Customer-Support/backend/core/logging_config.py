import logging
import sys
def configure_logging()->None:
    logging.basicConfig(
        level = logging.INFO,
        format = (
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
            ),
            handlers = [logging.StreamHandler(sys.stdout)]
            #logging.StreamHandler send logging output to streams like sys.stdout, sys.stderr, or any file-like object. It is responsible for directing your log messages to your terminal console or interactive session.
    )