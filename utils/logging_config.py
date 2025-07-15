import logging
import sys

def setup_logger(level=logging.INFO):
    """
    Set up the root logger.

    Args:
        level (int): The logging level.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove all existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler to write to the console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

if __name__ == '__main__':
    # Example usage:
    setup_logger(logging.DEBUG)
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")