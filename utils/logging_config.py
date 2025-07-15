import logging
import sys
import colorlog

def setup_logger(level="INFO"):
    """
    Set up the root logger with colored output.

    Args:
        level (str or int): The logging level.
    """
    # Get the root logger
    logger = logging.getLogger()

    # If level is a string, convert it to the corresponding logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)

    # Remove all existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler to write to the console (stdout)
    console_handler = colorlog.StreamHandler(sys.stdout)
    
    # Create a colored formatter and set it for the handler
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

if __name__ == '__main__':
    # Example usage:
    setup_logger("DEBUG")
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")