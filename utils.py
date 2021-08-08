import logging

'''record configurations'''

def setup_logging(log_file='log.txt', filemode='w'):
    """
    Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%m/%d %I:%M:%S %p')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


