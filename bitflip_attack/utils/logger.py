import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name=None, level=logging.INFO, log_file=None, console=True):
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    

    logger.handlers.clear()
    

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_attack_logger(attack_name, log_dir='logs', level=logging.INFO):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{attack_name}_{timestamp}.log"
    


    return setup_logger(
        name=attack_name,
        level=level,
        log_file=log_file,
        console=True
    )



def debug_mode_logger(name=None):
    return setup_logger(name, level=logging.DEBUG)


def quiet_mode_logger(name=None):
    return setup_logger(name, level=logging.WARNING)


def get_bit_manipulation_logger():
    return logging.getLogger('bitflip_attack.bit_manipulation')




def get_optimization_logger():
    return logging.getLogger('bitflip_attack.optimization')




def get_evaluation_logger():
    return logging.getLogger('bitflip_attack.evaluation')

