import logging
import os
from glob import glob

def setup_logging():
    log_directory = 'logger'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Find the highest numbered log file
    log_files = glob(os.path.join(log_directory, 'nn_log*.log'))
    log_numbers = [int(f.split('nn_log')[1].split('.log')[0]) for f in log_files if 'nn_log' in f and '.log' in f and f.split('nn_log')[1].split('.log')[0].isdigit()]
    next_log_number = max(log_numbers) + 1 if log_numbers else 1
    
    # Setup logging with the new log file
    logging.basicConfig(filename=os.path.join(log_directory, f'nn_log{next_log_number}.log'),
                        level=logging.INFO,
                        filemode='w',  # 'w' for overwrite, 'a' for append
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
def change_logging_level(new_level):
    logger = logging.getLogger()
    logger.setLevel(new_level)
