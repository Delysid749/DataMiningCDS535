import os.path

from src.common.directory_exists import ensure_directory_exists
def log_output(message, log_file_path, log_only_important=False):
    """将信息输出到控制台，并根据需要选择性写入日志文件"""
    print(message)
    log_dir = os.path.dirname(log_file_path)
    ensure_directory_exists(log_dir)
    if log_only_important:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{message}\n")