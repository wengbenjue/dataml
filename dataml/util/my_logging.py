import logging
import colorlog

# 创建一个 logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG

# 创建一个控制台处理器
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(name)s:%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

# 将格式设置到处理器中
handler.setFormatter(formatter)

# 如果 logger 已经有处理器，先移除所有处理器，避免重复输出
if not logger.hasHandlers():
    logger.addHandler(handler)
else:
    logger.handlers = []
    logger.addHandler(handler)

