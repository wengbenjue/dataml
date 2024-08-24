import logging
import colorlog
# 配置 logging
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='example.log',  # 指定日志文件
)

# 创建一个 logger 对象
logger = logging.getLogger(__name__)


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
# 将处理器添加到记录器中
logger.addHandler(handler)

# 输出日志到控制台
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)  # 设置控制台日志级别为 INFO
# console_formatter = logging.Formatter('%(levelname)s - %(message)s')
# console_handler.setFormatter(console_formatter)
# logger.addHandler(console_handler)
