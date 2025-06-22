import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_qa.log"),
        logging.StreamHandler()
    ]
)

class BaseChecker(ABC):
    """
    基础检查器抽象类，所有检查器都应继承此类
    """
    def __init__(self, dataset_root: str):
        """
        初始化基础检查器
        参数:
            dataset_root: 数据集根目录路径
        """
        self.dataset_root = dataset_root
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
    
    @abstractmethod
    def check(self) -> Dict[str, Any]:
        """
        执行检查
        返回:
            包含检查结果的字典
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取检查结果
        返回:
            包含检查结果的字典
        """
        return self.results