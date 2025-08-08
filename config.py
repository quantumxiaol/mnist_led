from dotenv import load_dotenv
from typing import Optional
import os

# 加载 .env 文件（从项目根目录开始）
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

class Config:
    """
    配置类：集中管理所有环境变量
    """
    DEVICE=os.getenv("DEVICE",default="cpu")

    @classmethod
    def validate(cls):
        """
        验证必要配置是否已设置
        """
    
    pass

config = Config()