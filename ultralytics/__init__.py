# 外层 ultralytics 的桥接：先提供一个本地版本号，再导入内层真正的包
__version__ = "0.0.0-local"

from .ultralytics import *   # 导出内层 ultralytics 的所有接口
