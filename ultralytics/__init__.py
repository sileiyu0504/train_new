# 外层 ultralytics 的桥接：先提供一个本地版本号，再导入内层真正的包
import os
import pkgutil

# 确保包搜索路径包含内层 ultralytics
_INNER_ROOT = os.path.join(os.path.dirname(__file__), "ultralytics")
if _INNER_ROOT not in __path__:
    __path__.append(_INNER_ROOT)
# 兼容命名空间扩展
__path__ = pkgutil.extend_path(list(__path__), __name__)

__version__ = "0.0.0-local"

from .ultralytics import *  # 导出内层 ultralytics 的所有接口
