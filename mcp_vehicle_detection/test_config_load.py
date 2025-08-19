#!/usr/bin/env python3
"""
测试配置文件加载
"""

import sys
import os
import logging
from pathlib import Path

# 添加当前目录到路径
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加项目根目录到路径  
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 设置工作目录
os.chdir(current_dir)

try:
    from server import VehicleDetectionMCPServer
    print("✓ 服务器导入成功")
except ImportError as e:
    print(f"✗ 服务器导入失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_loading():
    """测试配置文件加载"""
    
    try:
        logger.info("🔧 测试配置文件加载...")
        
        # 尝试创建服务器实例
        server = VehicleDetectionMCPServer(config_path="config.yaml")
        
        logger.info("✅ 配置文件加载成功!")
        logger.info(f"  服务器名称: {server.server_config.name}")
        logger.info(f"  服务器版本: {server.server_config.version}")
        logger.info(f"  检测模型: {server.detection_config.detection_model.model_path}")
        logger.info(f"  置信度阈值: {server.detection_config.detection_model.conf_threshold}")
        logger.info(f"  最大图片大小: {server.processing_config.max_image_size / (1024*1024):.0f}MB")
        logger.info(f"  最大图片尺寸: {server.processing_config.max_image_dimension}px")
        logger.info(f"  图片优化: {server.processing_config.enable_image_optimization}")
        logger.info(f"  支持格式: {server.processing_config.supported_formats}")
        
        logger.info("🎉 配置验证通过，MCP服务器可以正常启动!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config_loading()
    if success:
        logger.info("✅ 修复成功 - 可以在Claude Desktop中使用了!")
    else:
        logger.error("❌ 仍有问题需要修复")