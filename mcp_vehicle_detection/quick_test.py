#!/usr/bin/env python3
"""
快速测试检测功能
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到路径
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mcp_utils.unified_detection import UnifiedVehicleDetector
from models.config_models import DetectionConfig, ModelConfig, ProcessingConfig
from services.detection_service import VehicleDetectionService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def quick_test():
    """快速测试"""
    
    # 测试图片
    test_image = "/home/tyjt/桌面/onnx_vehicle_plate_recognition/data/R2_Ce_CamN.jpg"
    
    if not os.path.exists(test_image):
        logger.error(f"测试图片不存在: {test_image}")
        return
    
    try:
        # 简化配置
        detection_config = DetectionConfig(
            detection_model=ModelConfig(
                model_path="/home/tyjt/桌面/onnx_vehicle_plate_recognition/models/det.onnx",
                input_shape=[640, 640],
                conf_threshold=0.25,
                iou_threshold=0.5  # RT-DETR会忽略这个参数
            ),
            color_layer_model=ModelConfig(
                model_path="/home/tyjt/桌面/onnx_vehicle_plate_recognition/models/color_layer.onnx",
                conf_threshold=0.8
            ),
            ocr_model=ModelConfig(
                model_path="/home/tyjt/桌面/onnx_vehicle_plate_recognition/models/ocr.onnx", 
                conf_threshold=0.7
            ),
            class_names=['vehicle', 'plate'],
            colors={0: [255, 0, 0], 1: [0, 255, 0]},
            roi_top_ratio=0.5,
            plate_conf_threshold=0.2
        )
        
        logger.info("初始化检测服务...")
        detection_service = VehicleDetectionService(detection_config)
        logger.info("✅ 检测服务初始化成功")
        
        processing_config = ProcessingConfig(enable_image_optimization=True)
        unified_detector = UnifiedVehicleDetector(
            detection_service=detection_service,
            processing_config=processing_config
        )
        
        # 模拟上下文
        class MockContext:
            async def info(self, msg): 
                logger.info(f"[CONTEXT] {msg}")
            async def error(self, msg): 
                logger.error(f"[CONTEXT] {msg}")
        
        ctx = MockContext()
        
        logger.info(f"开始检测: {os.path.basename(test_image)}")
        response = await unified_detector.detect_vehicle_universal(
            image_input=test_image,
            confidence_threshold=0.2,
            ctx=ctx
        )
        
        result = response['detection_result']
        
        logger.info("🎉 检测结果:")
        logger.info(f"  图片尺寸: {result.image_info.width}x{result.image_info.height}")
        logger.info(f"  处理时间: {result.processing_time:.3f}秒")
        logger.info(f"  车辆数量: {result.vehicle_count}")
        logger.info(f"  车牌数量: {result.plate_count}")
        
        if result.detections:
            for i, detection in enumerate(result.detections, 1):
                logger.info(f"  检测{i}: {detection.type.value} - 置信度: {detection.confidence:.3f}")
                
        logger.info("✅ 快速测试完成 - 系统正常工作!")
        
        unified_detector.cleanup()
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(quick_test())