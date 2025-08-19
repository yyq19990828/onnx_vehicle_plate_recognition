"""
统一检测工具
借鉴DINO-X-MCP的架构设计，提供统一的图片检测接口
"""

import os
import logging
from typing import Union, Optional, Dict, Any
from pathlib import Path

from .image_processor import ImageProcessor, process_image_input
from .validation import validate_confidence_threshold

logger = logging.getLogger(__name__)


class UnifiedVehicleDetector:
    """
    统一车辆检测器
    借鉴DINO-X-MCP的DinoXApiClient设计思路，提供统一的检测接口
    """
    
    def __init__(self, detection_service, session_manager=None, processing_config=None):
        """
        初始化统一检测器
        
        Args:
            detection_service: 车辆检测服务实例
            session_manager: 会话管理器实例（可选）
            processing_config: 处理配置实例（可选）
        """
        self.detection_service = detection_service
        self.session_manager = session_manager
        self.processing_config = processing_config
        
        # 根据配置初始化图片处理器
        if processing_config:
            self.image_processor = ImageProcessor(
                max_image_size=processing_config.max_image_size,
                max_dimension=processing_config.max_image_dimension
            )
        else:
            self.image_processor = ImageProcessor()
    
    async def detect_vehicle_universal(
        self,
        image_input: str,
        session_id: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        include_visualization: bool = False,
        ctx=None
    ) -> Dict[str, Any]:
        """
        通用车辆检测方法，支持多种输入格式
        借鉴DINO-X-MCP的performDetection方法设计
        
        Args:
            image_input: 图片输入，支持：
                        - 文件路径
                        - file:// URI
                        - https:// URI
                        - data:image/ URI
                        - base64字符串
            session_id: 会话ID（可选）
            confidence_threshold: 置信度阈值覆盖（可选）
            include_visualization: 是否包含可视化结果
            ctx: 上下文对象
            
        Returns:
            检测结果字典
        """
        temp_file_path = None
        is_temp_file = False
        
        try:
            # 记录开始检测
            if ctx:
                await ctx.info(f"开始通用车辆检测...")
            
            # 1. 处理图片输入
            file_path, is_temp_file = self.image_processor.process_image_uri(image_input)
            temp_file_path = file_path if is_temp_file else None
            
            # 2. 预处理优化（根据配置决定是否启用）
            if self.processing_config and self.processing_config.enable_image_optimization:
                optimized_path = self.image_processor.preprocess_for_model(file_path)
                if optimized_path != file_path:
                    # 如果生成了优化后的文件，也需要在最后清理
                    if not is_temp_file:
                        temp_file_path = optimized_path
                    file_path = optimized_path
                    if ctx:
                        await ctx.info("应用图片预处理优化")
            
            if ctx:
                input_type = self._get_input_type(image_input)
                await ctx.info(f"输入类型: {input_type}, 处理后路径: {os.path.basename(file_path)}")
            
            # 3. 验证置信度阈值
            if confidence_threshold is not None and not validate_confidence_threshold(confidence_threshold):
                raise ValueError(f"无效的置信度阈值: {confidence_threshold}")
            
            # 4. 创建或获取会话
            if self.session_manager and session_id is None:
                session_id = self.session_manager.create_session("universal_detection")
                if ctx:
                    await ctx.info(f"创建新检测会话: {session_id}")
            elif self.session_manager and session_id:
                session_info = self.session_manager.get_session(session_id)
                if session_info is None:
                    raise ValueError(f"会话不存在: {session_id}")
            
            # 5. 应用置信度阈值覆盖
            original_threshold = None
            if confidence_threshold is not None:
                original_threshold = self.detection_service.config.detection_model.conf_threshold
                self.detection_service.config.detection_model.conf_threshold = confidence_threshold
                if ctx:
                    await ctx.info(f"应用置信度阈值覆盖: {confidence_threshold}")
            
            try:
                # 6. 执行检测
                result = self.detection_service.detect_single_image(file_path)
                
                # 7. 添加结果到会话
                if self.session_manager and session_id:
                    self.session_manager.add_result(session_id, result)
                
                # 8. 构建返回结果
                response = {
                    "detection_result": result,
                    "session_id": session_id,
                    "input_type": self._get_input_type(image_input),
                    "processing_info": {
                        "file_path": file_path,
                        "is_temp_file": is_temp_file,
                        "confidence_threshold": confidence_threshold or 
                                             self.detection_service.config.detection_model.conf_threshold
                    }
                }
                
                # 9. 可视化处理（如果需要）
                if include_visualization:
                    try:
                        visualization_path = await self._create_visualization(file_path, result)
                        response["visualization_path"] = visualization_path
                    except Exception as viz_error:
                        logger.warning(f"可视化创建失败: {viz_error}")
                        response["visualization_error"] = str(viz_error)
                
                if ctx:
                    await ctx.info(
                        f"检测完成: 发现 {result.vehicle_count} 辆车, "
                        f"{result.plate_count} 个车牌, "
                        f"处理时间: {result.processing_time:.3f}秒"
                    )
                
                return response
                
            finally:
                # 恢复原始置信度阈值
                if original_threshold is not None:
                    self.detection_service.config.detection_model.conf_threshold = original_threshold
            
        except Exception as e:
            if ctx:
                await ctx.error(f"通用车辆检测失败: {e}")
            raise
        
        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
    
    def _get_input_type(self, image_input: str) -> str:
        """获取输入类型描述"""
        if image_input.startswith("file://"):
            return "file_uri"
        elif image_input.startswith(("http://", "https://")):
            return "http_uri"
        elif image_input.startswith("data:image/"):
            return "data_uri"
        elif len(image_input) > 100 and self.image_processor._is_base64_string(image_input):
            return "base64_string"
        else:
            return "file_path"
    
    async def _create_visualization(self, image_path: str, result) -> str:
        """创建检测结果可视化"""
        # 这里可以实现可视化逻辑
        # 暂时返回原图路径，实际可以绘制检测框
        return image_path
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'image_processor'):
            self.image_processor.cleanup_temp_files()


class DetectionToolFactory:
    """
    检测工具工厂
    借鉴DINO-X-MCP的工具注册模式
    """
    
    @staticmethod
    def create_unified_detection_tool(mcp_server, unified_detector):
        """创建统一检测工具"""
        
        @mcp_server.tool(description="通用车辆检测工具，支持多种图片输入格式")
        async def detect_vehicle_universal(
            image_input: str,
            session_id: Optional[str] = None,
            confidence_threshold: Optional[float] = None,
            include_visualization: bool = False,
            ctx = None
        ):
            """
            通用车辆检测工具
            
            Args:
                image_input: 图片输入，支持：
                            - 本地文件路径: /path/to/image.jpg
                            - file URI: file:///path/to/image.jpg
                            - HTTP URI: https://example.com/image.jpg
                            - Data URI: data:image/jpeg;base64,/9j/4AAQ...
                            - Base64字符串: /9j/4AAQ...
                session_id: 可选的会话ID
                confidence_threshold: 可选的置信度阈值覆盖 (0.0-1.0)
                include_visualization: 是否生成可视化结果
                
            Returns:
                检测结果包含车辆和车牌信息
            """
            return await unified_detector.detect_vehicle_universal(
                image_input=image_input,
                session_id=session_id,
                confidence_threshold=confidence_threshold,
                include_visualization=include_visualization,
                ctx=ctx
            )
        
        return detect_vehicle_universal
    
    @staticmethod
    def create_image_info_tool(mcp_server):
        """创建图片信息工具"""
        
        @mcp_server.tool(description="获取图片信息和支持的格式")
        async def get_image_info(ctx = None):
            """
            获取图片处理信息
            
            Returns:
                支持的图片格式和处理能力信息
            """
            processor = ImageProcessor()
            
            info = {
                "supported_formats": list(processor.supported_formats),
                "max_image_size_mb": processor.max_image_size // (1024 * 1024),
                "supported_inputs": [
                    "本地文件路径",
                    "file:// URI",
                    "http:// / https:// URI",
                    "data:image/ URI (base64)",
                    "纯base64字符串"
                ],
                "example_usage": {
                    "file_path": "/path/to/image.jpg",
                    "file_uri": "file:///path/to/image.jpg",
                    "http_uri": "https://example.com/image.jpg",
                    "data_uri": "data:image/jpeg;base64,/9j/4AAQ...",
                    "base64": "/9j/4AAQ..."
                }
            }
            
            if ctx:
                await ctx.info("获取图片处理信息成功")
            
            return info
        
        return get_image_info