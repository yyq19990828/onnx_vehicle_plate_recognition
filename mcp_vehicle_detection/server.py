#!/usr/bin/env python3
"""
车辆检测MCP服务器
基于FastMCP框架，为车辆和车牌检测模型提供标准化的MCP接口
"""

import sys
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# 添加项目根目录到路径
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mcp.server.fastmcp import FastMCP, Context
import yaml

# 导入自定义模块
from models.detection_models import DetectionResult, BatchDetectionResult, DetectionStatistics
from models.config_models import DetectionConfig, ModelConfig, ServerConfig, ProcessingConfig
from services.detection_service import VehicleDetectionService
from services.session_manager import SessionManager
# 导入本地工具模块  
from mcp_utils.validation import (
    validate_image_path, validate_image_paths, validate_video_path,
    validate_confidence_threshold, sanitize_filename, validate_image_input,
    is_base64_image, decode_base64_image
)
from mcp_utils.image_processor import ImageProcessor, process_image_input, cleanup_temp_files
from mcp_utils.unified_detection import UnifiedVehicleDetector, DetectionToolFactory


class VehicleDetectionMCPServer:
    """车辆检测MCP服务器类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化MCP服务器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.server_config = None
        self.detection_config = None
        self.processing_config = None
        self.detection_service = None
        self.session_manager = None
        self.unified_detector = None
        
        # 加载配置
        self._load_config()
        
        # 创建FastMCP实例
        self.mcp = FastMCP(
            name=self.server_config.name,
            lifespan=self._lifespan_manager
        )
        
        # 注册工具、资源和提示
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                # 使用默认配置
                config_data = self._get_default_config()
                # 保存默认配置文件
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                logging.info(f"创建默认配置文件: {self.config_path}")
            
            # 解析配置
            self.server_config = ServerConfig(**config_data['server'])
            self.detection_config = DetectionConfig(**config_data['detection'])
            self.processing_config = ProcessingConfig(**config_data.get('processing', {}))
            
            logging.info("配置文件加载成功")
            
        except Exception as e:
            logging.error(f"配置文件加载失败: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'server': {
                'name': '车辆检测MCP服务器',
                'version': '1.0.0',
                'description': '基于ONNX的车辆和车牌检测服务',
                'max_concurrent_sessions': 10,
                'session_timeout': 3600,
                'cache_size': 100,
                'log_level': 'INFO'
            },
            'detection': {
                'detection_model': {
                    'model_path': '../models/yolov8s_640.onnx',
                    'input_shape': [640, 640],
                    'conf_threshold': 0.5,
                    'iou_threshold': 0.5
                },
                'color_layer_model': {
                    'model_path': '../models/color_layer.onnx',
                    'conf_threshold': 0.8
                },
                'ocr_model': {
                    'model_path': '../models/ocr.onnx',
                    'conf_threshold': 0.7
                },
                'class_names': ['vehicle', 'plate'],
                'colors': {0: [255, 0, 0], 1: [0, 255, 0]},
                'roi_top_ratio': 0.5,
                'plate_conf_threshold': 0.6
            },
            'processing': {
                'frame_skip': 0,
                'batch_size': 1,
                'max_image_size': 50 * 1024 * 1024,  # 50MB
                'max_image_dimension': 4096,
                'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
                'max_video_duration': 3600,
                'enable_image_optimization': True
            }
        }
    
    @asynccontextmanager
    async def _lifespan_manager(self, server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
        """应用生命周期管理"""
        try:
            # 启动时初始化
            logging.info("正在初始化车辆检测服务...")
            
            # 初始化检测服务
            self.detection_service = VehicleDetectionService(self.detection_config)
            
            # 初始化会话管理器
            self.session_manager = SessionManager(
                max_sessions=self.server_config.max_concurrent_sessions,
                session_timeout=self.server_config.session_timeout,
                cache_size=self.server_config.cache_size
            )
            
            # 初始化统一检测器
            self.unified_detector = UnifiedVehicleDetector(
                detection_service=self.detection_service,
                session_manager=self.session_manager,
                processing_config=self.processing_config
            )
            
            context = {
                'detection_service': self.detection_service,
                'session_manager': self.session_manager,
                'unified_detector': self.unified_detector,
                'server_config': self.server_config,
                'detection_config': self.detection_config,
                'processing_config': self.processing_config
            }
            
            logging.info("车辆检测MCP服务器启动成功")
            yield context
            
        finally:
            # 关闭时清理
            logging.info("正在关闭车辆检测MCP服务器...")
            # 清理临时文件
            cleanup_temp_files()
            if self.unified_detector:
                self.unified_detector.cleanup()
    
    def _register_tools(self):
        """注册MCP工具"""
        
        # 注册统一检测工具
        DetectionToolFactory.create_unified_detection_tool(self.mcp, self.unified_detector)
        
        # 注册图片信息工具  
        DetectionToolFactory.create_image_info_tool(self.mcp)
        
        @self.mcp.tool(description="通过文件路径检测图片中的车辆和车牌")
        async def detect_vehicle_by_path(
            image_path: str,
            session_id: Optional[str] = None,
            confidence_threshold: Optional[float] = None,
            ctx: Context = None
        ) -> DetectionResult:
            """
            通过文件路径检测单张图片中的车辆和车牌
            
            Args:
                image_path: 图片文件的绝对路径
                session_id: 可选的会话ID，如果不提供将创建新会话
                confidence_threshold: 可选的置信度阈值覆盖
                
            Returns:
                检测结果
            """
            try:
                # 验证文件路径
                if not validate_image_path(image_path):
                    raise ValueError(f"无效的图片路径: {image_path}")
                
                if confidence_threshold is not None and not validate_confidence_threshold(confidence_threshold):
                    raise ValueError(f"无效的置信度阈值: {confidence_threshold}")
                
                # 获取服务实例
                detection_service = ctx.request_context.lifespan_context['detection_service']
                session_manager = ctx.request_context.lifespan_context['session_manager']
                
                # 创建或获取会话
                if session_id is None:
                    session_id = session_manager.create_session("path_detection")
                    await ctx.info(f"创建新检测会话: {session_id}")
                else:
                    session_info = session_manager.get_session(session_id)
                    if session_info is None:
                        raise ValueError(f"会话不存在: {session_id}")
                
                await ctx.info(f"开始检测图片: {image_path}")
                
                # 应用置信度阈值覆盖
                original_threshold = None
                if confidence_threshold is not None:
                    original_threshold = detection_service.config.detection_model.conf_threshold
                    detection_service.config.detection_model.conf_threshold = confidence_threshold
                
                try:
                    # 执行检测
                    result = detection_service.detect_single_image(image_path)
                    
                    # 添加结果到会话
                    session_manager.add_result(session_id, result)
                    
                    await ctx.info(f"检测完成: 发现 {result.vehicle_count} 辆车, {result.plate_count} 个车牌")
                    
                    return result
                finally:
                    # 恢复原始置信度阈值
                    if original_threshold is not None:
                        detection_service.config.detection_model.conf_threshold = original_threshold
                
            except Exception as e:
                await ctx.error(f"图片检测失败: {e}")
                raise
        
        @self.mcp.tool(description="通过base64数据检测图片中的车辆和车牌")
        async def detect_vehicle_by_base64(
            base64_data: str,
            session_id: Optional[str] = None,
            confidence_threshold: Optional[float] = None,
            ctx: Context = None
        ) -> DetectionResult:
            """
            通过base64编码数据检测单张图片中的车辆和车牌
            
            Args:
                base64_data: base64编码的图片数据（支持data URL格式）
                session_id: 可选的会话ID，如果不提供将创建新会话
                confidence_threshold: 可选的置信度阈值覆盖
                
            Returns:
                检测结果
            """
            temp_file_path = None
            try:
                # 检查输入数据长度
                if len(base64_data) > 100 * 1024 * 1024:  # 100MB base64数据限制
                    raise ValueError("图片数据太大，请使用小于50MB的图片")
                
                # 验证和解码base64数据
                if not is_base64_image(base64_data):
                    raise ValueError("无效的base64图片数据")
                
                temp_file_path, _ = decode_base64_image(base64_data)
                
                if confidence_threshold is not None and not validate_confidence_threshold(confidence_threshold):
                    raise ValueError(f"无效的置信度阈值: {confidence_threshold}")
                
                # 获取服务实例
                detection_service = ctx.request_context.lifespan_context['detection_service']
                session_manager = ctx.request_context.lifespan_context['session_manager']
                
                # 创建或获取会话
                if session_id is None:
                    session_id = session_manager.create_session("base64_detection")
                    await ctx.info(f"创建新检测会话: {session_id}")
                else:
                    session_info = session_manager.get_session(session_id)
                    if session_info is None:
                        raise ValueError(f"会话不存在: {session_id}")
                
                await ctx.info(f"开始检测base64图片数据")
                
                # 应用置信度阈值覆盖
                original_threshold = None
                if confidence_threshold is not None:
                    original_threshold = detection_service.config.detection_model.conf_threshold
                    detection_service.config.detection_model.conf_threshold = confidence_threshold
                
                try:
                    # 执行检测
                    result = detection_service.detect_single_image(temp_file_path)
                    
                    # 添加结果到会话
                    session_manager.add_result(session_id, result)
                    
                    await ctx.info(f"检测完成: 发现 {result.vehicle_count} 辆车, {result.plate_count} 个车牌")
                    
                    return result
                finally:
                    # 恢复原始置信度阈值
                    if original_threshold is not None:
                        detection_service.config.detection_model.conf_threshold = original_threshold
                
            except Exception as e:
                await ctx.error(f"base64图片检测失败: {e}")
                raise
            finally:
                # 清理临时文件
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except OSError:
                        pass
        
        @self.mcp.tool()
        async def detect_vehicle_batch(
            image_paths: List[str],
            session_id: Optional[str] = None,
            confidence_threshold: Optional[float] = None,
            ctx: Context = None
        ) -> BatchDetectionResult:
            """
            批量检测多张图片中的车辆和车牌
            
            Args:
                image_paths: 图片路径列表
                session_id: 可选的会话ID
                confidence_threshold: 可选的置信度阈值覆盖
                
            Returns:
                批量检测结果
            """
            try:
                # 验证输入
                valid_paths = validate_image_paths(image_paths)
                if not valid_paths:
                    raise ValueError("没有有效的图片路径")
                
                if len(valid_paths) != len(image_paths):
                    await ctx.warning(f"过滤了 {len(image_paths) - len(valid_paths)} 个无效路径")
                
                # 获取服务实例
                detection_service = ctx.request_context.lifespan_context['detection_service']
                session_manager = ctx.request_context.lifespan_context['session_manager']
                
                # 创建或获取会话
                if session_id is None:
                    session_id = session_manager.create_session("batch_detection")
                
                await ctx.info(f"开始批量检测 {len(valid_paths)} 张图片")
                
                # 执行批量检测
                batch_result = detection_service.detect_batch_images(
                    valid_paths, confidence_threshold=confidence_threshold
                )
                
                # 添加所有结果到会话
                for result in batch_result.results:
                    session_manager.add_result(session_id, result)
                
                await ctx.info(f"批量检测完成: 成功率 {batch_result.success_rate:.2%}")
                
                return batch_result
                
            except Exception as e:
                await ctx.error(f"批量检测失败: {e}")
                raise
        
        @self.mcp.tool()  
        async def get_detection_statistics(ctx: Context = None) -> DetectionStatistics:
            """
            获取检测统计信息
            
            Returns:
                检测统计信息
            """
            try:
                session_manager = ctx.request_context.lifespan_context['session_manager']
                statistics = session_manager.get_statistics()
                
                await ctx.info("获取统计信息成功")
                return statistics
                
            except Exception as e:
                await ctx.error(f"获取统计信息失败: {e}")
                raise
        
        @self.mcp.tool()
        async def create_detection_session(
            session_type: str = "detection",
            ctx: Context = None
        ) -> Dict[str, str]:
            """
            创建新的检测会话
            
            Args:
                session_type: 会话类型
                
            Returns:
                会话信息
            """
            try:
                session_manager = ctx.request_context.lifespan_context['session_manager']
                session_id = session_manager.create_session(session_type)
                
                await ctx.info(f"创建会话成功: {session_id}")
                
                return {
                    "session_id": session_id,
                    "session_type": session_type,
                    "status": "created"
                }
                
            except Exception as e:
                await ctx.error(f"创建会话失败: {e}")
                raise
        
        @self.mcp.tool()
        async def list_active_sessions(ctx: Context = None) -> List[Dict[str, Any]]:
            """
            列出所有活跃会话
            
            Returns:
                活跃会话列表
            """
            try:
                session_manager = ctx.request_context.lifespan_context['session_manager']
                sessions = session_manager.list_active_sessions()
                
                await ctx.info(f"当前有 {len(sessions)} 个活跃会话")
                return sessions
                
            except Exception as e:
                await ctx.error(f"获取会话列表失败: {e}")
                raise
        
        @self.mcp.tool(description="使用项目中的示例图片进行车辆检测演示")
        async def detect_sample_image(
            ctx: Context = None
        ) -> DetectionResult:
            """
            使用项目中的示例图片进行检测演示
            
            Returns:
                检测结果
            """
            try:
                # 使用项目中现有的示例图片
                sample_image_path = "/home/tyjt/桌面/onnx_vehicle_plate_recognition/data/sample.jpg"
                
                if not os.path.exists(sample_image_path):
                    # 尝试其他可用的图片
                    data_dir = "/home/tyjt/桌面/onnx_vehicle_plate_recognition/data"
                    for filename in ["R1_Ce_CamS.jpg", "R1_Dn_CamE.jpg", "R1_Dn_CamW.jpg", "R2_Ce_CamN.jpg"]:
                        potential_path = os.path.join(data_dir, filename)
                        if os.path.exists(potential_path):
                            sample_image_path = potential_path
                            break
                    else:
                        raise ValueError("没有找到可用的示例图片")
                
                await ctx.info(f"使用示例图片: {os.path.basename(sample_image_path)}")
                
                # 获取服务实例
                detection_service = ctx.request_context.lifespan_context['detection_service']
                session_manager = ctx.request_context.lifespan_context['session_manager']
                
                # 创建新会话
                session_id = session_manager.create_session("sample_detection")
                await ctx.info(f"创建演示会话: {session_id}")
                
                # 执行检测
                result = detection_service.detect_single_image(sample_image_path)
                
                # 添加结果到会话
                session_manager.add_result(session_id, result)
                
                await ctx.info(f"检测完成: 发现 {result.vehicle_count} 辆车, {result.plate_count} 个车牌")
                
                return result
                
            except Exception as e:
                await ctx.error(f"示例图片检测失败: {e}")
                raise
    
    def _register_resources(self):
        """注册MCP资源"""
        
        @self.mcp.resource("detection://results/{session_id}")
        def get_session_results(session_id: str) -> str:
            """获取指定会话的检测结果"""
            try:
                results = self.session_manager.get_session_results(session_id)
                
                if not results:
                    return json.dumps({"message": f"会话 {session_id} 没有检测结果"}, 
                                    ensure_ascii=False, indent=2)
                
                # 序列化结果
                results_data = [result.model_dump() for result in results]
                return json.dumps(results_data, ensure_ascii=False, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"获取会话结果失败: {e}"}, 
                                ensure_ascii=False, indent=2)
        
        @self.mcp.resource("detection://config")
        def get_detection_config() -> str:
            """获取当前检测配置"""
            try:
                return json.dumps(self.config.model_dump(), ensure_ascii=False, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"获取配置失败: {e}"}, 
                                ensure_ascii=False, indent=2)
        
        @self.mcp.resource("detection://models/info")
        def get_models_info() -> str:
            """获取模型信息"""
            try:
                models_info = {
                    "detection_model": {
                        "path": self.config.detection.detection_model.model_path,
                        "input_shape": self.config.detection.detection_model.input_shape,
                        "conf_threshold": self.config.detection.detection_model.conf_threshold,
                        "iou_threshold": self.config.detection.detection_model.iou_threshold
                    },
                    "color_layer_model": {
                        "path": self.config.detection.color_layer_model.model_path
                    },
                    "ocr_model": {
                        "path": self.config.detection.ocr_model.model_path
                    },
                    "class_names": self.config.detection.class_names,
                    "supported_classes": len(self.config.detection.class_names)
                }
                
                return json.dumps(models_info, ensure_ascii=False, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"获取模型信息失败: {e}"}, 
                                ensure_ascii=False, indent=2)
    
    def _register_prompts(self):
        """注册MCP提示"""
        
        @self.mcp.prompt("analyze_detection")
        def analyze_detection_prompt(
            session_id: str,
            analysis_type: str = "summary",
            ctx: Context = None
        ) -> str:
            """分析检测结果的提示模板"""
            try:
                session_manager = ctx.request_context.lifespan_context['session_manager']
                results = session_manager.get_session_results(session_id)
                
                if not results:
                    return f"会话 {session_id} 没有检测结果可供分析"
                
                if analysis_type == "summary":
                    total_vehicles = sum(r.vehicle_count for r in results)
                    total_plates = sum(r.plate_count for r in results)
                    avg_time = sum(r.processing_time for r in results) / len(results)
                    
                    return f"""
请分析以下车辆检测结果摘要：

检测会话: {session_id}
检测图片数量: {len(results)}
总检测车辆数: {total_vehicles}
总检测车牌数: {total_plates}
平均处理时间: {avg_time:.3f}秒

请提供对这些结果的专业分析和建议。
"""
                
                elif analysis_type == "detailed":
                    details = []
                    for i, result in enumerate(results, 1):
                        details.append(f"""
图片 {i}:
- 车辆数量: {result.vehicle_count}
- 车牌数量: {result.plate_count}
- 车牌文字: {', '.join(result.plate_texts)}
- 处理时间: {result.processing_time:.3f}秒
""")
                    
                    return f"""
请分析以下详细的车辆检测结果：

检测会话: {session_id}

{''.join(details)}

请对每张图片的检测结果进行详细分析。
"""
                
                else:
                    return f"不支持的分析类型: {analysis_type}"
                
            except Exception as e:
                return f"生成分析提示失败: {e}"
        
        @self.mcp.prompt("format_report")
        def format_report_prompt(
            session_id: str,
            report_format: str = "markdown",
            ctx: Context = None
        ) -> str:
            """格式化检测报告的提示模板"""
            try:
                session_manager = ctx.request_context.lifespan_context['session_manager']
                results = session_manager.get_session_results(session_id)
                
                if not results:
                    return f"会话 {session_id} 没有检测结果可生成报告"
                
                statistics = session_manager.get_statistics()
                
                return f"""
请将以下检测数据格式化为 {report_format} 格式的专业报告：

## 检测会话信息
- 会话ID: {session_id}
- 检测图片数量: {len(results)}
- 检测时间范围: {results[0].timestamp} 到 {results[-1].timestamp}

## 统计摘要
- 总检测次数: {statistics.total_detections}
- 总车辆数量: {statistics.total_vehicles}
- 总车牌数量: {statistics.total_plates}
- 平均处理时间: {statistics.average_processing_time:.3f}秒

## 详细结果
{json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2)}

请生成一份专业的检测报告，包含数据分析、趋势观察和改进建议。
"""
                
            except Exception as e:
                return f"生成报告提示失败: {e}"
    
    def run(self, transport: str = "stdio", **kwargs):
        """运行MCP服务器"""
        try:
            logging.basicConfig(
                level=getattr(logging, self.server_config.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            logging.info(f"启动车辆检测MCP服务器 (transport: {transport})")
            self.mcp.run(transport=transport, **kwargs)
            
        except Exception as e:
            logging.error(f"服务器运行失败: {e}")
            raise


def main():
    """主入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='车辆检测MCP服务器')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--transport', default='stdio', 
                       choices=['stdio', 'streamable-http'], help='传输协议')
    parser.add_argument('--port', type=int, default=8000, help='HTTP端口号')
    parser.add_argument('--host', default='localhost', help='HTTP主机地址')
    
    args = parser.parse_args()
    
    # 创建并运行服务器
    server = VehicleDetectionMCPServer(config_path=args.config)
    
    if args.transport == 'streamable-http':
        server.run(transport='streamable-http', port=args.port, host=args.host)
    else:
        server.run(transport='stdio')


if __name__ == '__main__':
    main()