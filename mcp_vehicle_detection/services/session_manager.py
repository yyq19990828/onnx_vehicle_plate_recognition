"""
会话管理服务
管理检测会话、缓存结果和统计信息
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import uuid

from models.detection_models import (
    DetectionResult, BatchDetectionResult, DetectionStatistics,
    VideoAnalysisResult, PlateColor
)


class SessionManager:
    """会话管理器"""
    
    def __init__(self, max_sessions: int = 10, session_timeout: int = 3600,
                 cache_size: int = 100):
        """
        初始化会话管理器
        
        Args:
            max_sessions: 最大并发会话数
            session_timeout: 会话超时时间（秒）
            cache_size: 缓存大小
        """
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.cache_size = cache_size
        
        # 会话存储
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_results: Dict[str, List[DetectionResult]] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        
        # 结果缓存 (LRU)
        self.result_cache: deque = deque(maxlen=cache_size)
        self.cache_index: Dict[str, DetectionResult] = {}
        
        # 统计信息
        self.total_detections = 0
        self.total_vehicles = 0
        self.total_plates = 0
        self.total_processing_time = 0.0
        self.plate_color_count = defaultdict(int)
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 启动清理线程
        self._start_cleanup_thread()
    
    def create_session(self, session_type: str = "detection") -> str:
        """
        创建新会话
        
        Args:
            session_type: 会话类型
            
        Returns:
            会话ID
        """
        with self._lock:
            # 清理过期会话
            self._cleanup_expired_sessions()
            
            # 检查会话数量限制
            if len(self.active_sessions) >= self.max_sessions:
                # 移除最旧的会话
                oldest_session = min(self.session_timestamps.items(), 
                                   key=lambda x: x[1])[0]
                self.close_session(oldest_session)
            
            # 创建新会话
            session_id = str(uuid.uuid4())
            self.active_sessions[session_id] = {
                "type": session_type,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "status": "active"
            }
            self.session_results[session_id] = []
            self.session_timestamps[session_id] = datetime.now()
            
            logging.info(f"创建新会话: {session_id}")
            return session_id
    
    def close_session(self, session_id: str):
        """
        关闭会话
        
        Args:
            session_id: 会话ID
        """
        with self._lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                del self.session_results[session_id]
                del self.session_timestamps[session_id]
                logging.info(f"关闭会话: {session_id}")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息
        """
        with self._lock:
            if session_id in self.active_sessions:
                # 更新最后活动时间
                self.active_sessions[session_id]["last_activity"] = datetime.now()
                self.session_timestamps[session_id] = datetime.now()
                return self.active_sessions[session_id].copy()
            return None
    
    def add_result(self, session_id: str, result: DetectionResult):
        """
        添加检测结果到会话
        
        Args:
            session_id: 会话ID
            result: 检测结果
        """
        with self._lock:
            if session_id in self.session_results:
                self.session_results[session_id].append(result)
                
                # 添加到缓存
                self._add_to_cache(result)
                
                # 更新统计信息
                self._update_statistics(result)
                
                logging.debug(f"为会话 {session_id} 添加检测结果")
    
    def get_session_results(self, session_id: str) -> List[DetectionResult]:
        """
        获取会话的所有结果
        
        Args:
            session_id: 会话ID
            
        Returns:
            检测结果列表
        """
        with self._lock:
            return self.session_results.get(session_id, []).copy()
    
    def get_result_by_id(self, result_id: str) -> Optional[DetectionResult]:
        """
        通过结果ID获取检测结果
        
        Args:
            result_id: 结果ID (session_id)
            
        Returns:
            检测结果
        """
        # 首先在缓存中查找
        if result_id in self.cache_index:
            return self.cache_index[result_id]
        
        # 在会话结果中查找
        with self._lock:
            for results in self.session_results.values():
                for result in results:
                    if result.session_id == result_id:
                        return result
        
        return None
    
    def get_statistics(self) -> DetectionStatistics:
        """
        获取检测统计信息
        
        Returns:
            统计信息
        """
        with self._lock:
            avg_processing_time = (self.total_processing_time / self.total_detections 
                                 if self.total_detections > 0 else 0.0)
            
            most_common_color = None
            if self.plate_color_count:
                most_common_color = max(self.plate_color_count.items(), 
                                      key=lambda x: x[1])[0]
                most_common_color = PlateColor(most_common_color)
            
            return DetectionStatistics(
                total_detections=self.total_detections,
                total_vehicles=self.total_vehicles,
                total_plates=self.total_plates,
                average_processing_time=avg_processing_time,
                most_common_plate_color=most_common_color
            )
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有活跃会话
        
        Returns:
            活跃会话列表
        """
        with self._lock:
            sessions = []
            for session_id, session_info in self.active_sessions.items():
                session_data = session_info.copy()
                session_data["session_id"] = session_id
                session_data["result_count"] = len(self.session_results.get(session_id, []))
                sessions.append(session_data)
            return sessions
    
    def _add_to_cache(self, result: DetectionResult):
        """添加结果到缓存"""
        # 如果缓存已满，移除最旧的条目
        if len(self.result_cache) >= self.cache_size:
            oldest_result = self.result_cache.popleft()
            if oldest_result.session_id in self.cache_index:
                del self.cache_index[oldest_result.session_id]
        
        # 添加新结果
        self.result_cache.append(result)
        self.cache_index[result.session_id] = result
    
    def _update_statistics(self, result: DetectionResult):
        """更新统计信息"""
        self.total_detections += 1
        self.total_vehicles += result.vehicle_count
        self.total_plates += result.plate_count
        self.total_processing_time += result.processing_time
        
        # 统计车牌颜色
        for detection in result.detections:
            if detection.plate_info:
                self.plate_color_count[detection.plate_info.color.value] += 1
    
    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, timestamp in self.session_timestamps.items():
            if now - timestamp > timedelta(seconds=self.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.close_session(session_id)
            logging.info(f"清理过期会话: {session_id}")
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # 每5分钟清理一次
                    with self._lock:
                        self._cleanup_expired_sessions()
                except Exception as e:
                    logging.error(f"清理线程错误: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logging.info("会话清理线程已启动")


# 全局会话管理器实例
session_manager = SessionManager()