# Polygraphy 使用指南

Polygraphy 是一个深度学习推理原型设计和调试工具包，提供 Python API 和命令行工具，支持跨框架模型比较、转换和调试。

## 🚀 快速开始

### 安装

```bash
# 基础安装
pip install polygraphy

# 或从源码构建
make install  # Linux
# 或 .\install.ps1  # Windows
```

### 基本用法

```bash
# 跨框架推理比较
polygraphy run model.onnx --onnxrt --trt

# 模型转换 
polygraphy convert model.onnx --convert-to trt --output model.engine

# 模型检查
polygraphy inspect model model.onnx
```

## 📚 核心概念

### 懒加载机制
Polygraphy 采用懒加载设计，只有在实际使用时才导入相关依赖，提高启动速度并减少内存占用。

### 后端系统
- **ONNX**: 模型加载、修改和验证
- **ONNX Runtime**: CPU/GPU 推理执行
- **TensorRT**: NVIDIA GPU 优化推理
- **TensorFlow**: TF 模型支持

### 构建者模式
使用 Loader 构建处理链，支持延迟求值和灵活配置。

## 🛠️ 主要工具命令

| 命令 | 功能 | 文档链接 |
|------|------|----------|
| `run` | 跨框架推理比较 | [详细文档](./run.md) |
| `convert` | 模型格式转换 | [详细文档](./convert.md) |
| `inspect` | 模型结构分析 | [详细文档](./inspect.md) |
| `surgeon` | ONNX 模型修改 | [详细文档](./surgeon.md) |
| `debug` | 调试工具集 | [详细文档](./debug.md) |
| `check` | 模型验证工具 | [详细文档](./check.md) |

## 🔧 环境配置

### 重要环境变量

```bash
# 自动安装依赖
export POLYGRAPHY_AUTOINSTALL_DEPS=1

# 安装确认提示
export POLYGRAPHY_ASK_BEFORE_INSTALL=1

# 自定义安装命令
export POLYGRAPHY_INSTALL_CMD="pip install"
```

### 依赖管理

Polygraphy 支持自动依赖管理：
- 缺失模块时自动安装
- 版本不匹配时自动升级
- 支持自定义安装标志和源

## 📋 常用工作流程

### 1. 准确性调试流程
```bash
# 1. 跨框架比较找出问题
polygraphy run model.onnx --onnxrt --trt

# 2. 逐层输出比较
polygraphy run model.onnx --onnxrt --trt --mark-all

# 3. 减少失败模型
polygraphy debug reduce model.onnx --output reduced.onnx
```

### 2. 性能优化流程  
```bash
# 1. INT8 量化
polygraphy convert model.onnx --convert-to trt --int8 --calibration-cache cache.cache

# 2. FP16 精度
polygraphy convert model.onnx --convert-to trt --fp16

# 3. 动态形状优化
polygraphy convert model.onnx --convert-to trt --trt-min-shapes input:[1,3,224,224] --trt-max-shapes input:[8,3,224,224]
```

### 3. 模型分析流程
```bash
# 1. 模型结构检查
polygraphy inspect model model.onnx

# 2. 数据检查
polygraphy inspect data inputs.json

# 3. ONNX 验证
polygraphy check lint model.onnx
```

## 📖 详细文档

### 核心工具
- [run - 跨框架推理比较](./run.md)
- [convert - 模型格式转换](./convert.md) 
- [inspect - 模型结构分析](./inspect.md)
- [surgeon - ONNX 模型修改](./surgeon.md)
- [debug - 调试工具集](./debug.md)
- [check - 模型验证工具](./check.md)

### 高级主题
- [API 编程指南](./api-guide.md)
- [调试最佳实践](./debugging-best-practices.md)
- [性能优化技巧](./performance-optimization.md)
- [常见问题解答](./faq.md)

## 🎯 使用场景

### 模型验证
- 跨框架一致性检查
- ONNX 模型语法验证
- 精度损失分析

### 性能优化
- TensorRT 引擎构建
- 量化配置调优
- 动态形状处理

### 问题调试
- 推理失败定位
- 模型减少和简化
- 策略问题诊断

## 💡 最佳实践

1. **使用懒加载**: 避免导入不必要的依赖
2. **环境变量配置**: 合理设置自动安装选项
3. **逐步调试**: 从简单比较开始，逐步深入分析
4. **缓存利用**: 使用校准缓存和策略缓存提高效率
5. **版本管理**: 明确指定依赖版本避免兼容性问题

## 📞 获取帮助

```bash
# 查看帮助
polygraphy -h
polygraphy <command> -h

# 详细日志
polygraphy run model.onnx --verbose

# 额外调试信息
polygraphy run model.onnx --extra-verbose
```

## 🔗 相关资源

- [官方文档](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/)
- [GitHub 仓库](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)
- [示例代码](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples)

---

*最后更新: 2024年8月*