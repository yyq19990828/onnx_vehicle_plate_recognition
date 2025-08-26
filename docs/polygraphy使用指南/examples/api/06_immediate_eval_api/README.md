# 立即评估函数式 API

## 简介

### [延迟加载器](../lazy_module.md) vs 立即评估

<!-- Polygraphy Test: Ignore Start -->
在大多数情况下，Polygraphy 附带的**延迟加载器**（Lazy Loaders）具有以下优点：

- **延迟执行**：它们允许我们将工作推迟到实际需要时才执行，这可以节省时间。
- **轻量级复制**：由于构造的加载器非常轻量级，使用延迟评估加载器的运行器可以轻松地复制到其他进程或线程中，然后在那里启动。如果运行器引用的是整个模型/推理会话，那么以这种方式复制它们将并非易事。
- **链式操作**：它们允许我们通过将加载器链接在一起来预先定义一系列操作，这提供了一种构建可重用函数的简单方法。例如，我们可以创建一个从 ONNX 导入模型并生成序列化 TensorRT 引擎的加载器：

    ```python
    build_engine = EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))
    ```

- **智能对象管理**：它们允许特殊的语义，即如果向加载器提供可调用对象，它将获得返回值的所有权，否则则不会。这些特殊的语义对于在多个加载器之间共享对象很有用。

### 延迟加载器的问题

然而，延迟加载器有时会导致代码可读性较差，甚至完全令人困惑。例如，考虑以下内容：

```python
# 这个例子中的每一行看起来几乎都一样，但行为却大相径庭。
# 其中一些行甚至会导致内存泄漏！

# 第1行：只是创建了一个加载器，没有执行任何实际工作
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))
# 返回类型：EngineBytesFromNetwork 加载器实例
# 内存状态：只分配了轻量级加载器对象

# 第2行：执行了完整的加载管道
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))()
# 返回类型：bytes (序列化的TensorRT引擎)
# 内存状态：已分配ONNX模型、TensorRT网络、构建器等资源

# 第3行：NetworkFromOnnxPath()先执行，返回(builder, network, parser)
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")())
# 等价于：EngineBytesFromNetwork((builder, network, parser))
# 返回类型：EngineBytesFromNetwork 加载器实例（但持有实际对象）
# 内存状态：已分配TensorRT对象，但引擎未构建

# 第4行：第3行的结果再次调用
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")())()
# 返回类型：bytes (序列化的TensorRT引擎)
# 内存状态：完成引擎构建

# 第5行：试图对bytes对象调用()，这会失败！
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))()()
# 第一个()返回bytes，第二个()尝试调用bytes() -> TypeError!
```

### 对象所有权和生命周期的复杂性

这种复杂性源于Polygraphy的**智能对象管理机制**：

#### 规则1：可调用对象 vs 实际对象
```python
# 传入加载器（可调用）- 外部加载器获得所有权
inner_loader = NetworkFromOnnxPath("/path/to/model.onnx")
outer_loader = EngineBytesFromNetwork(inner_loader)  # outer_loader拥有控制权

# 传入实际对象 - 不转移所有权
builder, network, parser = network_from_onnx_path("/path/to/model.onnx")  
outer_loader = EngineBytesFromNetwork((builder, network, parser))  # 用户仍需管理这些对象
```

#### 规则2：延迟执行的生命周期陷阱
```python
# 危险：可能的内存泄漏
def create_engine():
    # 这里创建的TensorRT对象可能不会被正确释放
    return EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")())()

# 安全：明确的生命周期管理
def create_engine_safe():
    builder, network, parser = network_from_onnx_path("/path/to/model.onnx")
    try:
        engine_bytes = engine_bytes_from_network((builder, network, parser))
        return engine_bytes
    finally:
        # 显式清理资源（如果需要）
        pass
```

### 函数式 API 的解决方案

因此，Polygraphy 提供了每个加载器的**立即评估函数式等价物**。每个函数式变体都使用与加载器相同的名称，但使用 `snake_case` 而不是 `PascalCase`。

#### 库实现原理：`@mod.export(funcify=True)` 装饰器

Polygraphy 通过元编程自动生成函数式API：

```python
# 在源码中，每个加载器类都使用这个装饰器
@mod.export(funcify=True)
class NetworkFromOnnxPath(BaseLoader):
    def __init__(self, path, **kwargs):
        # 初始化参数
        pass
    
    def call_impl(self, **kwargs):
        # 实际执行逻辑，返回 (builder, network, parser)
        pass
```

**装饰器的魔法过程**：

1. **签名合并**：分析 `__init__` 和 `call_impl` 的参数
2. **命名转换**：`NetworkFromOnnxPath` → `network_from_onnx_path`
3. **代码生成**：动态创建包装函数
4. **模块注入**：将生成的函数添加到模块的 `__dict__`

```python
# 装饰器生成的等价代码：
def network_from_onnx_path(path, **kwargs):
    """立即评估的函数式变体"""
    loader = NetworkFromOnnxPath(path, **kwargs)  # 创建加载器实例
    return loader(**kwargs)                        # 立即调用并返回结果
```

#### API对比详解

**延迟加载器方式**（类式API）：
```python
# 步骤1：创建加载器对象（轻量级，无实际工作）
parse_network = NetworkFromOnnxPath("/path/to/model.onnx")   
create_config = CreateConfig(fp16=True, tf32=True)          
build_engine = EngineFromNetwork(parse_network, create_config) 

# 步骤2：实际执行所有工作
engine = build_engine()  # 这时才解析ONNX、构建网络、创建引擎
```

**立即评估函数式方式**：
```python
# 每个函数调用都立即执行并返回实际对象
builder, network, parser = network_from_onnx_path("/path/to/model.onnx") # 立即解析ONNX
config = create_config(builder, network, fp16=True, tf32=True)          # 立即创建配置  
engine = engine_from_network((builder, network, parser), config)        # 立即构建引擎
```

#### 内存管理责任的转移

```python
# 延迟加载器：框架负责内存管理
with TrtRunner(EngineFromNetwork(NetworkFromOnnxPath("model.onnx"))) as runner:
    # 框架自动管理所有TensorRT对象的生命周期
    outputs = runner.infer(inputs)

# 函数式API：用户负责内存管理  
builder, network, parser = network_from_onnx_path("model.onnx")
try:
    engine = engine_from_network((builder, network, parser))
    with TrtRunner(engine) as runner:
        outputs = runner.infer(inputs)
finally:
    # 用户可能需要显式清理某些资源
    # （虽然Python的GC通常会处理，但GPU资源可能需要特别注意）
    pass
```

### 关键区别

| 特性 | 延迟加载器（类式） | 立即评估（函数式） |
|------|------------------|-------------------|
| **执行时机** | 调用时才执行 | 立即执行并返回结果 |
| **返回值** | 加载器对象 | 实际的TensorRT对象 |
| **内存管理** | 框架管理 | 用户负责 |
| **代码复杂度** | 需要理解延迟概念 | 直观易懂 |
| **适用场景** | 复杂管道、跨进程 | 简单脚本、调试 |

<!-- Polygraphy Test: Ignore End -->

### 本示例内容

在此示例中，我们将了解如何利用**函数式 API** 将 ONNX 模型转换为 TensorRT 网络，修改网络，构建启用 FP16 精度的 TensorRT 引擎，并运行推理。我们还将引擎保存到文件中，以了解如何再次加载它并运行推理。


## 运行示例

1.  安装先决条件
    *   确保已安装 TensorRT
    *   使用 `python3 -m pip install -r requirements.txt` 安装其他依赖项

2.  **[可选]** 在运行示例前检查模型：

    ```bash
    polygraphy inspect model identity.onnx
    ```

3.  运行构建和运行引擎的脚本：

    ```bash
    python3 build_and_run.py
    ```

4.  **[可选]** 检查示例构建的 TensorRT 引擎：

    ```bash
    polygraphy inspect model identity.engine
    ```

5.  运行加载先前构建的引擎，然后运行它的脚本：

    ```bash
    python3 load_and_run.py
    ```
