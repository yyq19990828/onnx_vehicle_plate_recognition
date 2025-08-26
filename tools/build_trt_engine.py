#!/usr/bin/env python3
"""
tools/build_trt_engine.py

Build a TensorRT engine from an ONNX model using the TensorRT Python API.

Features:
- Load ONNX model bytes and parse with ONNX parser
- Configure builder and builder config (FP16, workspace, max_batch)
- Optionally load TRT plugin libraries
- Optionally force specific layer names to run in FP32
- Serialize and save engine to disk

Usage:
  python tools/build_trt_engine.py --onnx model.onnx --out engine.plan --fp16 --workspace 4096

Note: Requires TensorRT Python bindings available in the environment.
"""
import argparse
import os
import sys
from typing import List

try:
    import tensorrt as trt
except Exception as e:
    print("TensorRT import failed:", e)
    print("Ensure TensorRT is installed and the Python bindings are available.")
    sys.exit(1)


def build_engine(onnx_path: str, out_path: str, fp16: bool, workspace_mb: int, plugins: List[str], force_fp32_layers: List[str], opt_shapes: List[str] = None):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load plugins if provided
    for p in plugins:
        if os.path.exists(p):
            try:
                trt.init_libnvinfer_plugins(TRT_LOGGER, "")
                # dlopen happens automatically when TRT loads plugin registry
                print(f"Attempting to load plugin library: {p}")
                # Try to load via ctypes to ensure symbols are available
                import ctypes

                ctypes.cdll.LoadLibrary(p)
            except Exception as ex:
                print(f"Failed to load plugin {p}: {ex}")
        else:
            print(f"Plugin file not found: {p}")

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()

    if not parser.parse(model_bytes):
        print("Failed to parse ONNX model:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return False

    builder.max_batch_size = 1
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * 1024 * 1024

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Enabled FP16 builder flag")
        else:
            print("Warning: platform reports no fast FP16; enabling flag anyway")
            config.set_flag(trt.BuilderFlag.FP16)

    # Optionally force certain layers to run in FP32
    if force_fp32_layers:
        # Iterate layers and set precision
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            lname = layer.name or ""
            if any(name in lname for name in force_fp32_layers):
                try:
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(0, trt.DataType.FLOAT)
                    print(f"Forcing layer to FP32: {lname}")
                except Exception:
                    print(f"Could not force layer precision for: {lname}")

    # Handle dynamic shapes: build optimization profile if inputs are dynamic
    def has_dynamic_inputs(net):
        for i in range(net.num_inputs):
            inp = net.get_input(i)
            if any(d == -1 for d in inp.shape):
                return True
        return False

    # If user provided opt_shapes, parse and add them
    if opt_shapes:
        for spec in opt_shapes:
            # Expected format: input_name:min1xmin2x...,opt1xopt2x...,max1xmax2x...
            try:
                inp_name, shapes = spec.split(":", 1)
                min_s, opt_s, max_s = shapes.split(",")
                def parse_shape(s):
                    return tuple(int(x) for x in s.split("x") if x)
                profile = builder.create_optimization_profile()
                profile.set_shape(inp_name, parse_shape(min_s), parse_shape(opt_s), parse_shape(max_s))
                config.add_optimization_profile(profile)
                print(f"Added user profile for {inp_name}: min={min_s} opt={opt_s} max={max_s}")
            except Exception as ex:
                print(f"Failed to parse opt-shape '{spec}': {ex}")
    else:
        # If inputs are dynamic, attempt to build a default optimization profile
        if has_dynamic_inputs(network):
            print("Detected dynamic input shapes. Creating a default optimization profile.")
            profile = builder.create_optimization_profile()
            # For each network input, derive a reasonable profile from its shape
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                name = inp.name
                shape = list(inp.shape)
                # Replace -1 with 1 for min, keep a moderate max (e.g., 1 -> 8x)
                min_shape = [1 if d == -1 else d for d in shape]
                opt_shape = [s if s != -1 else max(1, min(8, s if s > 0 else 1)) for s in shape]
                max_shape = [s if s != -1 else max(1, s if s > 0 else 8) for s in shape]
                # Fallback: set opt and max to min if unknown
                try:
                    profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
                    print(f"Added profile for input {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                except Exception as ex:
                    print(f"Failed to set profile for {name}: {ex}")
            try:
                config.add_optimization_profile(profile)
            except Exception as ex:
                print(f"Failed to add optimization profile: {ex}")

    # Build serialized engine (recommended API)
    print("Building serialized engine...")
    try:
        # Use build_serialized_network which returns serialized engine bytes
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        print("Engine build threw exception:", e)
        serialized = None

    if serialized is None:
        print("Failed to build engine")
        return False

    # Write serialized bytes
    with open(out_path, "wb") as f:
        f.write(serialized)
    print(f"Saved engine to: {out_path}")
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--out", required=True, help="Output engine path (.plan)")
    p.add_argument("--fp16", action="store_true", help="Enable FP16")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace size in MB")
    p.add_argument("--plugin", action="append", default=[], help="Path to TRT plugin .so (can be repeated)")
    p.add_argument("--force-fp32-layer", action="append", default=[], help="Substring of layer names to force to FP32 (can repeat)")
    p.add_argument("--opt-shape", action="append", default=[], help=(
        "Optimization profile shape specifier. Format: input_name:min1xmin2x...,opt1xopt2x...,max1xmax2x... "
        "(can be repeated for multiple inputs). If not provided, a default profile will be created for dynamic inputs."))
    return p.parse_args()


def main():
    args = parse_args()
    ok = build_engine(args.onnx, args.out, args.fp16, args.workspace, args.plugin, args.force_fp32_layer, opt_shapes=args.opt_shape)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
