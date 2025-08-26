bool Model::build_engine_only_gpu() {
    // 我们也希望在build一个engine的时候就把一系列初始化全部做完，其中包括
    //  1. build一个engine
    //  2. 创建一个context
    //  3. 创建推理所用的stream
    //  4. 创建推理所需要的device空间
    // 这样，我们就可以在build结束以后，就可以直接推理了。这样的写法会比较干净
    auto builder       = shared_ptr<IBuilder>(createInferBuilder(*m_logger), destroy_trt_ptr<IBuilder>);
    auto network       = shared_ptr<INetworkDefinition>(builder->createNetworkV2(1), destroy_trt_ptr<INetworkDefinition>);
    auto config        = shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), destroy_trt_ptr<IBuilderConfig>);
    auto parser        = shared_ptr<IParser>(createParser(*network, *m_logger), destroy_trt_ptr<IParser>);

    config->setMaxWorkspaceSize(m_workspaceSize);
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED); //这里也可以设置为kDETAIL;

    // // if Ampere GPU
    // config->clearFlag(BuilderFlag::kTF32);

    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
        return false;
    }

    if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {
        config->setFlag(BuilderFlag::kFP16);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        if (m_params->det_model == model::detection_model::DETR) {
            for (int i = 0; i < network->getNbLayers(); i++) {
                auto layer = network->getLayer(i);
                if (layer->getType() == LayerType::kSOFTMAX || layer->getType() == LayerType::kNORMALIZATION) {
                    layer->setPrecision(DataType::kFLOAT);
                }
            }
        }
    } else if (builder->platformHasFastInt8() && m_params->prec == model::INT8) {
        config->setFlag(BuilderFlag::kFP16);
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        if (m_params->det_model == model::detection_model::DETR) {
            for (int i = 0; i < network->getNbLayers(); i++) {
                auto layer = network->getLayer(i);
                if (layer->getType() == LayerType::kSOFTMAX || layer->getType() == LayerType::kNORMALIZATION) {
                    layer->setPrecision(DataType::kFLOAT);
                }
            }
        }
    }

    std::string calib_table_name;
    switch(m_params->det_model) {
        case model::detection_model::YOLOV5:
            calib_table_name = "src/dev_pkg_det2d/calibration/calibration_table_yolov5.txt";
            break;
        case model::detection_model::YOLOV7:
            calib_table_name = "src/dev_pkg_det2d/calibration/calibration_table_yolov7.txt";
            break;
        case model::detection_model::YOLOV8:
            calib_table_name = "src/dev_pkg_det2d/calibration/calibration_table_yolov8.txt";
            break;
    }

    shared_ptr<Int8EntropyCalibrator> calibrator(new Int8EntropyCalibrator(
        m_params->batch_size, 
        (m_params == nullptr ? "src/dev_pkg_det2d/calibration/calibration_list_coco.txt" : m_params->calib_data_file.c_str()), 
        (m_params == nullptr ? calib_table_name.c_str() : m_params->calib_table_file.c_str()),
        m_params->img.c * m_params->img.h * m_params->img.w, 
        m_params->img.h, 
        m_params->img.w, 
        m_params->task));
    config->setInt8Calibrator(calibrator.get());

    auto engine        = shared_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config), destroy_trt_ptr<ICudaEngine>);
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);

    // 保存序列化后的engine
    save_plan(*plan);

    // 根据runtime初始化engine, context, 以及memory
    setup(plan->data(), plan->size());

    // 把优化前和优化后的各个层的信息打印出来
    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("After TensorRT optimization");
    print_network(*network, true);

    return true;
}