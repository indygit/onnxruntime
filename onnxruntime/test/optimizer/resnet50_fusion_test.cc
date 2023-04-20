// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "iconv.h"
#include <vector>
#include "core/graph/graph.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_add_act_fusion.h"
#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
#define ORT_RUN_EXTERNAL_ONNX_TESTS
#define MLAS_F16VEC_INTRINSICS_SUPPORTED
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && !defined(DISABLE_CONTRIB_OPS)

class ResNet50FusionTests : public ::testing::Test {
 protected:
  ResNet50FusionTests() : logger(DefaultLoggingManager().CreateLogger("ResNet50FusionTest")) {
  }
  std::unique_ptr<logging::Logger> logger;




};
#if defined(ORT_RUN_EXTERNAL_ONNX_TESTS)
TEST_F(ResNet50FusionTests, FuseConvAddRelu) {
  std::basic_string<ORTCHAR_T>  fp32_model_path = ORT_TSTR("../models/opset10/Resnet50_Fusion_Testing/resnet50.onnx");
  std::shared_ptr<Model> fp32_model;
  std::basic_string<ORTCHAR_T> fp16_model_path = ORT_TSTR("../models/opset10/Resnet50_Fusion_Testing_fp16/resnet50.fp16.onnx");
  std::shared_ptr<Model> fp16_model;
  if(Model::Load(fp32_model_path, fp32_model, nullptr, *logger)!=Status::OK()){
    GTEST_SKIP() << "Failed to load model: " << fp32_model_path;
  }
  if (Model::Load(fp16_model_path, fp16_model, nullptr, *logger) != Status::OK()) {
    GTEST_SKIP() << "Failed to load model: " << fp16_model_path;
  }
//  ASSERT_STATUS_OK(Model::Load(fp32_model_path, fp32_model, nullptr, *logger));
  Graph& fp32_graph = fp32_model->MainGraph();
  for (auto& node : fp32_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  Graph& fp16_graph = fp16_model->MainGraph();
  for (auto& node : fp16_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  //  std::cout << "-------Op Counts Before Fusion---------" << std::endl;
  std::map<std::string, int> fp32_op_count = CountOpsInGraph(fp32_graph);
  std::map<std::string, int> fp16_op_count = CountOpsInGraph(fp16_graph);
  for (auto& op : fp32_op_count) {
    //    std::cout << op.first << " " << op.second << std::endl;
    ASSERT_EQ(op.second, fp16_op_count[op.first]);
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr_32{5};
  ASSERT_STATUS_OK(graph_transformation_mgr_32.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr_32.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr_32.ApplyTransformers(fp32_graph, TransformerLevel::Level3, *logger));
  ASSERT_STATUS_OK(Model::Save(*fp32_model, "resnet50_fused.onnx"));

  onnxruntime::GraphTransformerManager graph_transformation_mgr_16{5};
  ASSERT_STATUS_OK(graph_transformation_mgr_16.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr_16.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr_16.ApplyTransformers(fp16_graph, TransformerLevel::Level3, *logger));
  ASSERT_STATUS_OK(Model::Save(*fp16_model, "resnet50_fp16_fused.onnx"));
  //  std::cout << "-------Op Counts After Fusion---------" << std::endl;
  fp32_op_count = CountOpsInGraph(fp32_graph);
  fp16_op_count = CountOpsInGraph(fp16_graph);
  for (auto& op : fp32_op_count) {
    //    std::cout << op.first << " " << op.second << std::endl;
    ASSERT_EQ(op.second, fp16_op_count[op.first]);
  }
}

#endif  // ORT_RUN_EXTERNAL_ONNX_TESTS
#endif
}  // namespace test
}  // namespace onnxruntime