// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#include "gtest/gtest.h"

#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/util/thread_utils.h"
#include "test/test_environment.h"

std::unique_ptr<Ort::Env> ort_env;
void ortenv_setup() {
  OrtThreadingOptions tpo;
  ort_env.reset(new Ort::Env(&tpo, ORT_LOGGING_LEVEL_WARNING, "Default"));
}

static std::optional<std::string> GetEnvironmentVariable(const char* name) {
#if !defined(_WIN32)
  const char* value = std::getenv(name);
  if (value) {
    return std::string{value};
  }
  return std::nullopt;
#else
  // Windows builds complain about std::getenv, so use the suggested replacement _dupenv_s
  char* value;
  size_t length;
  errno_t err = _dupenv_s(&value, &length, name);
  if (err) {
    // just return nullopt if there's an error
    return std::nullopt;
  }
  auto free_value = gsl::finally([value]() { if (value) { GSL_SUPPRESS(r.10) free(value); } });
  if (!value) {
    return std::nullopt;
  }
  return std::string(value, length);
#endif
}

#ifdef USE_TENSORRT
// TensorRT will load/unload libraries as builder objects are created and torn down. This will happen for
// every single unit test, which leads to excessive test execution time due to that overhead.
// Nvidia suggests to keep a placeholder builder object around to avoid this.
#include "NvInfer.h"
class DummyLogger : public nvinfer1::ILogger {
 public:
  DummyLogger(Severity verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {}
};
DummyLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
auto const placeholder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
#endif

#define TEST_MAIN main

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_SIMULATOR || TARGET_OS_IOS
#undef TEST_MAIN
#define TEST_MAIN main_no_link_  // there is a UI test app for iOS.
#endif
#endif

int TEST_MAIN(int argc, char** argv) {
  int status = 0;

  ORT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    ortenv_setup();

    // allow verbose logging to be enabled by setting this environment variable to a numeric log level
    constexpr auto kLogLevelEnvironmentVariableName = "ORT_UNIT_TEST_MAIN_LOG_LEVEL";
    if (auto log_level_str = GetEnvironmentVariable(kLogLevelEnvironmentVariableName); log_level_str.has_value()) {
      const auto log_level = std::clamp(std::stoi(*log_level_str),
                                        static_cast<int>(ORT_LOGGING_LEVEL_VERBOSE),
                                        static_cast<int>(ORT_LOGGING_LEVEL_FATAL));
      std::cout << "Setting log level to " << log_level << "\n";
      ort_env->UpdateEnvWithCustomLogLevel(static_cast<OrtLoggingLevel>(log_level));
    }

    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  // TODO: Fix the C API issue
  ort_env.reset();  // If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  // make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
