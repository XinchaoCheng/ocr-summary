# ocr-summary


### 1.OCR
####1.1概述
    Optical Character Recognition（光学字符识别），指对文本资料的图像文件进行分析识别处理，获取文字及版面信息的过程。
OCR一般包含两步: 1. detection-->找到包含文字的区域(proposal); 2. classification-->识别区域中的文字。
####1.2好未来新版ocr
    检测部分使用ssd inception-v2模型，识别部分采用基于attention的分类模型。
    检测部分，ssd经过preprocess后，会将图片resize成（1,300,300,3）的shape。后处理包括
# Tensorflow: Adding a new op

## Steps
### 1. 注册 python api
**tensorflow/core/ops/math_ops.py**
```cc
REGISTER_OP("FilterscoreDecodeboxNms")
    .Input("locs: N * float32")
    .Input("confs: N * float32")
    .Input("priors: float32")
    .Output("output: float32")
    .Attr("N: int >= 1")
    .Attr("num_classes: int")
    .Attr("background_label_id: int") 
    .Attr("code_type: int")
    .Attr("confidence_threshold: float")
    .Attr("nms_threshold: float")
    .Attr("nms_topk: int = 400")
    .Attr("keep_topk: int = 200")
    .Attr("variance_encoded_in_target: bool = False");
```

**tensorflow/python/ops/math_ops.py**
```py
@tf_export("math.filterscore_decodebox_nms")
def _filterscore_decodebox_nms(
    locs,
    confs,
    priors,
    output,
    num_classes,
    background_label_id,
    code_type,
    confidence_threshold,
    nms_threshold,
    nms_topk,
    keep_topk,
    variance_encoded_in_target,
    name=None):
    return gen_math_ops.filterscore_decodebox_nms(
        locs,
        confs,
        priors,
        output,
        num_classes,
        background_label_id,
        code_type,
        confidence_threshold,
        nms_threshold,
        nms_topk,
        keep_topk,
        variance_encoded_in_target,
    )
```

### 2. kernel 实现

**tensorflow/core/kernels/filterscore_decodebox_nms_op.h**

```h
// Copyright [2018] <Cambricon>
#if CAMBRICON_MLU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/mlu/mlu_stream.h"
namespace tensorflow {
class LaunchParseBoxesOp {
 public:
  void launch(OpKernelContext* ctx,
    std::vector<Tensor> locs,
    std::vector<Tensor> confs, 
    int num, 
    Tensor* priors, 
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout,
    Tensor* output) {
    auto* stream = ctx->op_device_context()->mlu_stream();
    stream->ParseBoxes(ctx,
        locs,
        confs, 
        num, 
        priors, 
        num_classes,
        share_location,
        background_label_id,
        code_type,
        variance_encoded_in_target,
        confidence_threshold,
        nms_threshold,
        nms_topk,
        keep_topk,
        input_layout,
        output);
  }
};
class ParseBoxesOp : public OpKernel {
 public:
  explicit ParseBoxesOp(OpKernelConstruction* context) :OpKernel(
          context) {
    OP_REQUIRES_OK(context, context->GetAttr("num", &num_));
    OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
    OP_REQUIRES_OK(context, context->GetAttr("share_location", &share_location_));
    OP_REQUIRES_OK(context, context->GetAttr("background_label_id", &background_label_id_));
    OP_REQUIRES_OK(context, context->GetAttr("code_type", &code_type_));
    OP_REQUIRES_OK(context, context->GetAttr("variance_encoded_in_target", &variance_encoded_in_target_));
    OP_REQUIRES_OK(context, context->GetAttr("confidence_threshold", &confidence_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("nms_threshold", &nms_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("nms_topk", &nms_topk_));
    OP_REQUIRES_OK(context, context->GetAttr("keep_topk", &keep_topk_));
    OP_REQUIRES_OK(context, context->GetAttr("input_layout", &input_layout_));
  }
  ~ParseBoxesOp() {
  }
  void Compute(OpKernelContext* context) override {
    OpInputList locs;
    OP_REQUIRES_OK(context, context->input_list("locs", &locs));
    OpInputList confs;
    OP_REQUIRES_OK(context, context->input_list("confs", &confs));
    const Tensor* priors;
    OP_REQUIRES_OK(context, context->input("priors", &priors));
    //const Tensor& priors = context->input(2 * num_);
    TensorShape output_shape = TensorShape(
        {1, keep_topk_, 1, 6});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    std::vector<Tensor> locs_tensors, confs_tensors;
    const int N = locs.size();
    for (int i = 0; i < N; i++) {
        locs_tensors.push_back(locs[i]);
        confs_tensors.push_back(confs[i]);
    }
    launcher_.launch(context,
        locs_tensors,
        confs_tensors, 
        num_, 
        const_cast<Tensor*>(priors), 
        num_classes_,
        share_location_,
        background_label_id_,
        code_type_,
        variance_encoded_in_target_,
        confidence_threshold_,
        nms_threshold_,
        nms_topk_,
        keep_topk_,
        input_layout_,
        output);
  }
 private:
  LaunchParseBoxesOp launcher_;
  int num_;
  int num_classes_;
  bool share_location_;
  int background_label_id_;
  int code_type_;
  bool variance_encoded_in_target_;
  float confidence_threshold_;
  float nms_threshold_;
  int nms_topk_;
  int keep_topk_;
  int input_layout_;
};
}  // namespace tensorflow
#endif  // CAMBRICON_MLU
```

### 3. 注册 filerscore_decodebox_nms 的 mlu 实现
**tensorflow/core/kernels/matmul_op.cc**
```cc
// Line 37
#include "filterscore_decodebox_nms_op.h"

//  Line 648
REGISTER_KERNEL_BUILDER(Name("FilterscoreDecodeboxNms").Device(DEVICE_MLU), FilterscoreDecodeboxNmsOp);
```

**tensorflow/core/kernels/BUILD**
```build
# Line 2961
    hdrs = ["matmul_op.h",
        "mlp_mlu.h",
        #"mlp_and_relu_mlu.h",
        "fix8_mlp_op_mlu.h",
        "filterscore_decodebox_nms_op.h",  # Add a dep
    ],
```

### 4. mlu 实现
**tensorflow/stream_executor/mlu/lib_ops/mlu_filterscore_decodebox_nms_op.cc**

```cc
#if CAMBRICON_MLU
#include <iostream>
#include <vector>
#include <string>
#include "tensorflow/stream_executor/mlu/mlu_lib_common.h"
#include "tensorflow/stream_executor/mlu/mlu_lib_math_ops.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
namespace stream_executor {
namespace mlu {
namespace lib {
void CreateParseBoxesParam(MLUSsdDetectionOpParam** param,
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout) {
    MLUDataOrder input_layout_ = CNML_NHWC;
    MLU_LIB_CALL(cnmlCreateSsdDetectionOpParam(param,
        num_classes, share_location,
        background_label_id, code_type,
        variance_encoded_in_target, confidence_threshold,
        nms_threshold, nms_topk,
        keep_topk, input_layout_));
}
void DestroyParseBoxesParam(MLUSsdDetectionOpParam** param) {
  MLU_LIB_CALL(cnmlDestroySsdDetectionOpParam(param));
}
void CreateParseBoxesOp(MLUBaseOp** op,
    std::vector<Tensor> locs, std::vector<Tensor> confs, int num, Tensor* priors, Tensor* output, MLUSsdDetectionOpParam* param) {
    std::vector<MLUTensor*> locs_mlu_tensor, confs_mlu_tensor;
    const int N = locs.size();
    for (int i = 0; i < N; i++) {
        locs_mlu_tensor.push_back(locs[i].mlu_tensor());
        confs_mlu_tensor.push_back(confs[i].mlu_tensor());
    }
    MLU_LIB_CALL(cnmlCreateSsdDetectionOpV2(op,
        locs_mlu_tensor.data(), confs_mlu_tensor.data(), locs_mlu_tensor.size(), priors->mlu_tensor(), 
        output->mlu_tensor(), 
        param));
}
}  // namespace lib
}  // namespace mlu
}  // namespace stream_executor
#endif  // CAMBRICON_MLU
```

**tensorflow/stream_executor/mlu/ops/filterscore_decodebox_nms_op.cc**

```cc
/*Copyright 2018 Cambricon*/
#if CAMBRICON_MLU
#include <vector>
#include "tensorflow/stream_executor/mlu/mlu_lib_nn_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_stream.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
namespace perftools {
namespace gputools {
// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(b/77980417): Remove this once we've
// completed the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]
}  // namespace gputools
}  // namespace perftools
namespace stream_executor {
namespace mlu {
MLUStatus MLUStream::ParseBoxes(std::vector<Tensor> locs, 
    std::vector<Tensor> confs, 
    int num, 
    Tensor* priors, 
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout,
    Tensor* output) {
    const int N = locs.size();
    // prepare tensor descriptors
    for (int i = 0; i < N; i++) {
        CreateMLUTensor(&(locs[i]));
        CreateMLUTensor(&(confs[i]));
    }
    CreateMLUTensor(priors);
    CreateMLUTensor(output);
    // setup conv operator
    MLUSsdDetectionOpParam* parse_box_param;
    lib::CreateParseBoxesParam(&parse_box_param,
        num_classes,
        share_location,
        background_label_id,
        code_type,
        variance_encoded_in_target,
        confidence_threshold,
        nms_threshold,
        nms_topk,
        keep_topk,
        input_layout);
    MLUBaseOp* parse_box_op_ptr;
    lib::CreateParseBoxesOp(&parse_box_op_ptr,
        locs, 
        confs,  
        num, 
        priors, 
        output, 
        parse_box_param);
    CompileOp(parse_box_op_ptr);
    // Malloc
    const int N1 = locs.size();
    for (int i = 0; i < N1; i++) {
        MLUMalloc(&(locs[i]));
        MLUMalloc(&(confs[i]));
    }
    MLUMalloc(priors);
    MLUMalloc(output);
    // Copy
    const int N2 = locs.size();
    for (int i = 0; i < N2; i++) {
        MemcpyFromCPUToMLU(&(locs[i]));
        MemcpyFromCPUToMLU(&(confs[i]));
    }
    MemcpyFromCPUToMLU(priors);
    //MemcpyFromCPUToMLU(output);
    std::vector<Tensor*> input_tensors;
    const int N3 = locs.size();
    for (int i = 0; i < N3; i++) {
        input_tensors.push_back(&(locs[i]));
    }
    for (int i = 0; i < N3; i++) {
        input_tensors.push_back(&(confs[i]));
    }
    input_tensors.push_back(priors);
    ComputeOp(parse_box_op_ptr, CONV, input_tensors, {output});
    DestroyOp(&parse_box_op_ptr);
    lib::DestroyParseBoxesParam(&parse_box_param);
    return MLU_STATUS_SUCCESS;
}
MLUStatus MLUStream::ParseBoxes(OpKernelContext* ctx,
    std::vector<Tensor> locs,
    std::vector<Tensor> confs, 
    int num, 
    Tensor* priors, 
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout,
    Tensor* output) {
  OrganizeFusionContext(ctx);
  MLUStatus status = ParseBoxes(locs, confs, 
    num, 
    priors, 
    num_classes,
    share_location,
    background_label_id,
    code_type,
    variance_encoded_in_target,
    confidence_threshold,
    nms_threshold,
    nms_topk,
    keep_topk,
    input_layout,
    output);
  return status;
}
}  // namespace mlu
}  // namespace stream_executor
#endif  // CAMBRICON_MLU
```

**tensorflow/stream_executor/mlu/mlu_lib_nn_ops.h**
```h
// Line 101
void CreateParseBoxesParam(MLUSsdDetectionOpParam** param,
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout);
void DestroyParseBoxesParam(MLUSsdDetectionOpParam** param);
void CreateParseBoxesOp(MLUBaseOp** op,
    std::vector<Tensor> locs, std::vector<Tensor> confs, int num, Tensor* priors, Tensor* output, MLUSsdDetectionOpParam* param);
```

**tensorflow/stream_executor/mlu/mlu_stream.h**
```h
// Line 409
  MLUStatus ParseBoxes(std::vector<Tensor> locs, 
    std::vector<Tensor> confs, 
    int num, 
    Tensor* priors, 
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout,
    Tensor* output);
  MLUStatus ParseBoxes(OpKernelContext* ctx,
    std::vector<Tensor> locs,
    std::vector<Tensor> confs, 
    int num, 
    Tensor* priors, 
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    float confidence_threshold,
    float nms_threshold,
    int nms_topk,
    int keep_topk,
    int input_layout,
    Tensor* output);
```

**tensorflow/core/platform/mlu.h**
```h
typedef struct cnmlSsdDetectionOpParam MLUSsdDetectionOpParam;
typedef cnmlDataOrder_t MLUDataOrder;
```
