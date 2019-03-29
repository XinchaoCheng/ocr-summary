# ocr-summary


### 1.OCR

#### 1.1概述

Optical Character Recognition（光学字符识别），指对文本资料的图像文件进行分析识别处理，获取文字及版面信息的过程。
OCR一般包含两步: 1. detection-->找到包含文字的区域(proposal); 2. classification-->识别区域中的文字。

#### 1.2好未来新版ocr
   检测部分使用ssd inception-v2模型，识别部分采用基于attention的分类模型。检测部分，ssd经过preprocess后，会将图片resize成（1,300,300,3）的shape送入卷积层，经过nms等处理后输出100个框的坐标和对应分数；后处理包括文本结构识别等一系列操作，生成行检测信息送入识别网络。
   
   识别网络，前处理包括对输入逐行进行灰度和resize处理，对label进行编码操作，以及使用滑窗处理逐行图片的过程。前处理后输入变为（n,32,32,1）的形式；推理完成后output为（n，36），prob为（n,36,3677）。默认每行最多有36个字，3为终止符号。prob的最大值对应的index即为output的值。最后根据编码输出对应的字符，完成单张图的识别。
   
### 2.不同版本OCR优化过程
#### 2.1版本1
第一个版本采用了两个框架，检测网络使用了caffe，识别网络使用tensorflow。其中检测部分使用了基于ssd的VGG16模型，识别网络则使用了MobileNet v2结构。

由于网络输入图片的大小不一，有很多尺寸，使用原代码在GPU上运行时，检测部分每次推理是将整张图片当做输入。这样做虽然可以减小延时，但由于输入图片较大，占用的GPU显存资源较多，因此在多路时无法大幅度提高吞吐（qps）。为了减小内存的占用，同时适配MLU100板卡不能多尺度的问题（虽然每个尺度可以生成一个指令catch集，但因为图片大小非常不同，无法为每个尺度都生成一个指令集），使用了滑窗方法去分割图片。

滑窗方式就是利用一个固定大小的window以滑动的方式去截取数据。滑动过程中，需要设置重叠值，即overlap数值，保证不因为滑窗而丢失信息。当window超出图片范围时，未覆盖的地方填零，这样可将不同大小的图片分割成一系列相同大小的窗口，即检测网络每次推理过程输入的shape均为固定值，无需生成新的指令catch集。当将一张图片分成几个相同大小的部分进行推理后，需将这几个部分结果重新拼接起来，一块送入识别网络。这就需要记录每个窗口的相对位置和scale尺度，然后将四个坐标值从窗口的相对位置转换到完整图片的绝对位置。具体代码可参考public_fix_shape.py文档，代码部分截图如下所示。

```cc
def detect_box(input_image, input_net, max_accept_size, over_lap_rate, input_scale=1):
    img_height = int(input_image.shape[0] * input_scale)
    img_width  = int(input_image.shape[1] * input_scale)
    if SHOW_DEBUG_INFO:
        print "[public.detect_box]: orginal size: h {},w {}".format(img_height, img_width)
    img_size = img_height * img_width

    # preprocess the orginal image data
    transformer = caffe.io.Transformer({'data': (1, 3, img_height, img_width)})
    transformer.set_transpose('data', (2, 0, 1))
    if device_type!=2:
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel

    transformed_image = transformer.preprocess('data', input_image)

    box_list = []
    unit_height = 400 # this value should same with the 3th dimension of input data in caffe prototxt
    unit_width  = 400 # this value should same with the 4th dimension of input data in caffe prototxt
    x_overlap = 30
    y_overlap = 30
    if device_type!=2:
        data_paraller = 1
    else:
        data_paraller = 3*int(os.environ["MLU_CAFFE_DP"]) # first value should same with the first dimension of input data in caffe prototxt
    # untill now, cpu and gpu can only set data_paraller to 1
    if True:
        y_min = 0
        while y_min < (img_height):
            x_min = 0
            y_max = min(y_min + unit_height + y_overlap, img_height)
            while x_min < (img_width):
                x_max = min(x_min + unit_width + x_overlap, img_width)
                box_list.append([x_min, y_min, x_max, y_max])
                x_min += unit_width
            y_min += unit_height

    # forward
    box_len = len(box_list)
    detections = np.array([])
    roi_img_list = np.array([])
    x_offset_list = []
    y_offset_list = []
    x_scale_list = []
    y_scale_list = []
    separate_location_list=[]
    separate_location_list.append(0)
    count = 0
    remainder = box_len % data_paraller
    if remainder != 0:
        if box_len>data_paraller:
            for i in range(remainder):
                box_list.append([0, 0, 0, 0])
        else:
            for i in range(data_paraller-remainder):
                box_list.append([0, 0, 0, 0])
    if box_len>0:
        inv_img_width = 1.0 / img_width
        inv_img_height = 1.0 / img_height
        for box in box_list:
            if box[2]-box[0]<(unit_width+x_overlap) or box[3]-box[1]<(unit_height+y_overlap):
                if box[2]<(unit_height+y_overlap) or box[3]<(unit_width+x_overlap):
                    roi_img = np.zeros( (3,unit_height+y_overlap,unit_width+x_overlap) )
                    bias_x = box[2]-box[0]
                    bias_y = box[3]-box[1]

                    roi_img[:,0:bias_y,0:bias_x] = copy.deepcopy(transformed_image[:,box[1]:box[3],box[0]:box[2]])
                    x_offset = box[0] * inv_img_width
                    y_offset = box[1] * inv_img_height
                    x_scale = (unit_width+x_overlap) * inv_img_width
                    y_scale = (unit_height+y_overlap) * inv_img_height

                else:
                    bias_x = box[0]
                    bias_y = box[1]
                    box[0] = box[2]-unit_width-x_overlap
                    box[1] = box[3]-unit_height-y_overlap
                    roi_img = copy.deepcopy(transformed_image[:,box[1]:box[3],box[0]:box[2]]) #box contains four numbers means the location of an img
                    roi_img[:,0:bias_y-box[1],:] = 0
                    roi_img[:,:,0:bias_x-box[0]] = 0

                    x_offset = box[0] * inv_img_width
                    y_offset = box[1] * inv_img_height
                    x_scale = (unit_width+x_overlap) * inv_img_width
                    y_scale = (unit_height+y_overlap) * inv_img_height
            else:
                roi_img = transformed_image[:,box[1]:box[3],box[0]:box[2]] #box contains four numbers means the location of an img

                x_offset = box[0] * inv_img_width
                y_offset = box[1] * inv_img_height
                x_scale = (box[2] - box[0]) * inv_img_width
                y_scale = (box[3] - box[1]) * inv_img_height
            if len(roi_img_list)==0:
                roi_img_list = roi_img.copy()
            else:
                roi_img_list = np.concatenate((roi_img_list, roi_img), axis=0)           
            x_offset_list.append(x_offset)
            y_offset_list.append(y_offset)
            x_scale_list.append(x_scale)
            y_scale_list.append(y_scale)
            count +=1
            if count == data_paraller:
                count = 0
                roi_img_list = np.reshape(roi_img_list,(data_paraller,3,unit_height+y_overlap,unit_width+x_overlap))
                input_net.blobs['data'].data[...] = roi_img_list
                sub_detections_ = input_net.forward()['detection_out']

                if device_type!=2:
                    sub_detections=sub_detections_
                else:
                    for i in range(data_paraller):
                        temp = sub_detections_[np.where(sub_detections_[i:i+1,:,:,4]>0)].shape[0] + separate_location_list[i]
                        separate_location_list.append(temp)
                    
                    sub_detections_=sub_detections_[np.where(sub_detections_[:,:,:,4]>0)]
                    sub_detections_=sub_detections_[np.newaxis,np.newaxis, :]  
                        
                    new_shape=list(sub_detections_.shape)
                    new_shape[-1]=7
                    
                    new_detections=np.zeros(new_shape)
                    new_detections[0,0,:,0]=0
                    new_detections[0,0,:,3:7]=sub_detections_[0,0,:,0:4]
                    new_detections[0,0,:,1]=sub_detections_[0,0,:,5]
                    new_detections[0,0,:,2]=sub_detections_[0,0,:,4]
                            
                    sub_detections=new_detections

                if device_type!=2:
                    for i in range(data_paraller):
                        sub_detections[i, 0, :, 3] = sub_detections[i, 0, :, 3]*x_scale_list[i] + x_offset_list[i]
                        sub_detections[i, 0, :, 4] = sub_detections[i, 0, :, 4]*y_scale_list[i] + y_offset_list[i]
                        sub_detections[i, 0, :, 5] = sub_detections[i, 0, :, 5]*x_scale_list[i] + x_offset_list[i]
                        sub_detections[i, 0, :, 6] = sub_detections[i, 0, :, 6]*y_scale_list[i] + y_offset_list[i]
                else:
                    for i in range(data_paraller):
                        sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 3] = sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 3]*x_scale_list[i] + x_offset_list[i]
                        sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 4] = sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 4]*y_scale_list[i] + y_offset_list[i]
                        sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 5] = sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 5]*x_scale_list[i] + x_offset_list[i]
                        sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 6] = sub_detections[0, 0, separate_location_list[i]:separate_location_list[i+1], 6]*y_scale_list[i] + y_offset_list[i]

                if len(detections)==0:
                    detections = sub_detections.copy()
                else:
                    detections = np.concatenate((detections, sub_detections), axis=2)
                x_offset_list = []
                y_offset_list = []
                x_scale_list = []
                y_scale_list = []
                separate_location_list = []
                separate_location_list.append(0)
                roi_img_list = np.array([])
    else:
        print("Wrong!! box_list cannot be empty")
    return detections
```

该滑窗的实现代码同时适配CPU，MLU和GPU版本，同时在MLU上还支持数据并行度大于一的情况，即检测网络一次推理过程同时处理多个窗口，进一步减小延时。注意，当检测部分（基于caffe框架）MLU数据并行度大于一时，并行度的值需与模型prototxt中input的第一个dimension值相同；最后两个维度需与滑窗大小值（unit_width+x_overlap）设置相同。经过滑窗（430*430，不同尺寸会有不同精度）处理后，单尺度下的错误率从14.6%下降到12.54%；在1080Ti上，1000张测试图片平均端到端延时从0.1s上升到0.16s；在MLUd3卡上平均端到端时间为0.3442s。当使用变换的scale时（将图片的长边transform为2000，短边缩放相同比例），MLU上的错误率约为10.15%；GPU上为9.91%，不使用滑窗时为9.18%

同理，在识别网络时也可采用固定输入shape，即将batch size设为固定大小，最后一次识别推理时，补齐剩余的部分，以适配MLU的固定指令集。经过测试，不同尺寸的batch size会影响延时，在此项目中当设为128时性能最优。

关于如何确定模型的数据并行度和模型并行度，可使用find_best_mp.py脚本，绘制折线图来寻找，经过确定单线程时caffe最优的mp为16，tensorflow最优的mp为8。

以上为初步优化的过程，下面将阐述针对该模型深度优化的方案与结果。好未来对该版本的要求为单尺度下时延500ms以下，吞吐qps大于5。但是，MLU上的单线程延时大约为340ms，qps为2.9；3线程时延时已超过500ms，经过各种调试均未能同时满足时延吞吐要求。此时，一个方法是降低CPU利用率，具体可使用pycharm的profiling工具进行性能分析；将pycharm与服务器连接，在MLU上运行时，该工具会给出每个调用函数的执行次数与时间占比。经过分析得到两个模型的推理总共占据了约88%的时间，剩余中较大的占比为nms，值得注意的是此处nms仅去除掉了很少的冗余框却占据了很多的cpu利用率，因此可通过减小它的阈值（从0.5变为0.1）来去掉更多的框。经过测试，此方法还可以提升精度，单尺度从12.54%降低到12.38%；变换的scale时，MLU错误率从10.15%降低到8.04%（驱动版本为v3.2.1，不同driver版本也会得到不同的精度数据）。此外还可以根据得到了占比去进行进一步优化，优先优化掉CPU利用率高的操作。

虽然降低阈值可提升准确率，但延时吞吐仍不满足。结合MLU的特性，我们可知MLU有四个DDR通道，每个DDR通道控制两个cluster即8个核。每个DDR通道可访问所有的32个核，但执行效率不同，当且仅当DDR控制它紧凑的8个核时，DDR调用的效率最高。所以当多线程时，效率最高的应为使用4个线程，每个线程控制指定的DDR通道，DDR仅控制它紧凑的8个核，即将完整的MLU拆分为4份，各部分互不影响（最终效果为，使用单通道8个核与使用4通道32个核延时需相同，qps为4倍关系）。在caffe，tensorflow框架中均给出set_channel接口，但默认当MP大于1时，指定通道的方法会失效。因此多线程与单线程性能不成倍数关系。通过与sopa和驱动组的沟通，修该了sopa代码，得到了符合要求的cnml，cnrt库，最终将延时控制在400ms左右（由于去除了默认的优化环节，单线程延时比默认通道方法会增加），吞吐达到大约10qps。该方案预计会以环境变量的方式合入后续的版本中。以上均是在V7发布包的基础上，V8会有较大变动。

#### 2.2版本2

由于版本1中识别部分使用了逐个检测，虽然准确率达到要求，但有时会造成前后语义不符。版本2中识别部分替换为行检测，使用了attention机制，能够获取到前后信息，单尺度下提升了精度。

检测部分仍使用ssd VGG16模型，因此继续使用滑窗方式处理。识别部分在MLU上无法直接跑通，查询pb知含有不支持的数据类型（INT64）；编写脚本pb_change_device.py将不支持的数据类型修改为INT32，可顺利跑通。在v7.1版本中，无法跑融合模式，经测试v.7.3版本可跑通，因此不同版本也会造成不同的影响。

融合模式跑通后发现有很多不支持的算子，会造成很多分段，由于带宽限制，大大影响了延时。通过tensorboard查看网络结构却得不到device的相关信息，解决方法是在pb文件中对每个算子增加device相关信息。具体步骤如下：
1)将打印信息TF_CPP_MIN_MLU_LOG_LEVEL环境变量设置为1，跑通一次pb模型并将输出结果保存到txt文件中。
2)编写脚本，从上述txt文件中截取node,op和device信息，保存到txt2中。
3)编写新脚本，根据txt2中的信息，将每个node的device属性添加到pb文件中。
4)利用debug脚本，导出summary log信息，然后显示在tensorboard中。
5)根据device信息，查看graph，将MLU上可执行但造成分段过多且计算量小的层放在CPU上运行，减小分段的数量。
注：以上脚本在下文中具体介绍中。
初步优化后，精度提升0.8%；但因为CPU利用率很高造成瓶颈，导致MLU的利用率不足。四路下延时大约在1s，qps为3.9左右。

#### 2.3版本3
由于caffe在阿里云上部署困难，好未来放弃了caffe框架，将整个检测部分放在了tensorflow上。起初这一版也进行了划窗处理和识别部分固定shape处理，经过分析将前处理全部放在CPU运行后，取消了检测部分的滑窗处理。在CPU利用率略微上升的情况下，检测部分延时从超过200ms下降为100ms。添加了SSD后处理大算子和fp16——firstconv算子后，进一步将延时降低到约47ms。


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
