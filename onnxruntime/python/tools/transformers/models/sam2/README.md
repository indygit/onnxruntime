# SAM2 ONNX Model Export

## Setup Environment
It is recommend to setup a machine with python 3.10, 3.11 or 3.12. Then install [PyTorch 2.4.1](https://pytorch.org/) and [Onnx Runtime 1.19.2].

### CPU Only
To install the CPU-only version of PyTorch and Onnx Runtime for exporting and running ONNX models, use the following commands:
```
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install onnxruntime onnx opencv-python matplotlib
```

### GPU
If your machine has an NVIDIA GPU, you can install the CUDA version of PyTorch and Onnx Runtime for exporting and running ONNX models:

```
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install onnxruntime-gpu onnx opencv-python matplotlib
```

onnxruntime-gpu requires CUDA 12.x, cuDNN 9.x, and other dependencies (such as MSVC Runtime on Windows). For more information, see the [installation guide](https://onnxruntime.ai/docs/install/#python-installs).

## Download Checkpoints

Clone the SAM2 git repository and download the checkpoints:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
python3 -m pip install -e .
cd checkpoints
sh ./download_ckpts.sh
```

On Windows, you can replace `sh ./download_ckpts.sh` with the following commands:
```bash
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt > sam2_hiera_tiny.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt > sam2_hiera_small.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt > sam2_hiera_base_plus.pt
curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt > sam2_hiera_large.pt
```

## Export ONNX
To export ONNX models, run the convert_to_onnx.py script and specify the segment-anything-2 directory created by the above git clone command:
```bash
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2
```

The exported ONNX models will be found in the sam2_onnx_models sub-directory. You can change the output directory using the `--output_dir` option.

If you want the model outputs multiple masks, append the `--multimask_output` option.

To see all parameters, run the following command:
```bash
python3 convert_to_onnx.py  -h
```

## Optimize ONNX

To optimize the onnx models for CPU with float32 data type:
```bash
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2 --optimize --dtype fp32
```

To optimize the onnx models for GPU with float16 data type:
```bash
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2 --optimize --dtype fp16 --use_gpu
```

Another option is to use optimizer.py like the following:
```
cd ../..
python optimizer.py --input models/sam2/sam2_onnx_models/sam2_hiera_large_image_encoder.onnx \
                    --output models/sam2/sam2_onnx_models/sam2_hiera_large_image_encoder_fp16_gpu.onnx \
                    --use_gpu --model_type sam2 --float16
```
The optimizer.py could be helpful when you have SAM2 onnx models that is exported by other tools.

## Run Demo

The exported ONNX models can run on a CPU. The demo will output sam2_demo.png.
```bash
curl https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg > truck.jpg
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2 --demo
```

It is able to run demo on optimized model as well. For example,
```bash
python3 convert_to_onnx.py  --sam2_dir path/to/segment-anything-2 --optimize --dtype fp16 --use_gpu --demo
```

## Limitations
- The exported image_decoder model does not support batch mode for now.