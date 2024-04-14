# quantization-from-scratch
This repository offers a PyTorch reimplementation of the quantization approach outlined in the white paper "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation." Please consult the paper for detailed information and empirical data, which you can access at [arXiv:2004.09602](https://arxiv.org/abs/2004.09602). The codebase includes the reimplementation of 8-bit inference methods, drawing on the strategies discussed in "8-bit Inference with TensorRT," presented at GTC 2017, available [here](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).

![Example Image](/images/quantization_flow.drawio.png "Quantization Flow")

Goal: (1) To provide a basic baseline for obtaining quantization parameters, enabling researchers to easily perform quantization in a PyTorch environment.
      (2) To provide a lightweight model that demonstrates reduced latency and memory size, utilizing the obtained quantization parameters.
      (3) To enable the creation of actual lightweight models by applying new quantization techniques which implemented with pytorch.

Most quantization tools, such as PyTorch quantization, ONNX Runtime quantization, and TensorRT, are designed for ease of use but at the cost of limited flexibility. They often do not allow for the integration of new technologies or the direct calculation of quantization parameters. This repository provides a solution by inserting parameters obtained through PyTorch into an ONNX Runtime model, thereby offering a higher degree of freedom in quantization. This approach enables the use of custom quantization strategies and the exploration of new quantization technologies.

Furthermore, this repository aims to provide researchers who wish to understand quantization itself with highly readable code. This ensures that the intricacies of quantization are accessible, promoting a deeper understanding of the process among those looking to delve into this area of study.

## Requirements

- PyTorch: 2.2.1
- CUDA: 12.1
- timm: 0.9.12
- ONNX: 1.16.0
- ONNX Runtime: 1.17.1




