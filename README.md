# Olmo-3.1-UNIVA-VL

## Introduction
Olmo-3.1-UNIVA-VL is a large-scale Vision-Language Model (VLM) developed to integrate open-source Large Language Models with robust visual encoders. This model leverages the OLMo-3.1-32B-Instruct backbone, enhancing it with visual understanding capabilities through a high-resolution ViT and a specialized patch merging projector.

The model is designed to handle multi-modal tasks including image captioning and visual question answering, utilizing efficient training techniques such as Flash Attention 2 and DeepSpeed ZeRO-3.

## Architecture

The architecture follows a standard VLM paradigm: Vision Encoder $\rightarrow$ Patch Merger $\rightarrow$ LLM.

### LLM Backbone

- **Model**: Olmo-3.1-32B-Instruct
- **Type**: Causal Decoder-Only Transformer
- **Precision**: bfloat16
- **Optimization**: Utilizes flash_attention_2 for memory-efficient and fast context processing.
- **Role**: Acts as the reasoning engine, taking concatenated visual embeddings and text embeddings to generate autoregressive text responses.

### ViT

- **Model**: ViT-H-14-378-quickgelu (OpenCLIP)
- **Pretraining**: dfn5b (Data Filtering Network weights, known for high performance)
- **Patch Size**: 14x14
  
### Merger
The model employs a Patch Merger strategy to efficiently map visual tokens into the LLM's embedding space while reducing sequence length.

- **Mechanism**:
- **Spatial Pooling**: Uses a Conv2d layer with kernel_size=2 and stride=2. This reduces the visual grid size by half (e.g., merging 2x2 patches into 1), effectively reducing the token count by a factor of 4.
- **Flattening & Projection**: The pooled features are flattened and passed through a LayerNorm.
- **MLP**: A Multi-Layer Perceptron (Linear $\rightarrow$ GELU $\rightarrow$ Linear) projects the features to the LLM's hidden_size.
- **Positional Embeddings**: 2D Sin-Cos positional embeddings (get_2d_sincos_pos_embed) are added to the merged features to retain spatial awareness before entering the LLM.

## Image Preprocess

- **Smart Resizing**
1. Preserves the original Aspect Ratio of the image as much as possible.
2. Patch Alignment: Adjusts the width and height to be multiples of patch_step (28 pixels, from Patch size 14 $\times$ Merge 2). This prevents information loss or padding issues during the patch merging process.
3. Resolution Constraints: Scales the total pixel count to be between a minimum of 64 * 28 * 28 and a maximum of 256 * 28 * 28.

- **Normalization**
After resizing and tensor conversion, images are normalized using standard CLIP statistics.


## Task

- Image Desciption
- VQA
- OCR
  
## Train Data & Train Details

### Pretrain
The pretrain stage is divided into two stages. At stage 1, we freezed ViT and LLM, progressed full tuning in stage 2.

- 6,202,182 images, 340M text tokens
- COCO
- RedCAPS
- SynthText

### Fine-Tune
we progressed SFT in fine-tune stage

- 164,602 images, 3.8M text tokens
- GQA
- TextVQA
- DocVQA
- SynthText
