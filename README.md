# Visual-Transformers (ViT) from Scratch with PyTorch

This repository contains a modular and clean Vision Transformer (ViT) implementation built from scratch using PyTorch.
The goal is to break down the architecture step by step from patch embeddings to transformer blocks, attention mechanisms, and classification head while keeping the code easy to comprehend, follow and extend.

## In Features

1. Patch Embedding Layer -> Splits image into fixed size patches and projects them into an embedding space.

2. Multi-Head Self Attention (MHSA) -> Implements the scaled dot-product attention with multiple heads.

3. Transformer Blocks -> LayerNorm, residual connections, MHSA, and an MLP feed-forward network.

4. Class Token + Positional Embeddings -> For sequence modeling and classification tasks.

5. Custom weight & Bias Initialization -> Xavier/He initialization for stable training.

6. Attention Visualization -> Provided optional paramenter to return attention maps from each block.

7. Fully Modular → Components are written as separate classes so you can easily extend or modify.

## Architecture Overview

The ViT model follows the standard architecture:

1. Patch Embedding: Image -> non-overlapping patches (using conv2d) -> linear projection.

2. Positional Encoding: Learnable position embeddings added to patch embeddings.

3. [cls] Token: Added to represent the whole sequence for classification.

4. Stack of Transformer Blocks:

    4.1. LayerNorm -> Multi-Head Self Attention (MHSA) -> Residual

    4.2. LayerNorm -> MLP -> Residual

5. Classification Head: Linear layer on the final [cls] token output.

## Code Structure
.
├── ViT.py           # Main implementation (PatchEmbedding, MHSA, TransformerBlock, ViT)

├── rough notebooks           # Notebooks used during the development and testing process of each modules for convinient implementation of ViT

├── test

    ├── mnist_vit_test.py           # testing ViT model with MNIST dataset
  
├── README.md        # Project documentation

└── requirements.txt # Dependencies (PyTorch, etc.)

## Getting Started
1. Clone the Repository
'''bash
git clone https://github.com/<your-username>/vision-transformer.git
cd vision-transformer
'''

3. Install Dependencies
'''bash
pip install torch torchvision
'''


Creating a virtual environment before installing would be suggested

3. Usage
        ''' CODE
        
        # Example: Training on MNIST-like data (28x28 grayscale images)
       
        import torch
        from ViT import ViT
        model = ViT(
            n_channels=1, 
            num_classses=10, 
            img_size=28, 
            patch_size=4, 
            embed_dim=64, 
            num_heads=8, 
            depth=6
        )
    
        
        dummy_input = torch.randn(8, 1, 28, 28)  # batch of 8 grayscale images
        logits = model(dummy_input)


print("Output shape:", logits.shape)   # [8, 10]
'''

4. Extract Attention Maps
      '''CODE
   
        logits, attn_maps = model(dummy_input, return_all_attention=True)
        print(len(attn_maps))   # number of transformer blocks
        print(attn_maps[0].shape)  # shape: [batch_size, num_heads, num_tokens, num_tokens]
      '''
## Hyperparameters

n_channels: Channels of input image (default: 1 set for B&W images).

img_size: Size of input image (default: 28 set for MNIST).

patch_size: Size of each patch (default: 4).

embed_dim: Embedding dimension (default: 64).

num_heads: Number of attention heads (default: 8).

depth: Number of transformer blocks (default: 6).

mlp_ratio: Expansion ratio for hidden layer in MLP (default: 4).

## Example Applications

MNIST / FashionMNIST classification

CIFAR-10 (resize to 32x32 or 224x224)

Transfer learning with attention visualization

## Next Steps

Add training scripts (dataset loaders, optimizer, scheduler).

Experiment with larger models (ViT-B/16, ViT-L/32).

Compare against CNN baselines.

Implement hybrid models (CNN + ViT).

## Citation

If you use or reference this repository, please cite the original paper:

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). 
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 
arXiv preprint arXiv:2010.11929. https://doi.org/10.48550/arXiv.2010.11929