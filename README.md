# Multi-Modal Depth Estimation with Semantic Segmentation

This project implements a DepthNet-style architecture for depth completion using RGB images, sparse LiDAR/depth maps, and semantic segmentation maps as inputs. The model is trained on the NYU Depth v2 dataset and outputs dense depth maps. This project was inspired by the DepthNet and Pix2Pix papers.

    🧪 This was a fun experimental project completed during the first month of my summer vacation using free GPU time on Kaggle.

## 🧠 Key Highlights

    Input modalities: RGB, sparse depth, semantic segmentation

    Backbone architecture: DepthNet-inspired encoder-decoder with skip connections

    Multi-scale supervision: Supervision at multiple resolutions from coarse to fine

    Loss function: Weighted multi-scale L1 loss with optional relative normalization

    Semantic awareness: Uses instance maps and semantic masks to improve structural understanding

    Framework: Implemented in PyTorch
    

## 📂 Dataset

NYU Depth v2 was used for training and evaluation.

    Dataset: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

    Preprocessed by generating:

        RGB images

        Sparse depth maps (downsampled/quantized)

        Ground truth depth maps

        Semantic segmentation masks
        

## 🏗️ Model Architecture

The model follows a modified version of DepthNet, consisting of:

    Encoder: Several strided convolutional blocks to downsample the input and extract features

    Decoder: Transposed convolutions and skip connections for upsampling

    Multi-Scale Depth Prediction: Depth maps are predicted at 5 resolutions (64×64 up to 256×256), with the finest one used at test time

    Semantic Guidance: Semantic features are implicitly learned via additional input channels

Inputs (6 channels total):

[ RGB (3) + Sparse Depth (1) + Dummy Channels (2) ]


## 🏋️ Training
Requirements

torch==2.0.1
torchvision==0.15+
matplotlib
Pillow
tqdm

Custom Loss Function

def depth_metric_reconstruction_loss(pred, target, normalize=False):
    # Multi-scale loss computation
    ...

Training Script

python train.py --epochs 300 --batch-size 4 --lr 1e-4 --dataset nyu_dataset

You can also run it in Kaggle Notebooks with GPU enabled.

📈 Results
Epoch	L1 Loss	Visualization
90	~0.12	
150	~0.06	
250	~0.025
500 ~0.008

    Despite low losses, qualitative results highlighted the need for semantic guidance and better refinement strategies.
    

🔍 Inference

Use a saved model to generate predictions:

model = DepthNet()
model.load_state_dict(torch.load("best.pth"))
model.eval()

rgb = ...
sparse = ...
dummy = torch.zeros_like(...)

input_tensor = torch.cat([rgb, sparse, dummy], dim=1)
with torch.no_grad():
    output = model(input_tensor)
    

## 📝 Inspiration & References

    DepthNet Paper (Wofk et al., ICCV 2019)

    Pix2Pix: Image-to-Image Translation with Conditional GANs

    NYU Depth v2 Dataset

    Free GPU Compute via Kaggle Notebooks

    

### 🙌 Acknowledgements

Thanks to Kaggle for providing free GPU time and a smooth training experience.
This project was a part of my personal learning journey during summer vacation, and helped me gain hands-on experience with multi-modal deep learning pipelines and loss functions for dense prediction tasks.


Feel free to fork, experiment, and improve!
