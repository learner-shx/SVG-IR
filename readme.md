# SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering (CVPR 2025)

### <p align="center">[🌐Project Page](https://learner-shx.github.io/project_pages/SVG-IR/index) | [🖨️ArXiv](https://arxiv.org/abs/2504.06815) | [📰Paper](https://arxiv.org/pdf/2504.068153)</p>


<p align="center">
Hanxiao Sun<sup>1</sup>, Yupeng Gao<sup>2</sup>, Jin Xie<sup>2</sup>, Jian Yang<sup>1</sup>, Beibei Wang<sup>2</sup><i class="fa fa-envelope"> </i></sup></h5> <br><sup>1</sup>Nankai University <sup>2</sup>Nanjing University <br> 
</p>


This is official implement of Relightable 2D Gaussian for the paper.

*SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Renderin*.
![Alt text](https://learner-shx.github.io/project_pages/SVG-IR/static/images/teaser.png)


### Installation
#### Clone this repo
```shell
git clone https://github.com/learner-shx/SVG-IR.git
```
#### Install dependencies
```shell
# install environment
conda env create --file environment.yml
conda activate r3dg

# install pytorch=1.12.1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# install torch_scatter==2.1.1
pip install torch_scatter==2.1.1

# install kornia==0.6.12
pip install kornia==0.6.12

# install nvdiffrast=0.3.1
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

# install slang-torch
pip install slangtorch==1.2.1
```

#### Install the pytorch extensions
We recommend that users compile the extension with CUDA 11.8 to avoid the potential problems mentioned in [3D Guassian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```shell
# install knn-cuda
pip install ./submodules/simple-knn

# install rgss (relightable gaussian surfels splatting)
pip install ./rgss-rasterization

# install svgss (spatially-varying gaussian surfels splatting)
pip install ./svgss-rasterization
```
### Data preparation
#### TensoIR Synthetic Dataset
Download the TensoIR synthetic dataset from [LINK](https://zenodo.org/records/7880113#.ZE68FHZBz18) provided by [TensoIR](https://github.com/Haian-Jin/TensoIR).

#### Data Structure
We organize the datasets like this:
```
Relightable3DGaussian
├── datasets
    ├── TensoIR
    |   ├── armadillo
    |   ├── ...
```

#### Ground Points for composition
For multi-object composition, we manually generate a ground plane with relightable 3D Gaussian representation, which can be downloaded [here](https://box.nju.edu.cn/f/c51d9de245f04d0fb872/?dl=1). We put the *ground.ply* in the folder *./point*.

### Running
We run the code in a single NVIDIA GeForce RTX 3090 GPU (24G). To reproduce the results in the paper, please run the following code.
TensoIR Synthetic dataset:
```
sh script/run_tensoIR.sh
```

### Evaluating
Run the following command to evaluate Novel View Synthesis:
```
# e.g. TensoIR dataset
# stage 1
python eval_nvs.py --eval \
    -m output/TensoIR/${i}/3dgs \
    -c output/TensoIR/${i}/3dgs/chkpnt30000.pth

# stage 2
python eval_nvs.py --eval \
    -m output/TensoIR/${i}/render_relight \
    -c output/TensoIR/${i}/render_relight/chkpnt50000.pth \
    -t render_relight
```
Run the following command to evaluate Relighting (for Synthetic4Relight only):
```
# e.g.
python eval_relighting_tensoIR.py \
    -m output/TensoIR/hotdog/render_relight \
    -c output/TensoIR/hotdog/render_relight/chkpnt50000.pth \
    --sample_num 384
```

### Trying on your own data
We recommend that users reorganize their own data as render_relightpp-like dataset and then optimize. Modified VisMVSNet and auxiliary scripts to prepare your own data will come soon.


### Citation
If you find our work useful in your research, please be so kind to cite:
```
@article{SVGIR2025,
    author    = {Sun, Hanxiao and Gao, Yupeng and Xie, Jin and Yang, Jian and Wang, Beibei},
    title     = {SVG-IR:Spatially-Varying Gaussian Splatting for Inverse Rendering},
    journal   = {arXiv:2504.06815},
    year      = {2025},
}
```

### Acknowledgement
The code was built on [Relightable3DGS](https://github.com/NJU-3DV/Relightable3DGaussian), [GaussianSurfels](https://github.com/turandai/gaussian_surfels), [MIRRes](https://github.com/brabbitdousha/MIRReS-ReSTIR_Nerf_mesh). Thanks for these great projects!