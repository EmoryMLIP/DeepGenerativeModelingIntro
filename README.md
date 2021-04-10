# DeepGenerativeModelingIntro

PyTorch Code used in the paper *Introduction to Deep Generative Modeling*:
 
    @article{RuthottoHaber2021,
      title = {An Introduction to Deep Generative Modeling},
      year = {2021},
      journal = {arXiv preprint arXiv:tbd},
      author = {L. Ruthotto and E. Haber},
      pages = {25 pages},
      url={https://arxiv.org/abs/2103.05180}
    }

## Run Examples from the Terminal

To reproduce the examples from the paper (up to randomization), we provide a shell script `runAll.sh`. 
    
## Run Examples in Colab

The `examples` directory contains interactive version of the examples from the paper. Those can be run locally or using 
Google Colab. For the latter option, you may click the badges below:
 
1.   [Two-Dimensional Normalizing Flow Examples with Real NVP](https://github.com/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/RealNVP.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/RealNVP.ipynb)  
1.   [Two-Dimensional Continuous Normalizing Flow Example with OT-Flow](https://github.com/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/OTFlow.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/OTFlow.ipynb)  
1.   [Variational Autoencoder for MNIST Image Generation](https://github.com/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/VAE.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/VAE.ipynb)  
1.   [DCGAN for MNIST Image Generation](https://github.com/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/DCGAN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/DCGAN.ipynb)  
1.   [WGAN  for MNIST Image Generation](https://github.com/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/WGAN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmoryMLIP/DeepGenerativeModelingIntro/blob/main/examples/WGAN.ipynb)  
 
## Dependencies

The code is based on pytorch and some other standard machine learning packages.  In addition,  training the continuous normalizing flow example requires [OT-Flow](https://github.com/EmoryMLIP/OT-Flow).
  
## Acknowledgements

This material is in part based upon work supported by the National Science Foundation under Grant Number 1751636, the Air Force Office of Scientific Research under Grant Number 20RT0237, and 
the US DOE's Office of Advanced Scientific Computing Research Field Work Proposal 20-023231. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.
 