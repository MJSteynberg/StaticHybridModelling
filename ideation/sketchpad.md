# The Well: A Polymathic AI Project <sup>[1](#thewell)</sup>

The Well is a large-scale, open-source collection of scientific datasets released by the Polymathic AI initiative. It comprises over 15 terabytes of data from 16 diverse datasets, including numerical simulations of biological systems, fluid dynamics, acoustic scattering, supernova explosions, and more. All datasets are provided in standardized formats (HDF5 with metadata) and are split for training, validation, and testing.

The community has used them for benchmarking and developing surrogate models for complex physical systems. The authors provide a set of pre-implemented baseline models.

I suggest we start by comparing HYCO against the existing baselines (most synthetic ML architectures).

Project website: https://polymathic-ai.org/the_well/


# NVIDIA PhysicsNeMo <sup>[2](#physicsnemo)</sup>

![NVIDIA PhysicsNeMo framework overview](images/250527_NVIDIA%20PhysicsNeMo.png)

NVIDIA PhysicsNeMo is an open-source Python framework for building, training, and fine-tuning physics-informed AI models. It was specifically designed for the development of surrogate models that combine physics-constraints with data. PhysicsNeMo supports a wide range of architectures, including physics-informed neural networks (PINNs), neural operators, graph neural networks (GNNs), and generative AI models. 

Key features include:
- Baseline suite of physics-ML models like the above
- PhysicsNeMo Symbolic: support for physics-informed model training: integrating symbolic PDEs, computing PDE-based residuals, domain sampling
- Built on PyTorch (should think about switching from JAX to PyTorch) and released under the Apache 2.0 license

PyTorch:
- Finite elements for the dynamic case missing (e.g., FEniCS)

Project website: https://developer.nvidia.com/physicsnemo  
GitHub: https://github.com/NVIDIA/physicsnemo


## References

<a name="thewell">[1]</a> Ohana, R., McCabe, M., Meyer, L., Morel, R., Agocs, F. J., Beneitez, M., … Ho, S. (2025). **The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning**. arXiv [Cs.LG]. Retrieved from 
[![ArXiv](https://img.shields.io/badge/ArXiv-2412.00568-blue)](http://arxiv.org/abs/2412.00568)


