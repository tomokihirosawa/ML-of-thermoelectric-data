# ML-of-thermoelectric-data

# Abstract
According to the linear response theory, the thermoelectric transport properties of electrons is determined by the spectral conductivity, i.e., the electrical conductivity as a function of the Fermi energy.
However, the spectral conductivity depends on carrier concentrations, vacancies, charge impurities, chemical compositions, and material microstructures, making it difficult to predict theoretically. 
Here, we propose a data-driven approach based on machine learning to reconstruct the spectral conductivity and chemical potential from the thermoelectric transport data. 


# How to run
We have tested our code with Python 3.8.8 and PyTorch 1.12.0. Instructions to run:
- install PyTorch, Numpy, jupyter-notebook and Matplotlib
- run the jupyter-notebook 
- we have used a method from ["torch_interpolations"](https://github.com/sbarratt/torch_interpolations)

# About this repository
It contains a minimal working example for the method introduced in the [paper](https://arxiv.org/abs/2206.01100). 
Here, we propose a data-driven approach based on machine learning to reconstruct the spectral conductivity \change{and chemical potential} from the thermoelectric transport data. Using this machine learning method, we first demonstrate that the spectral conductivity and temperature-dependent chemical potentials can be recovered within a simple toy model. In a second step, we apply our method to experimental data in doped one-dimensional telluride Ta$_4$SiTe$_4$~[T. Inohara, \textit{et al.}, Appl. Phys. Lett. \textbf{110}, 183901 (2017)] to reconstruct the spectral conductivity and chemical potential for each sample. Furthermore, the thermal conductivity of electrons and the maximal figure of merit $ZT$ are estimated from the reconstructed spectral conductivity, which provides accurate estimates beyond the Wiedemann-Franz law. Our study clarifies the connection between the thermoelectric transport properties and the low-energy electronic states of real materials, and establishes a promising route to incorporate experimental data into traditional theory-driven workflows.

If you found this work useful, please cite our paper.
- 
# Authors

