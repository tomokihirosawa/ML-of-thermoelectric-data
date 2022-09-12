# ML-of-thermoelectric-data

# Abstract
According to the linear response theory, the thermoelectric transport properties of electrons is determined by the spectral conductivity, i.e., the electrical conductivity as a function of the Fermi energy.
However, the spectral conductivity depends on carrier concentrations, vacancies, charge impurities, chemical compositions, and material microstructures, making it difficult to predict theoretically. 
Here, we propose a data-driven approach based on machine learning to reconstruct the spectral conductivity and chemical potential from the thermoelectric transport data. 

![pic1](https://user-images.githubusercontent.com/24930817/189618242-fd3e3fc9-00c3-4e7c-95cc-16a3e52d76ea.png)

# How to run
We have tested our code with Python 3.8.8 and PyTorch 1.12.0. Instructions to run:
- install PyTorch, Numpy, jupyter-notebook and Matplotlib
- run the jupyter-notebook 
- we have used a method from ["torch_interpolations"](https://github.com/sbarratt/torch_interpolations)

# About this repository
It contains a minimal working example for the method introduced in the [paper](https://arxiv.org/abs/2206.01100). 
Using the machine learning method, we first demonstrate that the spectral conductivity and temperature-dependent chemical potentials can be recovered within a simple toy model. In a second step, we apply our method to [the experimental data of doped one-dimensional telluride Ta$_4$SiTe$_4$](https://aip.scitation.org/doi/10.1063/1.4982623) to reconstruct the spectral conductivity and chemical potential for each sample. 

If you found this work useful, please cite our [paper](https://arxiv.org/abs/2206.01100).



