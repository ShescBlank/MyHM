# MyHM 

[IN DEVELOPMENT]

Custom hierarchical matrix (H-matrix) compression method made in Python and accelerated with Numba.

This code is part of my master's work at the Pontificia Universidad Católica de Chile with Professor Elwin van 't Wout. The main objective is to apply H-matrix compression to high intensity focused ultrasound (HIFU) contexts, seeking to compare and couple simulations of homogeneous and heterogeneous materials, as well as to study the changes produced in the ultrasound focus. The homogeneous simulations were made based on the Bempp-cl library ([bempp-cl](https://github.com/bempp/bempp-cl/tree/main)) with the Boundary Element Method. On the other hand, the heterogeneous simulations were based on the research work of Danilo Aballay (research partner with the same professor) in Volume Integral Equations ([github profile](https://github.com/daniloaballayf)).

The entire repository is under development and is updated on a regular basis.

<p align="center">
  <img src="https://github.com/ShescBlank/MyHM/blob/main/Images/bboxes.gif">
</p>
<p align="center">
Example of application of hierarchical matrices on ribs
</p>

<p align="center">
  <img src="https://github.com/ShescBlank/MyHM/blob/main/Images/compression_image_0.5_edit.gif">
</p>
<p align="center">
Representative image of the compression ratios of a dense BEM matrix for different epsilon values
</p>

## Dependencies:

    Numpy
    Bempp-cl
    Scipy
    Numba
    Pandas
    Matplotlib
    Seaborn
    Plotly
