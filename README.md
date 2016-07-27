# latenttrees
This repository contains the Python code from the following publication:

S. Kaltwang, S. Todorovic, and M. Pantic, 
“Latent Trees for Estimating Intensity of Facial Action Units,” 
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

Please cite the publication above, in case that you use this software.

Dependencies:
- numpy
- scipy
- matplotlib
- networkx
- graphviz (optional, to draw the tree)
- pydotplus (optional, to draw the tree)

Usage:
- make sure the modules latenttrees and misc can be found in the Python path
- execute "python example_latenttrees.py" for a demo on random data
- see the comments within example_latenttrees.py to adapt the model to your needs

This code has been tested with Python 3.4.4 and 3.5.2

Note: See also the [original Matlab implementation of this model](https://github.com/kaltwang/2015latent).