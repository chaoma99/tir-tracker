% ----- MATLAB demo code [v1.0 Jan 2013]: -----
% Deep Learning of Invariant Features via Simulated Fixations in Video
% Will Zou, Shenghuo Zhu, Andrew Ng, Kai Yu
% NIPS 2012
% -----
% 
% README 

The demo code should produce a classification accuracy on the Caltech-101 test set between 73% to 74%, close to results stated in the paper. 

The code consists of three parts: 
@ training two layer temporal slowness network (extendable to three)
@ use learned features to classify images on Caltech-101
$ visualize first and second layer invariances

@ The first two parts are tied together in MATLAB script: 
[ master.m ]

before running master.m, please download data files
http://dl.dropbox.com/u/29142041/deepslow/data/vanH_tracked_patches.mat
http://dl.dropbox.com/u/29142041/deepslow/data/vanH_untracked_patches.mat
http://dl.dropbox.com/u/29142041/deepslow/data/caltech_images150.mat
to ./data/

learned bases (weights/parameters) are stored in ./bases/caltech/

$ The visualization can be run with MATLAB script: 
[ vis_1l.m ]
[ vis_2l.m ]

with bases 'filename' correctly specified

Please refer to the above-mentioned scripts for detailed comments.

To run the whole pipeline, the code requires ~9GB of RAM on your cluster (this can be alleviated by perhaps using optimized PCA implementations to replace pca_mod.m). 

% wzou@stanford.edu

----------------------------------
This package includes modified external code written by researchers, such as matlab code for Natural Image Statistics Book (such as pf.m, pca_mod.m), and: 
v
minFunc optimization package
----------------------------------
See:  http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

minFunc, written by Mark Schmidt, is included with this code.  The
minFunc license follows:

"This software is made available under the Creative Commons 
Attribution-Noncommercial License.  You are free to use, copy, modify, and 
re-distribute the work.  However, you must attribute any re-distribution or 
adaptation in the manner specified below, and you may not use this work for 
commercial purposes without the permission of the author.

Any re-distribution or adaptation of this work must contain the author's name 
and a link to the software's original webpage.  For example, any 
re-distribution of the 'minFunc' software must contain a link to:
http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

This software comes with no guarantees, and all use of these codes is
entirely at the user's own risk."
