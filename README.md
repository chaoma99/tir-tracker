# Learning a Temporally Invariant Representation for Visual Tracking - ICIP 2015

Chao Ma, Xiaokang Yang, Chongyang Zhang, and Ming-Hsuan Yang

ICIP 2015 (TOP 10% Paper)


> main entrance: run_tracker

> learning features in ./Feature


<h1 align="center"><font size="5">Learning a Temporally Invariant Representation for Visual Tracking</font></h1>
<p align="center">Chao Ma<img src="https://www.google.com/chart?cht=tx&amp;chf=bg,s,FFFFFF00&amp;chco=000000&amp;chl=%5E%7B%5Cstar%5Cdagger%7D">&nbsp;&nbsp;&nbsp;&nbsp; 
Xiaokang Yang<img src="https://www.google.com/chart?cht=tx&amp;chf=bg,s,FFFFFF00&amp;chco=000000&amp;chl=%5E%7B%5Cstar%7D">&nbsp; &nbsp;&nbsp;&nbsp;
Chongyang Zhang<img src="https://www.google.com/chart?cht=tx&amp;chf=bg,s,FFFFFF00&amp;chco=000000&amp;chl=%5E%7B%5Cstar%7D"> &nbsp; &nbsp; &nbsp;&nbsp; 
Ming-Hsuan Yang<img src="https://www.google.com/chart?cht=tx&amp;chf=bg,s,FFFFFF00&amp;chco=000000&amp;chl=%5E%7B%5Cdagger%7D"></p>
<p align="center"><img src="https://www.google.com/chart?cht=tx&amp;chf=bg,s,FFFFFF00&amp;chco=000000&amp;chl=%5E%7B%5Cstar%7D">Shanghai Jiao Tong University &nbsp; &nbsp; &nbsp; 
<img src="https://www.google.com/chart?cht=tx&amp;chf=bg,s,FFFFFF00&amp;chco=000000&amp;chl=%5E%7B%5Cdagger%7D">University of California at Merced</p>

<img alt="" src="https://sites.google.com/site/chaoma99/icip15_tracking.png" width="800px">

<p style="margin-left:8em;margin-right:8em;text-align:justify;text-justify:inter-ideograph"><font size="2">
Figure 1. <b>Left:</b> Neural network architecture with square root subspace
space pooling. The input training data are small patches
of size 16x16 pixels with temporal slowness. <b>Right:</b> Learned
filters with the main half of components.
</font></p>
</div>

<p>&nbsp;</p>
<p><strong><span>Abstract</span></strong></p>
<p>In this paper, we propose to learn temporally invariant features
from a large number of image sequences to represent
objects for visual tracking. These features are trained on a
convolutional neural network with temporal invariance constraints
and robust to diverse motion transformations. We employ
linear correlation filters to encode the appearance templates
of targets and perform the tracking task by searching
for the maximum responses at each frame. The learned filters
are updated online and adapt to significant appearance
changes during tracking. Extensive experimental results on
challenging sequences show that the proposed algorithm performs
favorably against state-of-the-art methods in terms of
efficiency, accuracy, and robustness.</p>
<p>&nbsp;</p>

