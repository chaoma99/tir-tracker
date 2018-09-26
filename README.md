### Learning a Temporally Invariant Representation for Visual Tracking

This is the research code for the ICIP 2015 paper:

Chao Ma, Xiaokang Yang, Chongyang Zhang, and Ming-Hsuan Yang, Learning a Temporally Invariant Representation for Visual Tracking, in ICIP 2015 (Top 10% paper)

In this paper, we propose to learn temporally invariant features
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
efficiency, accuracy, and robustness.


### Quick Start

> run_tracker.m
