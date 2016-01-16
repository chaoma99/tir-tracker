addpath ./kmeans_demo/
addpath ./minFunc/
addpath ./tbox/

no_vis_bases = 1; % disable visualization during training 

% gamma (slow) and lambda (sparse) parameters
l1_gamma = 50; 
l1_lambda = 150; 
l2_gamma = 1; 
l2_lambda = 500; 
l3_gamma = 1; 
l3_lambda = 500; 

num_layers = 2; % choose number of layers to train
use_track = 1;  % whether or not use tracked data for training

numch = 1; % number of color channels

% sample/visualization options [obsolete]
sample_patches = 0;
testvis = 0;

% [DATASETS]
p.traindataset = 'vanH';
p.testdataset = 'caltech';
p.numch = numch;
p.l1.num_patches = 3e4;
p.l2.num_patches = 5e4;
p.l3.num_patches = 5e4;

% [SIZE]
l1_patchWidth = 16;
l2_patchWidth = 32;
l3_patchWidth = 32;

l1_numFeatures = 256;
l2_numFeatures = 300;
l2_pca_dim = l2_numFeatures;
l3_numFeatures = 300;
l3_pca_dim = l3_numFeatures;

% [CONVNET ARCHITECTURE]
l1_psz = 6;
l2_psz = 4;
l3_psz = 2;

p.l1.stride = 2;
p.l2.stride = 2;
p.l3.stride = 2;
p.l3.strideraw = p.l1.stride*p.l2.stride;

p.l2.cnorm_sz = 4;
p.l3.cnorm_sz = 4;

% [OPTIMIZATION]
l1_maxiter = 750;
l2_maxiter = 1500;
l3_maxiter = 1500;

p.num_layers = num_layers;
p.use_track = use_track;

%% ----- layer 1 ------
p.l1.params.patchWidth = l1_patchWidth;
p.l1.params.n = p.l1.params.patchWidth^2;
p.l1.psz = l1_psz;

p.l1.params.numFeatures = l1_numFeatures;
p.l1.params.gamma = l1_gamma;
p.l1.params.lambda = l1_lambda;
p.l1.params.lambda2 = 0;
p.l1.params.lambda3 = 0;
p.l1.params.lambda4 = 0;
p.l1.params.gamma2 = 0;
p.l1.params.gamma3 = 0;
p.l1.params.gamma4 = 0;
p.l1.params.group_size = 2;
p.l1.params.group_size2 = 2;
p.l1.params.group_size3 = 2;
p.l1.params.group_size4 = 2;
p.l1.params.push_ratio = 0;
p.l1.params.phaselambda = 0;
p.l1.params.weightcost = 0;
p.l1.params.skipframes = 1;

H = subspacematrix(p.l1.params.numFeatures, p.l1.params.group_size);
p.l1.params.H = H(1:p.l1.params.group_size:end, :);

% [optimization params]
options.Method = 'lbfgs'; options.MaxFunEvals = Inf;
options.Corr = 5;
options.maxoutIter = 1;
options.num_batches = 1;
options.permute = 0;
options.MaxIter = l1_maxiter;
if ~exist('no_vis_bases', 'var')
  options.outputFcn = 'showBases_color';
end
p.l1.mF_options = options; clear options

%% ----- layer 2 ------
p.l2.params.patchWidth = l2_patchWidth;
p.l2.params.pca_dim = l2_pca_dim;
p.l2.params.n = l2_pca_dim;
p.l2.psz = l2_psz;

p.l2.params.numFeatures = l2_numFeatures;
p.l2.params.gamma = l2_gamma;
p.l2.params.lambda = l2_lambda;
p.l2.params.lambda2 = 0;
p.l2.params.lambda3 = 0;
p.l2.params.lambda4 = 0;
p.l2.params.gamma2 = 0;
p.l2.params.gamma3 = 0;
p.l2.params.gamma4 = 0;
p.l2.params.group_size = 2;
p.l2.params.group_size2 = 2;
p.l2.params.group_size3 = 2;
p.l2.params.group_size4 = 2;
p.l2.params.push_ratio = 0;
p.l2.params.phaselambda = 0;
p.l2.params.weightcost = 0;
p.l2.params.skipframes = 1;

H = subspacematrix(p.l2.params.numFeatures, p.l2.params.group_size);
p.l2.params.H = H(1:p.l2.params.group_size:end, :);

% [optimization params]
options.Method = 'lbfgs'; options.MaxFunEvals = Inf;
options.Corr = 5;
options.maxoutIter = 1;
options.num_batches = 1;
options.permute = 0;
options.MaxIter = l2_maxiter;
p.l2.mF_options = options;

%% ----- layer 3 ------
p.l3.params.patchWidth = l3_patchWidth;
p.l3.params.pca_dim = l3_pca_dim;
p.l3.params.n = l3_pca_dim;
p.l3.psz = l3_psz;

p.l3.params.numFeatures = l3_numFeatures;
p.l3.params.gamma = l3_gamma;
p.l3.params.lambda = l3_lambda;
p.l3.params.skipframes = 1;
p.l3.params.lambda2 = 0;
p.l3.params.lambda3 = 0;
p.l3.params.lambda4 = 0;
p.l3.params.gamma2 = 0;
p.l3.params.gamma3 = 0;
p.l3.params.gamma4 = 0;
p.l3.params.group_size = 2;
p.l3.params.group_size2 = 2;
p.l3.params.group_size3 = 2;
p.l3.params.group_size4 = 2;
p.l3.params.push_ratio = 0;
p.l3.params.phaselambda = 0;
p.l3.params.weightcost = 0;

H = subspacematrix(p.l3.params.numFeatures, p.l3.params.group_size);
p.l3.params.H = H(1:p.l3.params.group_size:end, :);

% [optimization params]
options.Method = 'lbfgs'; options.MaxFunEvals = Inf;
options.Corr = 5;
options.maxoutIter = 1;
options.num_batches = 1;
options.permute = 0;
options.MaxIter = l3_maxiter;
p.l3.mF_options = options;
