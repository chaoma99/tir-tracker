% load selected dataset 

addpath(genpath('./'));

fprintf('trained model:\n%s\n', p.savefilename); 
fprintf('test dataset:%s\n', p.testdataset); 

fprintf('loading caltech 150 gray ...\n')
load ./data/caltech_images150.mat
Xtrain = X; clear X;
%  Xtrain{1} = Xtrain{1}(:, 1:1000);
Xtest{1} = [];
if size(label, 1)>1
    Y = label;
else
    Y = label';
end

% ----- define parameters -----
% -- convolution (geometry) --
s_sp_size = p.l1.params.patchWidth;   % layer 1 (small) receptive field size
s_tp_size = p.numch; % number of channels (gray->1)
c_sp_size = sqrt(size(Xtrain{1}, 1)); % layer 2 (large) receptive field size
c_tp_size = p.numch; % number of channels (gray->1)

% -- pool size for the three layers --
psz1 = p.l1.psz;
psz2 = p.l2.psz;
psz3 = p.l3.psz;

% -- stride --
sp_stride = p.l1.stride;    % layer 1 stride (on image)
tp_stride = 1;

l2_sp_stride = p.l2.stride; % layer 2 stride (on layer 1 response map)
l3_sp_stride = p.l3.stride; % layer 3 stride (on layer 2 response map)
l2_fovea_size = (p.l2.params.patchWidth-p.l1.params.patchWidth)/sp_stride + 1;
l3_fovea_size = (p.l3.params.patchWidth-p.l2.params.patchWidth)/p.l3.strideraw + 1;

% -- constrast normalization: window size --
cnorm_sz = p.l2.cnorm_sz;

% -- load network --
network = load(p.savefilename);

% ----- define activation functions ------
act_func1_nm = @(x)actHslow_norm(x, network, 1, 10);     %1st layer activation function: norm
act_func1_ph = @(x)actHslow_phase(x, network, 1);        %1st layer activation function: phase

act_func2_nm = @(x)actHslow_PCA(x, network.layer2nm, 1); %2nd layer activation function: norm and phase

% -- find feature dimensions --
fdim1 = size(act_func1_nm(randn(s_sp_size^2*s_tp_size, 10)), 1); 
fdim2 = size(act_func2_nm(randn(fdim1*l2_fovea_size^2, 10)), 1);

if p.num_layers ==3
    act_func3_nm = @(x)actHslow_PCA(x, network.layer3nm, 1); %3rd layer activation function
    fdim3 = size(act_func3_nm(randn(fdim2*l3_fovea_size^2, 10)), 1); %note fdim2: want norm
end

% ----- define convolutional activation functions -----
% -- layer1: input images, function returns stringed convolutional response maps of first layer 16x16 filters --
conv_func1_nm = @(in)transactConv(in, act_func1_nm, fdim1, c_sp_size, c_tp_size, s_sp_size, s_tp_size, sp_stride, tp_stride); 
conv_func1_ph = @(in)transactConv(in, act_func1_ph, fdim1, c_sp_size, c_tp_size, s_sp_size, s_tp_size, sp_stride, tp_stride); 

% square response map 'side': layer 1
a = floor((c_sp_size - s_sp_size)/sp_stride+1); 
b = floor((a - l2_fovea_size)/l2_sp_stride+1);  % square repsonse map 'side': layer 2
c3 = floor((b - l3_fovea_size)/l3_sp_stride+1);  % square repsonse map 'side': layer 3

% -- layer2: input images, function returns stringed convolutional response maps of second layer (32x32 receptive field) --
conv_func2_nm = @(in)transactConvHW(in, act_func2_nm, fdim2, fdim1*a, a, 1, fdim1*l2_fovea_size, l2_fovea_size, 1, l2_sp_stride*fdim1, l2_sp_stride, 1);

% -- layer3: input images, function returns stringed convolutional response maps of second layer (dummy for now) --
if p.num_layers == 3, conv_func3_nm = @(in)transactConvHW(in, act_func3_nm, fdim3, fdim2*b, b, 1, fdim2*l3_fovea_size, l3_fovea_size, 1, l3_sp_stride*fdim2, l3_sp_stride, 1); end

% ----- SPM/average pooling matrices -----
%% SPM meanfilters for layer 1: [hard code a 3 layer SPM]
%% 17, 34, 68 are sizes of pooling regions
meanf_M = create_filter_matrix(a, a, 17); % this function creates a matrix, when multiplied, performs spatial average pooling on the features
meanf_M_a = create_filter_matrix(a, a, 34); 
meanf_M_b = create_filter_matrix(a, a, 68);

%% SPM meanfilters for layer 2: [hard code a 3 layer SPM]
meanf_M2 = create_filter_matrix(b, b, 8);
meanf_M2_a = create_filter_matrix(b, b, 16);
meanf_M2_b = create_filter_matrix(b, b, 31);

%% SPM meanfilters for layer 3
meanf_M3 = create_filter_matrix(c3, c3, psz3); % matrix for average pooling

num_train_images = size(Xtrain{1}, 2);
num_test_images = size(Xtest{1}, 2);

X = Xtrain{1}'; Xtest = Xtest{1}'; clear Xtrain;

% $$$--- work out feature dimensions ---$$$
t = randn(size(X, 2), 10);
c = conv_func1_nm(t);
d = transactConvHW(c, act_func2_nm, fdim2, fdim1*a, a, 1, fdim1*l2_fovea_size, l2_fovea_size, 1, l2_sp_stride*fdim1, l2_sp_stride, 1);

c = reshape(c, fdim1, [])';
c = conv_contrast_normalize(c, a, a, cnorm_sz);

d = reshape(d, fdim2, [])';
d = conv_contrast_normalize(d, b, b, cnorm_sz);

fprintf('-- average subsampling layer 1 --- \n');
c = reshape(c, [], 10*fdim1);
c_0 = meanf_M*c;
c_a = meanf_M_a*c; 
c_b = meanf_M_b*c; 

fprintf('-- average subsampling layer 2 --- \n');
d = reshape(d, [], 10*fdim2);
d_0 = meanf_M2*d;
d_a = meanf_M2_a*d;
d_b = meanf_M2_b*d;

fprintf('-- reshaping --- \n');
c_0 = reshape(reshape(c_0, [], fdim1)', [], 10);
c_a = reshape(reshape(c_a, [], fdim1)', [], 10);
c_b = reshape(reshape(c_b, [], fdim1)', [], 10);

d_0 = reshape(reshape(d_0, [], fdim2)', [], 10);
d_a = reshape(reshape(d_a, [], fdim2)', [], 10);
d_b = reshape(reshape(d_b, [], fdim2)', [], 10);

full_feature_dim=2*(size(c_0, 1)+size(c_a, 1)+size(c_b, 1)) + size(d_0, 1) + size(d_a, 1) + size(d_b, 1);

if p.num_layers==3
    e = conv_func3_nm(d);
    
    e = reshape(e, fdim3, [])';
    e = conv_contrast_normalize(e, c3, c3, cnorm_sz);
    
    fprintf('-- average subsampling layer 3 --- \n');
    e = reshape(e, [], 10*fdim3);
    e = meanf_M3*e;
    
    e = reshape(reshape(e, [], fdim3)', [], 10);
    
    full_feature_dim = full_feature_dim + size(e, 1);
end

% $$$--- [END] work out feature dimensions ---$$$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- start feature extraction ---

fprintf('calculating features for the training set ......\n');

ftrain = zeros(full_feature_dim, size(X, 1));
count = 0;
while count < size(X, 1)
    fprintf('batching: count = %d/%d\n', count, size(X, 1));
    inc = 400; % batch size: process this number of images per batch
    count_end = min(size(X, 1), count+inc);
    
    fprintf('-- computing layer 1 activations (convolution)--- \n');
    tic
    l1onm = conv_func1_nm(X(count+1:count_end, :)');
    l1oph = conv_func1_ph(X(count+1:count_end, :)');
    toc
    
    %std normalize l1 activations
    l1onm = stdnormalize(l1onm);
    
    fprintf('-- computing layer 2 activations (convolution)--- \n');
    
    tic
    fprintf('-- norm -- \n');
    l2onm = conv_func2_nm(l1onm);
    toc
    
    if p.num_layers ==3
    fprintf('-- computing layer 3 activations (convolution)--- \n');
    l2onm = stdnormalize(l2onm); 
    tic
    l3onm = conv_func3_nm(l2onm);
    toc
    end
    
    fprintf(' ----- perform constrast normalization -----\n')
    l1onm = reshape(l1onm, fdim1, [])';
    l1onm = conv_contrast_normalize(l1onm, a, a, cnorm_sz);
    l1oph = reshape(l1oph, fdim1, [])';
    l1oph = conv_contrast_normalize(l1oph, a, a, cnorm_sz);
    
    l2onm = reshape(l2onm, fdim2, [])';
    l2onm = conv_contrast_normalize(l2onm, b, b, cnorm_sz);
    
    if p.num_layers == 3
        l3onm = reshape(l3onm, fdim3, [])';
        l3onm = conv_contrast_normalize(l3onm, c3, c3, cnorm_sz);
    end

    fprintf('-- average subsampling layer 1 --- \n');
    tic
    % reshape to mean filter layer 1
    l1onm = reshape(l1onm, [], (count_end-count)*fdim1);
    l1oph = reshape(l1oph, [], (count_end-count)*fdim1);

    l1onm_0 = meanf_M*l1onm;
    l1oph_0 = meanf_M*l1oph;

    l1onm_a = meanf_M_a*l1onm;
    l1oph_a = meanf_M_a*l1oph;

    l1onm_b = meanf_M_b*l1onm;
    l1oph_b = meanf_M_b*l1oph;

    toc
    
    fprintf('-- average subsampling layer 2 --- \n');
    tic
    % reshape to mean filter layer 2
    l2onm = reshape(l2onm, [], (count_end-count)*fdim2);

    l2onm_0 = meanf_M2*l2onm;

    l2onm_a = meanf_M2_a*l2onm;

    l2onm_b = meanf_M2_b*l2onm;

    toc

    if p.num_layers == 3
    fprintf('-- average subsampling layer 3 --- \n');
    tic
    % reshape to mean filter layer 3
    l3onm = reshape(l3onm, [], (count_end-count)*fdim3);
    l3onm = meanf_M3*l3onm;
    toc
    end

    fprintf('-- reshaping --- \n');
    tic
    l1onm_0 = reshape(reshape(l1onm_0, [], fdim1)', [], (count_end-count));
    l1oph_0 = reshape(reshape(l1oph_0, [], fdim1)', [], (count_end-count));
    l1onm_a = reshape(reshape(l1onm_a, [], fdim1)', [], (count_end-count));
    l1oph_a = reshape(reshape(l1oph_a, [], fdim1)', [], (count_end-count));
    l1onm_b = reshape(reshape(l1onm_b, [], fdim1)', [], (count_end-count));
    l1oph_b = reshape(reshape(l1oph_b, [], fdim1)', [], (count_end-count));

    l2onm_0 = reshape(reshape(l2onm_0, [], fdim2)', [], (count_end-count));
    l2onm_a = reshape(reshape(l2onm_a, [], fdim2)', [], (count_end-count));
    l2onm_b = reshape(reshape(l2onm_b, [], fdim2)', [], (count_end-count));


    if p.num_layers ==3, l3onm = reshape(reshape(l3onm, [], fdim3)', [], (count_end-count)); end
    toc
    if p.num_layers == 3
    ftrain(:, count+1:count_end) = [l1onm; l1oph; l2onm; l3onm];
    else
    ftrain(:, count+1:count_end) = [l1onm_0; l1onm_a; l1onm_b; l1oph_0; l1oph_a; l1oph_b; l2onm_0; l2onm_a; l2onm_b];
    end

    count = count + inc;
end

toc

clear X

%% comp feature test set
fprintf('calculating features for the testing set ......\n');
tic
ftest = zeros(size(ftrain, 1), size(Xtest, 1));
count = 0;
while count < size(Xtest, 1)
    fprintf('batching: count = %d/%d\n', count, size(Xtest, 1));
    inc = 400;
    count_end = min(size(Xtest, 1), count+inc);

    fprintf('-- computing layer 1 activations (convolution)--- \n');
    tic
    l1onm = conv_func1_nm(Xtest(count+1:count_end, :)');
    l1oph = conv_func1_ph(Xtest(count+1:count_end, :)');
    toc

    %std normalize l1 activations
    l1onm = stdnormalize(l1onm);

    fprintf('-- computing layer 2 activations (convolution)--- \n');

    tic
    fprintf('-- norm -- \n');
    l2onm = conv_func2_nm(l1onm);
    toc

    if p.num_layers ==3
    fprintf('-- computing layer 3 activations (convolution)--- \n');
    l2onm = stdnormalize(l2onm); 
    tic
    l3onm = conv_func3_nm(l2onm);
    toc
    end
    
    fprintf(' ----- perform constrast normalization -----\n')
    l1onm = reshape(l1onm, fdim1, [])';
    l1onm = conv_contrast_normalize(l1onm, a, a, cnorm_sz);
    l1oph = reshape(l1oph, fdim1, [])';
    l1oph = conv_contrast_normalize(l1oph, a, a, cnorm_sz);
    
    l2onm = reshape(l2onm, fdim2, [])';
    l2onm = conv_contrast_normalize(l2onm, b, b, cnorm_sz);

    if p.num_layers == 3
    l3onm = reshape(l3onm, fdim3, [])';
    l3onm = conv_contrast_normalize(l3onm, c3, c3, cnorm_sz);
    end

    fprintf('-- average subsampling layer 1 --- \n');
    tic
    % reshape to mean filter layer 1
    l1onm = reshape(l1onm, [], (count_end-count)*fdim1);
    l1oph = reshape(l1oph, [], (count_end-count)*fdim1);

    l1onm_0 = meanf_M*l1onm;
    l1oph_0 = meanf_M*l1oph;

    l1onm_a = meanf_M_a*l1onm;
    l1oph_a = meanf_M_a*l1oph;

    l1onm_b = meanf_M_b*l1onm;
    l1oph_b = meanf_M_b*l1oph;

    toc
    
    fprintf('-- average subsampling layer 2 --- \n');
    tic
    % reshape to mean filter layer 2
    l2onm = reshape(l2onm, [], (count_end-count)*fdim2);
    
    l2onm_0 = meanf_M2*l2onm;
    
    l2onm_a = meanf_M2_a*l2onm;
    
    l2onm_b = meanf_M2_b*l2onm;
    
    toc
    
    if p.num_layers == 3
    fprintf('-- average subsampling layer 3 --- \n');
    tic
    % reshape to mean filter layer 3
    l3onm = reshape(l3onm, [], (count_end-count)*fdim3);
    l3onm = meanf_M3*l3onm;
    toc
    end
    
    fprintf('-- reshaping --- \n');
    tic
    l1onm_0 = reshape(reshape(l1onm_0, [], fdim1)', [], (count_end-count));
    l1oph_0 = reshape(reshape(l1oph_0, [], fdim1)', [], (count_end-count));
    l1onm_a = reshape(reshape(l1onm_a, [], fdim1)', [], (count_end-count));
    l1oph_a = reshape(reshape(l1oph_a, [], fdim1)', [], (count_end-count));
    l1onm_b = reshape(reshape(l1onm_b, [], fdim1)', [], (count_end-count));
    l1oph_b = reshape(reshape(l1oph_b, [], fdim1)', [], (count_end-count));

    l2onm_0 = reshape(reshape(l2onm_0, [], fdim2)', [], (count_end-count));
    l2onm_a = reshape(reshape(l2onm_a, [], fdim2)', [], (count_end-count));
    l2onm_b = reshape(reshape(l2onm_b, [], fdim2)', [], (count_end-count));

    if p.num_layers ==3, l3onm = reshape(reshape(l3onm, [], fdim3)', [], (count_end-count)); end
    toc
    if p.num_layers == 3
    ftest(:, count+1:count_end) = [l1onm; l1oph; l2onm; l3onm];
    else
    ftest(:, count+1:count_end) = [l1onm_0; l1onm_a; l1onm_b; l1oph_0; l1oph_a; l1oph_b; l2onm_0; l2onm_a; l2onm_b];
    end
    
    count = count + inc;
end 
toc

clear Xtest

ftrain = ftrain';
ftest = ftest';

%% ----- run svm classification ------- 
C = 100 % example weight cost for the SVM 
fprintf('cost = %f\n', C);
run_svm_caltech

fprintf('%s\n', p.savefilename);
