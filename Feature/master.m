% ----- MATLAB demo code -----
% Deep Learning of Invariant Features via Simulated Fixations in Video
% Will Zou, Shenghuo Zhu, Andrew Ng, Kai Yu
% NIPS 2012
% -----
% 
% Master file for running temporal slowness with two-layer linear autoencoders

% ----- [section1] set parameters -----
preset_params; 

%% [GLOBAL] %%
global params;

% ------ [section2] load data ------
if p.use_track
    load ./data/vanH_tracked_patches.mat
else
    load ./data/vanH_untracked_patches.mat
end

% ------ [section3] set savename and run training ------
p.basespath = sprintf('./bases/%s', p.testdataset);
mkdir(p.basespath);
% p.savefilename = sprintf('%s/convl3_usetrack%d_trda%s_teda%s_l1pch%d_l2pch%d_l3pch%d_l1str%d_l2str%d_l3str%d_l1nf%d_l2nf%d_l3nf%d_l2pca%d_l3pca%d_l1gam%.3f_l1lamb%.3f_l2gam%.3f_l2lamb%.3f_l3gam%.3f_l3lamb%.3f_wc%.5f_l1iter%d_l2iter%d_l3iter%d.mat', ...
%                          p.basespath, p.use_track, p.traindataset, p.testdataset, p.l1.params.patchWidth, p.l2.params.patchWidth, p.l3.params.patchWidth, ...
%                          p.l1.stride, p.l2.stride, p.l3.stride, p.l1.params.numFeatures, p.l2.params.numFeatures, p.l3.params.numFeatures, p.l2.params.pca_dim, p.l3.params.pca_dim, ...
%                          p.l1.params.gamma, p.l1.params.lambda, p.l2.params.gamma, p.l2.params.lambda, p.l3.params.gamma, p.l3.params.lambda, 0, l1_maxiter, l2_maxiter, l3_maxiter);

p.savefilename = sprintf('%s/convl3_usetrack%d_trda%s_teda%s_l1pch%d_l2pch%d_l3pch%d_l1str%d_l2str%d_l3str%d_l1nf%d_l2nf%d_l3nf%d_l2pca%d_l3pca%d_wc%.5f_l1iter%d_l2iter%d_l3iter%d.mat', ...
                         p.basespath, p.use_track, p.traindataset, p.testdataset, p.l1.params.patchWidth, p.l2.params.patchWidth, p.l3.params.patchWidth, ...
                         p.l1.stride, p.l2.stride, p.l3.stride, p.l1.params.numFeatures, p.l2.params.numFeatures, p.l3.params.numFeatures, p.l2.params.pca_dim, p.l3.params.pca_dim, ...
                         0, l1_maxiter, l2_maxiter, l3_maxiter);


if ~exist(p.savefilename)
    options = p.l1.mF_options;
    trphase1;
    p.l2.nmph = 1; trphase2;
else
    data = load(p.savefilename)
    assert(isfield(data, 'layer1'));
    if ~isfield(data, 'layer2nm')
        p.l2.nmph = 1;  trphase2;
        % [strip layer3] p.l3.nmph = 1;  trphase3;
    elseif ~isfield(data, 'layer3nm') && p.num_layers == 3
        % [strip layer3] p.l3.nmph = 1;  trphase3;
    end
end

clear l1patches l2patches;

fprintf('training ended---\n');
test_conv2l_phase_spm_large
