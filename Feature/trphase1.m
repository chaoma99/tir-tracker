%% ----- transfer preset params ----
params = p.l1.params; 
options = p.l1.mF_options;

%% ----- std normalize data ---- 
x = l1patches; clear l1patches;
[x, mx, stdx] = stdnormalize(x);

%% ----- Whitening -----
epsilon = 1e-4;        % regularization for whitening
sigma = x * x' / size(x, 2);
[U,S,V] = svd(sigma);
ZCAfilter = U * diag(1./sqrt(diag(S)+epsilon)) * U';
xw = ZCAfilter*x; clear x; 

%% Run optimization
if ~exist('no_vis_bases', 'var')
    options.outputFcn = 'showBases';
end

randTheta = randn(params.numFeatures*params.n,1)*1/sqrt(params.n);
warning off

[opttheta, cost, exitflag] = minFunc( @(theta)ricaC_gs2(theta, xw, params), randTheta, options); % Use x or xw

layer1.W = reshape(opttheta, params.numFeatures, params.n);
layer1.ZCAfilter = ZCAfilter;
layer1.mx = mx;
layer1.stdx = stdx;
p.l1.params
save(p.savefilename, 'layer1', 'p');
clear xw params
