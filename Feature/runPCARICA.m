%% add this in?
%x = bsxfun(@minus, x, mean(x, 1));    % mean normalization
%x = bsxfun(@rdivide, x, std(x, 1)+0.1);         % variance normalization

%% Whitening
epsilon = 1e-4;       % regularization for whitening
[V, E, D] = pca_mod(x, epsilon); % 
PCAfilter = V(1:params.pca_dim, :);
xw = PCAfilter *x;
clear x;
%PCAfilter = randn(1:params.pca_dim, size(x, 1));
%xw = x(1:params.pca_dim, :);

%% Run optimization
randTheta = randn(params.numFeatures*params.n,1)*1/sqrt(params.n);
warning off
[opttheta, cost, exitflag] = minFunc( @(theta) ricaC_gs2(theta, xw, params), randTheta, options);
clear xw;
