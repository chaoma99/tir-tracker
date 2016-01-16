%----- std normalize data -----
l2patches = stdnormalize(l2patches);

% ------- load trained bases ------
fprintf('--- forward activate layer1 conv ---\n');
x = fwactconv_phase(l2patches, p);

[x, mx, stdx] = stdnormalize(x);

%% --- set params ---
clear options
params = p.l2.params;
options = p.l2.mF_options;
fprintf(' --- running pca and rica --- \n');

runPCARICA

%% ------- save bases etc in two layers --------
fprintf('---- saving results -----\n');
p.l2.params
switch p.l2.nmph
case 0
layer2all.W = reshape(opttheta, params.numFeatures, params.pca_dim);
layer2all.PCAfilter = PCAfilter;
save(p.savefilename, 'layer2all', 'p', '-append');
case 1
layer2nm.W = reshape(opttheta, params.numFeatures, params.pca_dim);
layer2nm.PCAfilter = PCAfilter;
layer2nm.mx = mx; 
layer2nm.stdx = stdx; 
save(p.savefilename, 'layer2nm', 'p', '-append');
case 2
layer2ph.W = reshape(opttheta, params.numFeatures, params.pca_dim);
layer2ph.PCAfilter = PCAfilter;
layer2ph.mx = mx; 
layer2ph.stdx = stdx;
save(p.savefilename, 'layer2ph', 'p', '-append');
end
