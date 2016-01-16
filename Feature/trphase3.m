%----- std normalize data -----
l3patches = stdnormalize(l3patches);

% ------- load trained bases ------
fprintf('--- forward activate layer1 conv ---\n');
x = fwact2lconv_rica(l3patches, p);

[x, mx, stdx] = stdnormalize(x);

%% --- set params ---
clear options
params = p.l3.params;
options = p.l3.mF_options;
fprintf(' --- running pca and rica --- \n');

runPCARICA

%% ------- save bases etc in two layers --------
fprintf('---- saving results -----\n');
p.l3.params
switch p.l3.nmph
case 0
layer3all.W = reshape(opttheta, params.numFeatures, params.pca_dim);
layer3all.PCAfilter = PCAfilter;
save(p.savefilename, 'layer3all', 'p', '-append');
case 1
layer3nm.W = reshape(opttheta, params.numFeatures, params.pca_dim);
layer3nm.PCAfilter = PCAfilter;
layer3nm.mx = mx; 
layer3nm.stdx = stdx; 
save(p.savefilename, 'layer3nm', 'p', '-append');
case 2
layer3ph.W = reshape(opttheta, params.numFeatures, params.pca_dim);
layer3ph.PCAfilter = PCAfilter;
layer3ph.mx = mx;
layer3ph.stdx = stdx;
save(p.savefilename, 'layer3ph', 'p', '-append');
end
