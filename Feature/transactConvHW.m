function [res, numsamplesper_spcolH, numsamplesper_spcolW, ...
          numsamplesper_tpcol] = transactConvHW(data, func, dim_p, cfovea_spsizeH, cfovea_spsizeW, cfovea_tpsize, sfovea_spsizeH, sfovea_spsizeW, sfovea_tpsize, spstrideH, spstrideW, tpstride, display_subsample)

% function: apply function 'func' on vectorized (col example) form
% of convolutional samples (for instance, grid sampled patches in
% an image or grid sampled blocks in a video cube)

% this function is customized for small image but large batch
% processing and has a for-loop over the convolutional samples    
    
% INPUTS
% data: vectorized image/video cube image_dimension X num_images
% func: function to apply on receptive field samples  
% dimp_p: dimension of func output
% cfovea_spsizeH/W: height and width of the image/video cube
% cfovea_tpsize: temporal size (1 for gray, 3 for color images, t
% for videos)
% sfovea_spsizeH/W: height and width of receptive field size
% sfovea_tpsize: temporal size of receptive field
% spstrideH/W: spatial convolution stride in height and width directions
% tpstride: temporal convolution stride
    
% OUTPUT
% matrix dim_p*num_convolution_samples_per_image/video X num_images
% the first dimension of the matrix is concactenated as follows:
% all features of the first response map position
% all features of the second response map position
% and so on    

if ~exist('display_subsample', 'var'), display_subsample = 1; end
% data = single(data);

% A combination of data transform and activations, fast & mememory efficient
%% verify data fits specs
num_patches_cfovea = size(data,2);

%% clarify simple fovea subsamples in a complex fovea patch
numsamplesper_spcolH = floor((cfovea_spsizeH - sfovea_spsizeH)/spstrideH+1);
numsamplesper_spcolW = floor((cfovea_spsizeW - sfovea_spsizeW)/spstrideW+1);
numsamplesper_tpcol = floor((cfovea_tpsize - sfovea_tpsize)/tpstride+1);

num_subsamples_cfovea = numsamplesper_spcolH*numsamplesper_spcolW*numsamplesper_tpcol;
patch_numel_sfovea = sfovea_spsizeH*sfovea_spsizeW*sfovea_tpsize;

%% initialize output
res = zeros(num_subsamples_cfovea*dim_p, size(data, 2));
%res = zeros(num_subsamples_cfovea*dim_p, size(data, 2), 'single');

%% reshape data into string of squares
data = reshape(data, [cfovea_spsizeH, size(data, 2)*cfovea_spsizeW*cfovea_tpsize]);

if display_subsample, fprintf('processing subsample: '); end

%% create a run-thru index
index = (1:size(data, 2))-1;
for it = 0: num_subsamples_cfovea-1
    if display_subsample, fprintf('%d ', it+1); end
    [y, x, t] = ind2sub([numsamplesper_spcolH,numsamplesper_spcolW, numsamplesper_tpcol], it+1);
    
    starty = (y-1)*spstrideH+1; % starting position of current subsample
    startx = (x-1)*spstrideW+1;
    startt = (t-1)*tpstride*cfovea_spsizeW+1;
    
    taily = starty -1 + sfovea_spsizeH;

    x_filter_spatial = (mod(index, cfovea_spsizeW)>=startx-1)&(mod(index, cfovea_spsizeW)< startx+sfovea_spsizeW-1);
    x_filter_temporal = (mod(index, cfovea_spsizeH*cfovea_tpsize)>=startt-1)&(mod(index, cfovea_spsizeH*cfovea_tpsize)<startt+cfovea_spsizeH*sfovea_tpsize-1);
    
    x_filter = logical(x_filter_spatial.*x_filter_temporal);

    %% use this to verify x_filter:
    %tailx = startx - 1 + sfovea_spsizeW;
    %fprintf('x = %d, y = %d, t = %d, rows: %d to %d \n', x, y, t, startx, tailx)
    %for i = 0:cfovea_tpsize-1
    %    fprintf('    cols in cfovea square: %d to %d\n', find(x_filter(i*cfovea_spsizeW+1:(i+1)*cfovea_spsizeW),1, 'first'), find(x_filter(i*cfovea_spsizeW+1:(i+1)*cfovea_spsizeW),1, 'last'))
    %end

    %% data_sub: vectorized subsample patches indexed by 'it' 
    %size of data_sub: sfovea_spsize x (sfovea_spsize x sfovea_tpsize x num_patches_cfovea)
    
    %% activate function to get outputs
    res(blkidx(dim_p, it),:) = func(reshape(data(starty:taily, x_filter), patch_numel_sfovea, num_patches_cfovea));

end

if display_subsample, fprintf('\n'); end
end
