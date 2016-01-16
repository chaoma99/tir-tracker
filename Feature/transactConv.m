function res = transactConv(data, func, dim_p, cfovea_spsize, cfovea_tpsize, sfovea_spsize, sfovea_tpsize, spstride, tpstride, use_single)
    
    if exist('use_single', 'var')
        data = single(data);
    end
    
    % A combination of data transform and activations, fast & mem
    % memory efficient
    
    %% verify data fits specs
    num_patches_cfovea = size(data,2);
    
    %% clarify simple fovea subsamples in a complex fovea patch
    numsamplesper_spcol = floor((cfovea_spsize - sfovea_spsize)/spstride+1);
    numsamplesper_tpcol = floor((cfovea_tpsize - sfovea_tpsize)/tpstride+1);
    
    num_subsamples_cfovea = numsamplesper_spcol^2*numsamplesper_tpcol;
    patch_numel_sfovea = sfovea_spsize^2*sfovea_tpsize;
    
    %% initialize output
    if exist('use_single', 'var')
        res = zeros(num_subsamples_cfovea*dim_p, size(data, 2), 'single');
    else
        res = zeros(num_subsamples_cfovea*dim_p, size(data, 2));
    end
    
    %% reshape data into string of squares
        
    data = reshape(data, [cfovea_spsize, size(data, 2)*cfovea_spsize*cfovea_tpsize]);
    
    %% create a run-thru index
    index = (1:size(data, 2))-1;
    
    
    for it = 0: num_subsamples_cfovea-1
        %    fprintf('processing subsample %d\n', it+1);
        [x, y, t] = ind2sub([numsamplesper_spcol,numsamplesper_spcol, numsamplesper_tpcol], it+1);
        startx = (x-1)*spstride+1;
        starty = (y-1)*spstride+1;
        startt = (t-1)*tpstride*cfovea_spsize+1;
        
        tailx = startx -1 + sfovea_spsize;
        
        y_filter_spatial = (mod(index, cfovea_spsize)>=starty-1)&(mod(index, cfovea_spsize)< starty+sfovea_spsize-1);
        y_filter_temporal = (mod(index, cfovea_spsize*cfovea_tpsize)>=startt-1)&(mod(index, cfovea_spsize*cfovea_tpsize)<startt+cfovea_spsize*sfovea_tpsize-1);
        
        y_filter = logical(y_filter_spatial.*y_filter_temporal);
        
        %% use this to verify y_filter:
        %     fprintf('x = %d, y = %d, t = %d, rows: %d to %d \n', x, y, t, startx, tailx)
        %     for i = 0:cfovea_tpsize-1
        %       fprintf('    cols in cfovea square: %d to %d\n', find(y_filter(i*cfovea_spsize+1:(i+1)*cfovea_spsize),1, 'first'), find(y_filter(i*cfovea_spsize+1:(i+1)*cfovea_spsize),1, 'last'))
        %     end
        
        %% data_sub: vectorized subsample patches indexed by 'it' 
        %size of data_sub: sfovea_spsize x (sfovea_spsize x sfovea_tpsize x num_patches_cfovea)
        
        %% activate function to get outputs
        res(blkidx(dim_p, it),:) = func(reshape(data(startx:tailx, y_filter), patch_numel_sfovea, num_patches_cfovea));
        
    end    
end
