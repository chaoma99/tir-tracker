function result = conv_contrast_normalize(data, a, b, sz, type)

if ~exist('type', 'var')
  type = 'divisive';
end

% an implementation of contrast normalization in a convolutional network
% the input data is give in the form of a matrix of size: 
% num_images*spatial_num X num_features
% where spatial_num is numel(response_map)

% a and b are sides of response map

fdim1 = size(data, 2); 

% remove mean from the response maps; [DOES NOT WORK]
% data = reshape(data, a^2, []); data = bsxfun(@minus, data, mean(data, 1)); data = reshape(data, [], fdim1);

result = zeros(size(data)); 

% define gaussian kernel
gw = fspecial('gaussian', sz, sz);


if strcmp(type, 'divisive')
sumgd = 0;
for i = 1: fdim1
%    fprintf('%d ', i);
    response_maps = reshape(data(:, i), a, b, []);  % for all frames
    % filter with gaussian kernel
    gd = imfilter(response_maps.^2, gw, 'symmetric', 'same', 'corr');
    sumgd = sumgd + gd;
end
fprintf('\n');

sigma = sqrt(sumgd); 
c = mean(sigma(:)); 
sigma = max(c, sigma)+1e-6; 

sigma = reshape(sigma, [], 1);

result = data; 
result = bsxfun(@rdivide, data, sigma);

else  % subtractive

sumgd = 0;
for i = 1: fdim1
    fprintf('%d ', i);
    response_maps = reshape(data(:, i), a, b, []);  % for all frames
    % filter with gaussian kernel
    gd = imfilter(response_maps, gw, 'symmetric', 'same', 'corr');
    sumgd = sumgd + gd;
end
fprintf('\n');
sumgd = sumgd/fdim1; 

sumgd = reshape(sumgd, [], 1);

result = data; 
result = bsxfun(@minus, data, sumgd);

end
