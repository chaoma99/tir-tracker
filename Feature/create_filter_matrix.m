function meanf_M = create_filter_matrix(a, b, psz)

% function: create a matrix, when multiplied with the col example
% vectorized response map, performs mean pool and subsampling
% deals with zero-padded im2col result

index = 1:a*b;

cols = im2col(reshape(index, [a, b]), [psz, psz], 'distinct');

% figure
meanf_M = zeros(size(cols, 2), a*b);
zero_img = zeros(a, b);
for i = 1:size(cols, 2)
    template = zero_img;
    filter = cols(:, i);
    filter = filter(filter~=0);
    template(filter) = 1/numel(filter);
    %     imagesc(template); pause(0.5);
    meanf_M(i, :) = template(:)';
end
