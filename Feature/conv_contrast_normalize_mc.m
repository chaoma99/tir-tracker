function [ result ] = conv_contrast_normalize_mc( data, sz, fdim1,type)

if ~exist('type', 'var')
  type = 'substractive';
end
    
% define gaussian kernel
gw = fspecial('gaussian', sz, sz);

assert(ndims(data)==3,'Dimension of input should be 3');

if strcmpi(type,'divisive')

    gd = imfilter(data.^2, gw, 'symmetric', 'same', 'corr');
    sumgd=sum(gd,3);
    sigma=sqrt(sumgd);
    c = mean(sigma(:)); 
    sigma = max(c, sigma)+1e-6;
%     
%     sigma=sigma(:);
%     data=reshape(data,[],fdim1);
    result = bsxfun(@rdivide, data, sigma);

else % subtractive
    gd=imfilter(data,gw,'symmetric', 'same', 'corr');
    sumgd=sum(gd,3);
    sigma=sumgd/fdim1;
%     sigma=sigma(:);
%     data=reshape(data,[],fdim1);
    result = bsxfun(@minus, data, sigma);
    
end

end