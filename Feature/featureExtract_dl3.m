
function [feature]=featureExtract_dl3(im,step)

%%%% featureExtract

% preset_params; 
% 
% % ------ [section3] set savename and run training ------
% % p.basespath = sprintf('./bases/%s', p.testdataset);
% 
% p.savefilename = sprintf('%s/convl3_usetrack%d_trda%s_teda%s_l1pch%d_l2pch%d_l3pch%d_l1str%d_l2str%d_l3str%d_l1nf%d_l2nf%d_l3nf%d_l2pca%d_l3pca%d_wc%.5f_l1iter%d_l2iter%d_l3iter%d.mat', ...
%                          p.basespath, p.use_track, p.traindataset, p.testdataset, p.l1.params.patchWidth, p.l2.params.patchWidth, p.l3.params.patchWidth, ...
%                          p.l1.stride, p.l2.stride, p.l3.stride, p.l1.params.numFeatures, p.l2.params.numFeatures, p.l3.params.numFeatures, p.l2.params.pca_dim, p.l3.params.pca_dim, ...
%                          0, l1_maxiter, l2_maxiter, l3_maxiter);
% 

% step=4;

network = load('convl3_usetrack1_trdavanH_tedacaltech_l1pch16_l2pch32_l3pch32_l1str2_l2str2_l3str2_l1nf256_l2nf300_l3nf300_l2pca300_l3pca300_wc0.00000_l1iter750_l2iter1500_l3iter1500.mat');

p=network.p;

assert(isfield(network, 'layer1'));

% aa=imread('rice.png');

% aa=double(imresize(aa,[64 64]));

 aa=im2double(im);

s_sp_size = p.l1.params.patchWidth;   % layer 1 (small) receptive field size
s_tp_size = p.numch; % number of channels (gray->1)
c_sp_size = sqrt(size(numel(aa), 1)); % layer 2 (large) receptive field size
c_tp_size = p.numch; % number of channels (gray->1)

% -- pool size for the three layers --
% psz1 = p.l1.psz;
% psz2 = p.l2.psz;
% psz3 = p.l3.psz;

% -- stride --
% sp_stride = p.l1.stride;    % layer 1 stride (on image)
sp_stride=step;
tp_stride = 1;

l2_sp_stride = p.l2.stride; % layer 2 stride (on layer 1 response map)
l2_fovea_size = (p.l2.params.patchWidth-p.l1.params.patchWidth)/sp_stride + 1;


% l3_sp_stride = p.l3.stride; % layer 3 stride (on layer 2 response map)
% l2_fovea_size = (p.l2.params.patchWidth-p.l1.params.patchWidth)/sp_stride + 1;
% l3_fovea_size = (p.l3.params.patchWidth-p.l2.params.patchWidth)/p.l3.strideraw + 1;

% -- constrast normalization: window size --
cnorm_sz = p.l2.cnorm_sz;

l1_patch_num=fix((size(aa)-p.l1.params.patchWidth)/sp_stride) + 1;
% l2_patch_num=fix((l1_patch_num-l2_fovea_size)/l2_sp_stride) + 1;
%perform convol

x1=im2colstep(aa,[s_sp_size s_sp_size],[sp_stride sp_stride]);
% x1=bsxfun(@minus,x1,mean(x1,1));
x1=stdnormalize(x1);

ft = network.layer1.W*network.layer1.ZCAfilter*x1;

rr=ft(1:2:end,:);
II=ft(2:2:end,:);

% mag=min(abs(rr+1i*II),10);
l1onm=abs(rr+1i*II);
pha=angle(rr+1i*II);

l1onm=reshape(l1onm', [l1_patch_num, size(l1onm,1)]);
mag=conv_contrast_normalize_mc(l1onm,cnorm_sz,size(l1onm,3));
% 
pha=reshape(pha', [l1_patch_num, size(pha,1)]);
pha=conv_contrast_normalize_mc(pha,cnorm_sz,size(pha,3));

% feature=cat(3,mag, pha);

x2=im2colstep(l1onm,[9 9 size(l1onm,3)],[1 1 1]);
ft2 = network.layer2nm.W*network.layer2nm.PCAfilter*x2; 
 
rr=ft2(1:2:end,:);
II=ft2(2:2:end,:);

mag2=abs(rr+1i*II);
pha2=abs(rr+1i*II);


mag2=reshape(mag2', [l1_patch_num-8, size(mag2,1)]);
pha2=reshape(pha2', [l1_patch_num-8, size(pha2,1)]);

mag2=imPad(mag2,[4,4],0);
pha2=imPad(pha2,[4 4],0);

feature=cat(3,mag, pha, mag2); %, pha2);


% gw=ones(3*3)/9;

% feature=imfilter(feature,gw,'replicate', 'same');
% 
% feature=feature(1:2:end,1:2:end,:);

% feature= conv_contrast_normalize_mc(feature, cnorm_sz, size(feature,3));

% feature= conv_contrast_normalize_mc(feature, cnorm_sz, size(feature,3),'subtractive');

% feature=pha;


% ndim=8;
% 
% pha=fix((pha+pi)/(2*pi)*ndim)+1;
% pha(pha==ndim)=1;
% 
% feature=zeros(ndim,size(mag,2));
% 
% for kk=1:ndim
%     temp=zeros(size(mag));
%     temp(pha==kk)=mag(pha==kk);
%     temp=reshape(temp,size(mag));
%     feature(kk,:)=sum(temp);
% end
% 
% feature=bsxfun(@rdivide,feature,sqrt(sum(feature.^2)));
% feature=reshape(feature',[l1_patch_num,ndim]);

% % p = single(network.layer1.W)*single(network.layer1.ZCAfilter)*single(x);
% r = p(1:2:end, :); 
% I = p(2:2:end, :); 
% ph = angle(r + 1i*I);
% 
% act = ph;

% l1onm=actHslow_norm(x1,network,1);
% 
% fdim1=size(l1onm,1);
% 
% l1oph=actHslow_phase(x1,network,1);
% 
% fdim1=size(l1oph,1);
% l1onm=stdnormalize(l1onm);
% 
% l1onm=reshape(l1onm', [l1_patch_num,fdim1]);

% x2=im2colstep(l1onm,[l2_fovea_size l2_fovea_size fdim1],[l2_sp_stride l2_sp_stride 1]);

% l2onm=actHslow_PCA(x2,network.layer2nm,1);

% fdim2=size(l2onm,1);

% l1onm = reshape(l1onm, fdim1, [])';
% l1onm = conv_contrast_normalize_mc(l1onm, cnorm_sz, fdim1,'subtractive');

% gw=ones(2*2)/4;
% 
% l1onm=imfilter(l1onm,gw,'replicate', 'same', 'conv');

% l1onm=l1onm(1:2:end,1:2:end,:);

% l1oph = reshape(l1oph', [l1_patch_num, fdim1]);

% l1oph = conv_contrast_normalize_mc(l1oph, cnorm_sz, fdim1,'subtractive');

% l1oph=imfilter(l1oph,gw,'replicate', 'same', 'corr');

% l2onm = reshape(l2onm', [l2_patch_num, fdim2]);
% l2onm = conv_contrast_normalize_mc(l2onm, cnorm_sz, fdim2);


% a1=l1_patch_num(1); a2=l1_patch_num(2);
% b1=l2_patch_num(1); b2=l2_patch_num(2);
% 
% 
% 
% %% SPM meanfilters for layer 2: [hard code a 3 layer SPM]
% meanf_M2 = create_filter_matrix(b1, b2, min(floor(l2_patch_num/4)));
% meanf_M2_a = create_filter_matrix(b1, b2, min(floor(l2_patch_num/2)));
% meanf_M2_b = create_filter_matrix(b1, b2, min(l2_patch_num));
% 
% l1onm_0 = meanf_M*l1onm;
% l1oph_0 = meanf_M*l1oph;
% 
% l1onm_a = meanf_M_a*l1onm;
% l1oph_a = meanf_M_a*l1oph;
% 
% l1onm_b = meanf_M_b*l1onm;
% l1oph_b = meanf_M_b*l1oph;
% 
% l2onm_0 = meanf_M2*l2onm;
% 
% l2onm_a = meanf_M2_a*l2onm;
% 
% l2onm_b = meanf_M2_b*l2onm;
% 
% ffinal = [l1onm_0(:); l1onm_a(:); l1onm_b(:); l1oph_0(:); l1oph_a(:); l1oph_b(:); l2onm_0(:); l2onm_a(:); l2onm_b(:)];

end