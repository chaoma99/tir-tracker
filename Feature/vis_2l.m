% script to visualize the layer 2 phase changes -> other invariance

% specify bases filename
% filename = ['./bases/caltech/convl3_usetrack1_trdavanH_tedacoil_l1pch16_l2pch32_l3pch32_l1str2_l2str2_l3str2_l1nf256_l2nf300_l3nf300_l2pca300_l3pca300_l1gam50.000_l1lamb150.000_l2gam1.000_l2lamb500.000_l3gam1.000_l3lamb500.000_wc0.00000_l1iter750_l2iter1500_l3iter700.mat']
% 
% load(filename);

load('I:\Chao Ma\Tracking\nips12_willzhou\bases\caltech\convl3_usetrack1_trdavanH_tedacaltech_l1pch16_l2pch32_l3pch32_l1str2_l2str2_l3str2_l1nf256_l2nf300_l3nf300_l2pca300_l3pca300_wc0.00000_l1iter750_l2iter1500_l3iter1500.mat')

w = layer2nm.W*layer2nm.PCAfilter; 
W = layer1.W*layer1.ZCAfilter; 
nm = sqrt(W(1:2:end, :).^2 + W(2:2:end, :).^2);

imsz = 32;
side = (imsz-16)/2+1;

vis_patches = zeros(size(w, 1), imsz*imsz);

for j = 1:size(w, 1)
    p = zeros(imsz, imsz);
    
    for i = 1:side^2
        [a, b] = ind2sub([side, side], i);
        ws = w(j, (i-1)*128+1:128*i);
        
        cur = reshape(ws*nm, 16, 16);
        
        p((a-1)*2+1:2*(a-1)+16, 2*(b-1)+1:2*(b-1)+16) = p((a-1)*2+1:2*(a-1)+16, 2*(b-1)+1:2*(b-1)+16) + cur;
        
    end
    vis_patches(j, :) = p(:)';
end

N_32 = sqrt(vis_patches(1:2:end, :).^2 + vis_patches(2:2:end, :).^2); 

fig = figure(1); 
c = vis_patches(1:2:end, :) + 1i*vis_patches(2:2:end, :); 

N = 20;
for k = 1:5
    for j = 0: N-1
        j
        mult = sin(j/N*2*pi) + 1i*cos(j/N*2*pi);
        cc = c*mult;     
        pf(real(cc)', 16)
        title({'Visualize second layer invariance by changing'; ...
               'interpolation angle between two linear bases:'; ...
               'Each 32x32 pixel square is the visualization'; ...
               'of one second layer feature (after pooling);'; ...
               'The second layer pooled feature is invariant to'; ...
               'various type of motion seen in this visualization'});
    end
end
