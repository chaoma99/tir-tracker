% script to visualize the layer 1 phase changes -> translation invariance 

% specify bases filename 
% filename = ['./bases/caltech/convl3_usetrack1_trdavanH_tedacoil_l1pch16_l2pch32_l3pch32_l1str2_l2str2_l3str2_l1nf256_l2nf300_l3nf300_l2pca300_l3pca300_l1gam50.000_l1lamb150.000_l2gam1.000_l2lamb500.000_l3gam1.000_l3lamb500.000_wc0.00000_l1iter750_l2iter1500_l3iter700.mat'] 
% 
% load(filename); 

load('I:\Chao Ma\Tracking\nips12_willzhou\bases\caltech\convl3_usetrack1_trdavanH_tedacaltech_l1pch16_l2pch32_l3pch32_l1str2_l2str2_l3str2_l1nf256_l2nf300_l3nf300_l2pca300_l3pca300_wc0.00000_l1iter750_l2iter1500_l3iter1500.mat')


fig = figure(1); 

c = layer1.W(1:2:end, :) + 1i*layer1.W(2:2:end, :); 

N = 20; 
for k = 1:5 
    for j = 0: N-1 
        j 
        mult = sin(j/N*2*pi) + 1i*cos(j/N*2*pi); 
        cc = c*mult; 
        pf(real(cc)', 16) 
        title({'Visualize first layer invariance by changing'; ...
               'interpolation angle between two linear bases:'; ...
               'Each 16x16 pixel square is the visualization'; ...
              'of one first layer feature (after pooling);'; ...
            'The first layer pooled feature is invariant to'; ...
              'local translation motion seen in this visualization'}); 
    end 
end 
