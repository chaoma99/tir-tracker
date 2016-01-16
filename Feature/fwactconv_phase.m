function act = fwactconv_phase(input, p)

% activation function output convolutional output
% for a small convnet component

network = load(p.savefilename);

s_sp_size = p.l1.params.patchWidth;
s_tp_size = p.numch;
c_sp_size = p.l2.params.patchWidth;
c_tp_size = p.numch;
sp_stride = p.l1.stride;
tp_stride = p.numch;

switch p.l2.nmph
case 0
    act_func1 = @(x)actHslow(x, network, 1);
case 1
    act_func1 = @(x)actHslow_norm(x, network, 1, Inf); 
    % hard code thresh to be 10
case 2
    act_func1 = @(x)actHslow_phase(x, network, 1);
end

fdim = size(act_func1(randn(s_sp_size^2*s_tp_size, 10)), 1);
conv_func1 = @(in)transactConv(in, act_func1, fdim, c_sp_size, c_tp_size, s_sp_size, s_tp_size, sp_stride, tp_stride);

act = conv_func1(input);
