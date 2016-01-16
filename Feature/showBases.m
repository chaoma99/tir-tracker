function [] = showBases(theta, infoStruct, stuff, varargin)
global params
if mod(infoStruct.iteration,10) == 0
    W = reshape(theta, params.numFeatures, params.n);
    subplot(1, 2, 1); pf(W', round(sqrt(size(W, 1)))); drawnow;
    real = W(1:2:end, :);
    imag = W(2:2:end, :); 
    N = sqrt(real.^2+imag.^2); 
    subplot(1, 2, 2); pf(N', 8); drawnow;
end
