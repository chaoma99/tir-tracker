function [x, mx, stdx] = stdnormalize(x)

mx = mean(x(:)); 
stdx = (std(x(:))+1e-6); 
x = x - mx; % minus over all mean
x = x./stdx; % divide by overall std
