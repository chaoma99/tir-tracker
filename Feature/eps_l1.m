function [f, df] = eps_l1(x)

f = sum(sum(sqrt(1e-7 + x.^2)));

df = x./sqrt(1e-7+x.^2);
end
