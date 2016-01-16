function [f, df] = pull_loss_epsl1(x)

tdiff = x(:, 1:end-1)-x(:, 2:end);

[f, edf] = eps_l1(tdiff); 

f = f - eps_l1(tdiff(:, 20:20:end));

df1 = [edf, zeros(size(x, 1), 1)];

df1(:, 20:20:end) = 0;

df2 = [zeros(size(x, 1), 1), -edf];

df2(:, 21:20:end) = 0;

df = df1 + df2;

end
