function [cost,grad] = ricaC_gs2(theta, x, params)

%softICA cost with group size
addpath ~/s9/changping/

global usegpu;
H = params.H;

W = reshape(theta, params.numFeatures, params.n);
if usegpu
    W = gsingle(W);
end
Wold = W;
W = l2rowscaled(W, 1);

%norm(W'*W*x-x,'fro')^2 
%sum(sum(log(cosh(W*x))))

% fwprop
z = W*x;
h = H*z.^2;
p = sqrt(1e-7+h);

if isfield(params, 'skipframes');
    if params.skipframes ~= 1;
        [pullC, pullD] = pull_loss_epsl1_frames(p, params.skipframes);
    else
        [pullC, pullD] = pull_loss_epsl1(p);
    end
else
    [pullC, pullD] = pull_loss_epsl1(p);
end

%[pushC, pushD] = push_loss(p);
slowC = pullC; %+params.push_ratio*pushC;
slowD = pullD; %+params.push_ratio*pushD;

cost = norm(W'*z-x,'fro')^2 + params.gamma*sum(p(:)) + params.lambda*slowC + params.weightcost*sum(sum(W.^2));

%bkprop for pooling

outderv = params.gamma*ones(size(p)) + params.lambda*slowD;
outderv = outderv.*(0.5./p);
outderv = H'*outderv;
sparsedf = outderv.*z*2; % + params.phaselambda*phaseD;

grad = (2*(z*z')*W + 2*(W*W')*z*x' - 4*z*x') + sparsedf*x' +params.weightcost*2*W;

grad = l2rowscaledg(Wold, W, grad, 1);

grad = double(grad(:));
