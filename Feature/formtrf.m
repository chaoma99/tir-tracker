function trf = formtrf(Y, n)

% function to generate random training index filter for caltech 101
% chose random n images from each class for training
% during test, these n images are left out

num_labels = max(Y);
trf = false(numel(Y), 1);

count = 0;

for i = 1: num_labels
  inc = sum(Y == i);  % number of images with label i

  rand_series = false(inc, 1);
  rndidx = randperm(inc);
  rand_series(rndidx(1:n)) = true;

  trf(count +1: count + inc) = rand_series;

  count = count + inc;
  assert(Y(count) == i);
  if count~=numel(Y), assert(Y(count+1) == i+1); end
end
