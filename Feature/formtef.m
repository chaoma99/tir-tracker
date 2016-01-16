function tef = formtef(Y, trf, n)

% function to generate random testing index filter for caltech 101
% chose random n images from each class for classification test
% with the previously selected training images left out
% if there are less than n images left in the class, all left images are used for test

num_labels = max(Y);
tef = false(numel(Y), 1); 

count = 0;

indices = 1:numel(Y); % create indices

for i = 1: num_labels
  inc = sum(Y == i);  % number of images with label i 
  
  smalltrf = trf(count+1:count+inc);         % selected train indices in label group i
  smallindices = indices(count+1:count+inc); % all indices in label group i
  smallteindices = smallindices(~smalltrf);  % potential test indices in label i group
  if numel(smallteindices) >=n
  rndidx = randperm(numel(smallteindices));  % randomly select n
  selected_teindices = smallteindices(rndidx(1:n)); % finally selected indices for test in label group i
  else
    %fprintf('selecting all indices\n');
    selected_teindices = smallteindices;
  end
  tef(selected_teindices) = true; 

  count = count + inc;
end
