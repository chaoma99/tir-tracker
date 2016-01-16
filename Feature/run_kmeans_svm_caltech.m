addpath ~/scratch/kmeans_demo/
addpath ~/scratch/autoencoder/minFunc/

sum_acc = 0; 
num_trials = 10; 

% rand seed
rand('state', 10);
randn('state', 10);

for trial = 1:num_trials

% form train/test splits
trf = formtrf(Y, 30);
tef = formtef(Y, trf, 50); 

assert(sum(trf.*tef)==0);
trainXCs = ftrain(trf, :); %clear ftrain;
testXCs = ftrain(tef, :); %clear ftest;
trainY = Y(trf);
testY = Y(tef);

trainXC_mean = mean(trainXCs);
trainXC_sd = sqrt(var(trainXCs)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXCs, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

%train classifier using SVM
%C = 1000;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXCs, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
test_acc = 100 * (1 - sum(labels ~= testY) / length(testY));
fprintf('Test accuracy for trial %d is %f%%\n', trial, test_acc);

sum_acc = sum_acc + test_acc;
end

ave_acc = sum_acc/num_trials; 

fprintf('---- average accuracy across %d trials ----\n', num_trials); 
fprintf('---- %f ----\n', ave_acc);
