clear;
clc;
close all;
[x, t] = wine_dataset;

%import data
Xtrain= csvread("C:\Users\pc\Xtrain.csv")
Ytrain= csvread("C:\Users\pc\Ytrain.csv")
Xtest= csvread("C:\Users\pc\Xtest.csv")
Ytest= csvread("C:\Users\pc\Ytest.csv")

%transpose
Xtrain=Xtrain.'
Ytrain=Ytrain.'
Xtest=Xtest.'
Ytest=Ytest.'

%Randomly choose the first initial guess
%hidden layer 10 neurons, output 1 neuron
IW0 = 0.1*randn(20, 5);
b10 = 0.1*randn(20, 1);
LW0 = 0.1*randn(1, 20);
b20 = 0.1*randn(1, 1);


[global_best_wb, global_best_error,test_error, net]= local_optimizer(Xtrain, Ytrain,Xtest,Ytest, b10, b20, IW0, LW0);


best_test_error=[]

best_test_error=[best_test_error,test_error]
tier0_wb = global_best_wb;
prev_error = global_best_error;
current_error = 1;
k = size(tier0_wb);
%Using several arrays to record the changing process of the global best performance
%and the performances of the tier-1 local optimums
global_perf = [];
local_perf = [];
local_best=[];%test
global_perf = [global_perf, global_best_error];
local_perf = [local_perf, global_best_error];
local_best=[local_best,test_error];
%using an array to store the weights and biases of each local minimum
local_wb = {};
local_wb = [local_wb, global_best_wb];

for i = 1:k(1)
perturb_wb = tier0_wb;
while current_error>prev_error
prev_error = current_error;
perturb_wb(i) = perturb_wb(i)+0.01;
[b, iw, lw] = separatewb(net, perturb_wb);
b1 = b{1,1};
b2 = b{2,1};
IW = iw {1,1};
LW = lw{2,1};
current_error = get_error(b1, b2, IW, LW, Xtrain, Ytrain, net);
end
[b, iw, lw] = separatewb(net, perturb_wb);
b1 = b{1,1};
b2 = b{2,1};
IW = iw {1,1};
LW = lw{2,1};
[local_optimum_wb, local_optimum_error, local_test_error,net] = local_optimizer(Xtrain, Ytrain,Xtest,Ytest, b1, b2, IW, LW);
local_perf = [local_perf, local_optimum_error];
local_wb = [local_wb, local_optimum_wb];
local_best=[local_best,local_test_error];
if local_optimum_error < global_best_error
global_best_wb = local_optimum_wb;
global_best_error = local_optimum_error;
end
global_perf = [global_perf, global_best_error];
[b, iw, lw] = separatewb(net, global_best_wb);
b1 = b{1,1};
b2 = b{2,1};
IW = iw {1,1};
LW = lw{2,1};
test_error=get_error(b1, b2, IW, LW, Xtest, Ytest, net);
best_test_error=[best_test_error,test_error]
end

