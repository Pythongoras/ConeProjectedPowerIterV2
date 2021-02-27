
% addpath(genpath('/Users/yufei/Documents/2-CMU/PowerIter'))
%% Set up

clear; close all; clc
mat = csvread('so2_15to18.csv', 1, 1);
disp(size(mat))

% Take the difference
% mat = diff(mat, 1, 2);
% disp(size(mat))

% Use the first half as train set.
n_train = 291;
mat = mat(:, 1:n_train);
disp(size(mat))

% Normalize
% mat = normalize(mat, 2);

% Get the cov matrix of train set
C = cov(mat);


%% Hyperparameter tuning for ElasticNet SPCA

% Set up the grid of the possible lambdas
elas_lambda_list = linspace(10,100,10);

% Create the container of the estimated eigenvec under diff hyperparameters
v_elas_package = zeros(size(C, 1), length(elas_lambda_list));

% Estimate eigenvecs
for i = 1:length(elas_lambda_list)
    v_elas_package(:, i) = elas_spca_func(C, elas_lambda_list(i));
end

% Output matrix
writematrix(v_elas_package, "/Users/yufei/Documents/2-CMU/PowerIter/Data/AirQuality/v_elas_package.txt")


%% Hyperparameter tuning for truncated power iter

% Set up the grid of the possible cardis
trunc_cardi_list = linspace(0.1, 0.99, 10);

% Create the container of the estimated eigenvec under diff hyperparameters
v_trunc_package = zeros(size(C ,1), length(trunc_cardi_list));

% Estimate eigenvecs
for i = 1:length(trunc_cardi_list)
    v_trunc_package(:, i) = power_iter_func(C, @(x) proj_trunc(x,trunc_cardi_list(i)));
end

% Output matrix
writematrix(v_trunc_package, "/Users/yufei/Documents/2-CMU/PowerIter/Data/AirQuality/v_trunc_package.txt")


%% Experiments

%%%%% Monotone cone 
v_mnt = power_iter_func(C, @proj_mnt);
writematrix(v_mnt, "/Users/yufei/Documents/2-CMU/PowerIter/Data/AirQuality/v_mnt.txt")


%%%%% Ordinary power iter 
v_ordi = power_iter_func(C, @proj_ordinary);
writematrix(v_ordi, "/Users/yufei/Documents/2-CMU/PowerIter/Data/AirQuality/v_ordi.txt")


%%%%% Truncated power iter
v_trunc = power_iter_func(C, @(x) proj_trunc(x, 0.99));
writematrix(v_trunc, "/Users/yufei/Documents/2-CMU/PowerIter/Data/AirQuality/v_trunc.txt")


%%%%% ElasticNet SPCA
v_elas = elas_spca_func(C, 10000);
writematrix(v_elas, "/Users/yufei/Documents/2-CMU/PowerIter/Data/AirQuality/v_elas.txt")

