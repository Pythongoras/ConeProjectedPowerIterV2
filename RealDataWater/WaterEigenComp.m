% addpath(genpath('/Users/yufei/Documents/2-CMU/PowerIter'))
%% Hyperparameter tuning set up

clear; close all; clc
mat = csvread('oxygen_all_16171819.csv', 1, 1);
disp(size(mat))

% Take the difference
% mat = diff(mat, 1, 2);
% disp(size(mat))

% Use the first half as train set.
n_train = 372;
mat = mat(:, 1:n_train);
disp(size(mat))

% Normalize
% mat = normalize(mat, 2);

% Get the cov matrix of train set
C = cov(mat);



%% Experiments

%%%%% Monotone cone 
v_mnt = power_iter_func(C, @proj_mnt);
writematrix(v_mnt, "/Users/yufei/Documents/2-CMU/PowerIter/Data/WaterQuality/v_mnt.txt")


%%%%% Ordinary power iter 
v_ordi = power_iter_func(C, @proj_ordinary);
writematrix(v_ordi, "/Users/yufei/Documents/2-CMU/PowerIter/Data/WaterQuality/v_ordi.txt")


%%%%% Truncated power iter
v_trunc = power_iter_func(C, @(x) proj_trunc(x, 0.99));
writematrix(v_trunc, "/Users/yufei/Documents/2-CMU/PowerIter/Data/WaterQuality/v_trunc.txt")


%%%%% ElasticNet SPCA
v_elas = elas_spca_func(C, 10000);
writematrix(v_elas, "/Users/yufei/Documents/2-CMU/PowerIter/Data/WaterQuality/v_elas.txt")

