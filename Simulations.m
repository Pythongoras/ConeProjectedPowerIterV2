addpath(genpath('/Users/yufei/Documents/2-CMU/PowerIter'))
% addpath(genpath('/home/yufeiy1/PowerIter/Code'));
clear; close all; clc

% Set the value of p and n
p = 1000;

% n = int32(10*log(p));
n = 10 * p;


% Set the number of iteration rounds for hyperparameter tuning
hypertune_round = 10;


% Set the eigenvec generating function.
% mnt_cone_eigenvec_nonsparse is used for the nonsparse setting.
% mnt_cone_eigenvec_sparse01 is used for the sparse setting.
% To use one setting, just comment out the other.

% Simulation1: nonsparse monotone
% eigen_gen_func = @mnt_cone_eigenvec_nonsparse;

% Simulation2: sparse monotone as [0,...,0,1,...,1]
eigen_gen_func = @mnt_cone_eigenvec_sparse01;


%% Hyperparameter tuning for elasticNet SPCA

% Set up the grid of the possible regularize hyperparameter lambda
elas_lambda_list = linspace(0.01,10,10);

% Initialize the container of loss for every lambda
elas_loss = zeros(1, length(elas_lambda_list));

for i = 1:hypertune_round
    disp(i)
    % Generate the first principle eigenvector with dimension p
    eigenvec = eigen_gen_func(p);
    % Generate covariance matrix
    C = cov(gaussian_data_mat(n,p,eigenvec));
    % Augment the loss in every round
    for j = 1:length(elas_lambda_list)
        vec = elas_spca_func(C, elas_lambda_list(j));
        elas_loss(j) = elas_loss(j) + min(sum((eigenvec + vec).^2),sum((eigenvec - vec).^2))^0.5;
    end
end
% disp(elas_loss)

% Pick out the combination of hyperparameters with minimum loss
[~,idx_lambda] = min(elas_loss);
elas_lambda_opt = elas_lambda_list(idx_lambda);
disp(elas_lambda_opt)


%% Hyperparameter tuning for truncated power iteration

% Set up the grid of possible cardinarlity hyperparameter
trunc_cardi_list = linspace(0.05,1,20);

% Initialize the container of loss for every cardi
trunc_loss = zeros(length(trunc_cardi_list),1);

for i = 1:hypertune_round
    disp(i)
    % Generate the first principle eigenvector with dimension p
    eigenvec = mnt_cone_eigenvec_sparse01(p);
    % Generate covariance matrix
    C = cov(gaussian_data_mat(n,p,eigenvec));
    % Augment the loss in every round
    for l = 1:length(trunc_cardi_list)
        vec = power_iter_func(C,@(x) proj_trunc(x,trunc_cardi_list(l)));
        trunc_loss(l) = trunc_loss(l) + min(sum((eigenvec + vec).^2),sum((eigenvec - vec).^2))^0.5;
    end
end
% disp(trunc_loss)

% Pick out the combination of hyperparameters with minimum loss
[~,idx_cardi] = min(trunc_loss);
trunc_cardi_opt = trunc_cardi_list(idx_cardi);
disp(trunc_cardi_opt)


%% Experiments
% clear; close all; clc

% Set the number of repeat time of experiments
exp_num = 20;

% container of the run time, l2 error and variance of 1st principal component
run_time = zeros(1,4);
l2_error = zeros(1,4);
var_pc = zeros(1,4);
var_true = 0;

% The hyperparameters
elas_lambda = elas_lambda_opt;
trunc_cardi = trunc_cardi_opt;

for i = 1:exp_num
    disp(i)
    
    % Generate the first principle eigenvector with dimension p
    eigenvec = eigen_gen_func(p);
    % Generate covariance matrix
    data_mat = gaussian_data_mat(n,p,eigenvec);
    C = cov(data_mat);
    
    % 1) power iteration: cone proj
    t = cputime;
    v_cone = power_iter_func(C, @proj_mnt);
    run_time(1) = run_time(1) + cputime - t;
    l2_error(1) = l2_error(1) + min(sum((eigenvec-v_cone).^2), sum((eigenvec+v_cone).^2))^0.5;
    var_pc(1) = var_pc(1) + var(data_mat * v_cone);
    % 2) power iteration: ordinary
    t = cputime;
    v_ordinary = power_iter_func(C, @proj_ordinary);
    run_time(2) = run_time(2) + cputime - t;
    l2_error(2) = l2_error(2) + min(sum((eigenvec-v_ordinary).^2), sum((eigenvec+v_ordinary).^2))^0.5;
    var_pc(2) = var_pc(2) + var(data_mat * v_ordinary);
    % 3) power iteration: truncated
    t = cputime;
    v_trunc = power_iter_func(C, @(x) proj_trunc(x, trunc_cardi));
    run_time(3) = run_time(3) + cputime - t;
    l2_error(3) = l2_error(3) + min(sum((eigenvec-v_trunc).^2), sum((eigenvec+v_trunc).^2))^0.5;
    var_pc(3) = var_pc(3) + var(data_mat * v_trunc);
    % 4) elastic spca 
    t = cputime;
    v_elas = elas_spca_func(C, elas_lambda);
    run_time(4) = run_time(4) + cputime - t;
    l2_error(4) = l2_error(4) + min(sum((eigenvec-v_elas).^2), sum((eigenvec+v_elas).^2))^0.5;
    var_pc(4) = var_pc(4) + var(data_mat * v_elas); 
    
    var_true = var_true + var(data_mat * eigenvec); 
end

% Take the average of run time and l2 error
run_time = run_time / exp_num;
l2_error = l2_error / exp_num;
var_pc = var_pc / exp_num;
var_true = var_true / exp_num;

disp([n, p])
disp(l2_error)
disp(run_time)
disp(var_pc)
disp(var_true)




