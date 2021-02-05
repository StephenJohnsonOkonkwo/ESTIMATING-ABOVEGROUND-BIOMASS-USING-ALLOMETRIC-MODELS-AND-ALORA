cd 'C:/Users/Steve/Desktop/Adaptive Filtering Project'
load ("ZachDatalnAGBlnD2H.txt")
size (ZachDatalnAGBlnD2H)
corr (ZachDatalnAGBlnD2H)

%%% Splitting The Data into Two
P = 0.80 ;
n = 482
idx = randperm(n)  ;
Train_Data =  ZachDatalnAGBlnD2H(idx(1:round(P*n)),:)
Test_Data =  ZachDatalnAGBlnD2H(idx(round(P*n)+1:end),:) 

%%% Training Data
AGB_train = Train_Data (:, 1);
D2H_train = Train_Data (:, 2);

%%% Test Data
AGB_test = Test_Data (:, 1);
D2H_test = Test_Data (:, 2);

%%% Correlation Analysis of the Training Data
corr (AGB_train, D2H_train)
m_train = length (AGB_train)
l_train = ones (m_train, 1);

%%% Correlation Analysis of the Test Data
corr (AGB_test, D2H_test)
m_test = length (AGB_test)
l_test = ones (m_test, 1);

%%% X Variable . . .
X_train = [l_train, D2H_train];
X_test = [l_test, D2H_test];


%%% STOCHASTIC GRADIENT DESCENT %%%%%%%%
%% Initializing variables . . .
alpha = 0.001;
iterations = 1000;
theta0_vals = zeros (1, iterations);
theta1_vals = zeros (1, iterations);
theta = [0 ; 0];
J_vals = zeros (iterations);
J_history = zeros (iterations, 1);
theta_store = zeros (2, iterations);


%% Running Iterations . . . 
for i = 1:iterations
h = X_train * theta;
errors = h - AGB_train;
theta_change = (alpha * (X_train' * errors)) / m;
theta = theta - theta_change;
theta0_vals (i) = theta (1, :);
theta1_vals (i) = theta (2, :);
theta_store (:, i) = theta;
J_history(i) = (sum ((h - AGB_train).^2))/ (2 * m);
J_vals (i, i) = (sum ((h - AGB_train).^2))/ (2 * m);
end
%% End of Iterations.

SGD_Loss = J_history;
SGD_theta0 = theta0_vals;
SGD_theta1 = theta1_vals;
SGD_thetaStore = theta_store;
SGD_theta = theta;

min (SGD_Loss)
max (SGD_Loss)
SGD_theta

AGBSGD_cap = X_test * SGD_theta;

epoch = [1:iterations];
plot (epoch, SGD_Loss, '-')

%%% RUN STUDENT'S T-TEST N DATA %%%%%
pkg load statistics
[h, pval, ci, stats] = ttest2 (AGB_test', AGBSGD_cap')

%%%% Root Mean Square Error Test
RMSE_SGD = sqrt (sum((AGBSGD_cap - AGB_test).^2)/m_test)
