%%

clear all
clc

%% 
rng('default')

%% 
mu = [0 0 0 0 0 0 0 0];

%% 

i = 1:8;
matrix = abs(bsxfun(@minus,i',i));
covariance = repmat(.5,8,8).^matrix;
% covariance = 0.5*ones(8).^matrix;

%% 
X = mvnrnd(mu, covariance, 20);

%% Y = f(X)
Beta = [3; 1.5; 0; 0; 2; 0; 0; 0];

%% 
Y = X * Beta + 3 * randn(20,1);

% 
plotmatrix([Y X])

%% 
 
b = regress(Y,X);
[b Beta]

%%

[B Stats] = lasso(X,Y, 'CV', 5)

%% Create a Plot showing MSE versus lamba

lassoPlot(B, Stats, 'PlotType', 'CV')

%% 

%lassoPlot(B, Stats,'PlotType', 'Lambda','XScale','log')
lassoPlot(B, Stats)

%% 

[B(:,Stats.Index1SE) Beta b]

%%  
MSE = zeros(100,1);
mse = zeros(100,1);
Coeff_Num = zeros(100,1);
Betas = zeros(8,100);
%%

tic
parfor i = 1 : 100
    
    X = mvnrnd(mu, covariance, 20);
    Y = X * Beta + randn(20,1);
    
    [B Stats] = lasso(X,Y, 'CV', 5);
    Shrink = Stats.Index1SE -  ceil((Stats.Index1SE - Stats.IndexMinMSE)/2);
    Betas(:,i) = B(:,Shrink) > 0;
    Coeff_Num(i) = sum(B(:,Shrink) > 0);
    MSE(i) = Stats.MSE(:, Shrink);
    
    regf = @(XTRAIN, ytrain, XTEST)(XTEST*regress(ytrain,XTRAIN));
    cv_Reg_MSE(i) = crossval('mse',X,Y,'predfun',regf, 'kfold', 5);
        
end
toc

Number_Lasso_Coefficients = mean(Coeff_Num)
MSE_Ratio = median(cv_Reg_MSE)/median(MSE)
