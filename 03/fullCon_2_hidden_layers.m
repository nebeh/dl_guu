%% Load data
[x,t] = iris_dataset;

rng('default')
%% Linrear regression, one hidden layer
b = -[2; -3.4; 28.65];
W = randn(4,3); 
outLin = zeros(size(t));
for i = 1:(size(t,2))
   outLin(:,i) = W'*x(:,i) +b;
end

OUT = W'*x +b;

%% Logistic regression, one hidden layer
outSigm = zeros(size(t));
for i = 1:(size(t,2))
   outSigm(:,i) = 1./(1+exp(-W'*x(:,i) -b));
end
OUSIGM = 1./(1+exp(-W'*x -b));
%%
%% Linrear regression, two hidden layers
b1 = [-2; 3.4; -28.65; -10]; b2 = [28; -50; 39]; 
W1 = randn(4,4);  W2 = randn(4,3);
outLin1= zeros(size(x)); outLin2 = zeros(size(t));
for i = 1:(size(t,2))
   outLin1(:,i) = W1'*x(:,i) +b1; %  outLin2(:,i) = W2'*(W1'*x(:,i) +b1) +b2
    outLin2(:,i) = W2'*outLin1(:,i) +b2;
end

%% Logistic regression, two hidden layers
outSigm1 = zeros(size(x)); outSigm2 = zeros(size(t));
for i = 1:(size(t,2))
   outSigm1(:,i) = 1./(1+exp(-W1'*x(:,i) -b1)); %  outSigm2(:,i) = 1./(1+exp(-W2'*(1./(1+exp(-W1'*x(:,i) -b1));) -b2));
   outSigm2(:,i) = 1./(1+exp(-W2'*outSigm1(:,i) -b2));
end


% function [gradients, loss] = modelGradients(parameters, dlX, T)
% 
%     % Forward data through the model function.
%     dlY = model(parameters,dlX);
% 
%     % Compute loss.
%     loss = crossentropy(dlY,T);
% 
%     % Compute gradients.
%     gradients = dlgradient(loss,parameters);
% 
% end
