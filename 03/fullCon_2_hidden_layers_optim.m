%% Load data
[x,t] = iris_dataset;
rng('default')
%% Linrear regression, one hidden layer
params.b1 = dlarray(randn(4,1));
params.W1 = dlarray(randn(4,4)); 
params.b2 = dlarray(randn(3,1));
params.W2 = dlarray(randn(3,4)); 

%% 
% x = dlarray(rand(4,1));
% y = model(params,x);
trailingAvg = []; trailingAverageSq = [];
vel=[];
%% 

epoch = 100;
for i = 1:epoch
    for j = 1:size(x,2)
        X = x(:,j); Y = t(:,j);
     [grad,loss] = dlfeval(@modelGradients,params,X,Y);
      [params,vel] = sgdmupdate(params,grad,vel,0.003);
    end
      disp(loss)
end
%% 

y = model(params,x);
function y = model(params,x)
y = params.W1*x + params.b1;
y = 1./(1+exp(-y));
%y = relu(y);
y = params.W2*y + params.b2;
y = dlarray(y,'SC');
y = exp(y)./sum(exp(y));
end


function [gradients, loss] = modelGradients(parameters, dlX, T)

    % Forward data through the model function.
    dlY = model(parameters,dlX);

    % Compute loss.
    T = dlarray(T,'SC');
    loss = crossentropy(dlY,T);

    % Compute gradients.
    gradients = dlgradient(loss,parameters);

end
 

