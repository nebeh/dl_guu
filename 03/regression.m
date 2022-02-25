x0 = dlarray([-1,2]);
% y = a*x + b;

x = linspace(0,5);
t = 2*x + 1 + randn(size(x))/2;
plot(x,t)
hold on
%% 
alpha = 0.0001*6;
for i = 1:100
    x01 = 0; x02 = 0;
    [fval,gradval] = dlfeval(@regress_fun,x,x0,t);
    x0(1) = x0(1) - alpha*gradval(1) - 0.9*x01;
    x0(2) = x0(2) - alpha*4*gradval(2)-0.9*x02;
    pause(0.1)
    xx = extractdata(x0);
    x01 = x0(1); x02 = x0(2);
    disp(extractdata(fval))
plot(x,xx(1)*x + xx(2))
drawnow
end
figure
plot(x,t,x,xx(1)*x + xx(2))

%% 
% [fval,gradval] = dlfeval(@regress_fun,x,x0,t);



function [loss,grad] = regress_fun(x,x0,t)
n = numel(t);
loss = sum((x*x0(1)+x0(2)-t).^2);
grad = dlgradient(loss,x0);

end