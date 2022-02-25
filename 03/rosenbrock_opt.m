x0 = dlarray([-1,2]);
[fval,gradval] = dlfeval(@rosenbrock,x0)

function [f,grad] = rosenbrock(x)

f = 100*(x(2) - x(1).^2).^2 + (1 - x(1)).^2;
grad = dlgradient(f,x);

end