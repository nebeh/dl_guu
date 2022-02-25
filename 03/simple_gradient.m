X0 = 2;
x0 = dlarray(X0);
t = linspace(-1,4);
plot(t,sin(t)-2*sin(2*t),'b')
hold on
plot(X0,sin(X0)-2*sin(2*X0),'r*')
for i = 1:10
    [fval,gradval] = dlfeval(@simple_fun,x0);
    x = x0 - 0.1*gradval;
    x0 = x;
    x = extractdata(x);
    plot(x,sin(x)-2*sin(2*x),'r*')
end



function [f,grad] = simple_fun(x)

f = sin(x)-2*sin(2*x);
grad = dlgradient(f,x);

end