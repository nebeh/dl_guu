x = linspace(-5,5);
 y =@(x) sin(x) + x.^2/100;
% y = @(x) (x-1.5).^2 +10* cos(x);
% gradient
 g = @(x) cos(x) + 2*x/100;
% g = @(x) 2*x-3-10*sin(x);

f = figure;
f.Position= [1 1 1920 1004];
 plot(x,y(x))
 grid on
 
%% gradient


hold on
eps = 1e-2; X = 1;
eta =  1e-1;
i = 1;
while eps>1e-6 && i < 20
    dx = -eta*g(X);
    X1 = X+dx;
    eps = abs(dx);
    plot([X,X1],[y(X),y(X1)],'r-*')
    v = axis;
    x = linspace(v(1),v(2));
    plot(x,y(x),'b')
    pause(0.5)
    drawnow
    X = X1;
    i = i+1;
end