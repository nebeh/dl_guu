x = linspace(-5,5);
 y =@(x) sin(x) + x.^2/100;
 g = @(x) cos(x) + 2*x/100;


f = figure;
f.Position= [1 1 1920 1004];
 plot(x,y(x))
 grid on
 
%% 

hold on
eps = 1; X = 1;  X0 = 1; i = 1;
eta = 1; 
while eps>1e-6 && i < 20
    dx = -eta*g(X);

    X1 = X+dx +eta/3*(X-X0);
    X0 = X;
    
    eps = abs(dx);
    plot([X, X1],[y(X),y(X1)],'r-*')
    v = axis;
    x = linspace(v(1),v(2));
    plot(x,y(x),'b')
    drawnow
    X = X1;
    pause(0.5)
    i = i+1;
end