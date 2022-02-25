function  [xz,y,z] = peaksRMSProp(eta,gamma,X,Y)


if nargin == 0
X = 2; Y = -2;  eta =  0.05; gamma = 0.95;
elseif nargin == 1
    X = 2; Y = -2;   gamma = 0.95;
elseif nargin == 2
    X = 2; Y = -2;  
end
    dx = 1/8;
    [x,y] = meshgrid(-3:dx:3);


z =  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
   - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
   - 1/3*exp(-(x+1).^2 - y.^2);
Z = @(x,y)  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
   - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
   - 1/3*exp(-(x+1).^2 - y.^2);

 figure
    surf(x,y,z)

   figure
    contour(x,y,z,50)
    axis('tight')
    xlabel('x'), ylabel('y'), title('Peaks')
    grid on
    g = gca;
if nargout > 1
    xz = x;
elseif nargout == 1
    xz = z;
else
    % Self demonstration
    %disp(' ')
   % disp('z =  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ... ')
   % disp('   - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ... ')
   % disp('   - 1/3*exp(-(x+1).^2 - y.^2) ')
   % disp(' ')
   figure
    surf(x,y,z)

   figure
    contour(x,y,z,50)
    axis('tight')
    xlabel('x'), ylabel('y'), title('Peaks')
    grid on
    g = gca;
    
end

%% gradient
dEdx = @(x,y) (exp(- (x + 1).^2 - y.^2).*(2*x + 2))/3 + 3*exp(- (y + 1).^2 - x.^2).*(2*x - 2) + exp(- x.^2 - y.^2)*(30*x.^2 - 2) - 6*x.*exp(- (y + 1).^2 - x.^2).*(x - 1).^2 - 2*x.*exp(- x.^2 - y.^2).*(10*x.^3 - 2*x + 10*y.^5);
dEdy = @(x,y) (2*y.*exp(- (x + 1).^2 - y.^2))/3 + 50*y.^4.*exp(- x.^2 - y.^2) - 3*exp(- (y + 1).^2 - x.^2).*(2*y + 2).*(x - 1).^2 - 2*y.*exp(- x.^2 - y.^2).*(10*x.^3 - 2*x + 10*y.^5);
hold on
%%


%X0=X; Y0=Y;
%%
i = 1;  eps = 1; EX = 0; EY=0;
while eps>1e-6 && i < 30
   % dx =- eta*dEdx(X,Y); dy =- eta*dEdy(X,Y);
    g2X = dEdx(X,Y).^2; g2Y = dEdy(X,Y).^2;
    EX = gamma*EX+(1-gamma)*g2X; EY = gamma*EY+(1-gamma)*g2Y;
    X1 = X- eta*dEdx(X,Y)/sqrt(EX+1e-16);
    Y1 = Y- eta*dEdy(X,Y)/sqrt(EY+1e-16);
    eps = sqrt(abs(X-X1)+abs(Y-Y1));
    plot3(g,[X,X1],[Y,Y1],[Z(X,Y),Z(X1,Y1)],'r-*')
  
    %plot(x,y(x),'b')
    pause(0.5)
    drawnow
    % X0=X; Y0=Y;
    X = X1; Y = Y1;
    i = i+1;
end
eps