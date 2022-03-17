%%  Model
s1 = subplot(2,1,1);

x = linspace(-2,2,50);
y = 1./(1+25*x.^2);

plot(x,y,'LineWidth',3), hold on, grid on
s2 = subplot(2,1,2);
hold on, grid on

%% new model 
newModel = [2 5 20 40 45]; j = 1;
for i = newModel
    polyFit(x,y,i,s1,s2)
    str{j} = num2str(newModel(j)); j = j+1;
end

legend(s1,[{'model'},str])
legend(s2,str)

%%
function polyFit(x,y,n,s1,s2)
p = polyfit(x,y,n);
y1 = polyval(p,x);
err = y - y1;
plot(s1,x,y1)
plot(s2,x,err)

end