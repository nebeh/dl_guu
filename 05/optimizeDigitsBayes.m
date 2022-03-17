%% Create object imageDatastore
path =fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
imds.countEachLabel
% split each labels, 800 for train and 200 for test
[train, test] = imds.splitEachLabel(800,'randomize');
%% Network
layers = [imageInputLayer([28, 28]), ...
    convolution2dLayer(5,20),reluLayer,    maxPooling2dLayer(2,'Stride',2),...
    fullyConnectedLayer(10),    softmaxLayer,    classificationLayer];

%%

toc

    optVars = [
        optimizableVariable('FilterSize',[3 5],'Type','integer')
        optimizableVariable('NumFilter',[15, 30],'Type','integer')
         % optimizableVariable('Momentum',[0.8 0.95])
         optimizableVariable('InitialLearnRate',[1e-5, 30e-3],'Transform','log')
        % optimizableVariable('LearnRateDropFactor',[40e-2, 99e-2],'Transform','log')
        % optimizableVariable('LearnRateDropPeriod',[5, 45],'Type','integer')
        % optimizableVariable('MaxEpochs',[50, 500],'Type','integer')
         optimizableVariable('MiniBatchSize',[ceil(0.005*size(train.Labels,1)),ceil(0.05*size(train.Labels,1))],'Type','integer')
        % optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')
        ];
   ObjFcn = BayesOpt(train,test);
 tic; % Time Start
 BayesObject = bayesopt(ObjFcn,optVars,...
       'MaxObj',15,...
       'MaxTime',60*60,...
       'IsObjectiveDeterministic',false,...
       'UseParallel',false); % set UseParallel as true for faster results.
toc
%%
% layers = [imageInputLayer([28, 28]), ...
%     convolution2dLayer(5,18),reluLayer,    maxPooling2dLayer(2,'Stride',2),...
%     fullyConnectedLayer(10),    softmaxLayer,    classificationLayer];
% ops = trainingOptions('adam','InitialLearnRate',0.000367425421994577,'MiniBatchSize',399);

net = trainNetwork(test,layers,ops);
%%
l = test.Labels; c = classify(net,test);
sum(l == c)/numel(l)



