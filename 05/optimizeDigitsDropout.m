%% Create object imageDatastore
path =fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');

% split each labels, 800 for train and 200 for test
[train, test] = imds.splitEachLabel(800,'randomize');
%% Network
layers = [imageInputLayer([28, 28]), ...
    convolution2dLayer(5,20),...
    reluLayer,    maxPooling2dLayer(2,'Stride',2),...
    fullyConnectedLayer(10),    softmaxLayer,    classificationLayer];

%%  Optimization options

ops = trainingOptions('sgdm',...
    'InitialLearnRate',0.001,...
    'MaxEpochs',15,...
    'Plots','training-progress');
% train Networks
net = trainNetwork(train,layers,ops);
%
c = classify(net,test);
l = test.Labels;
sum(c==l)/numel(l)


%% Network whith dropoutLayer
layers = [imageInputLayer([28, 28]), ...
    convolution2dLayer(5,20),...
    reluLayer,    maxPooling2dLayer(2,'Stride',2),...
    dropoutLayer(0.05),...
    fullyConnectedLayer(10),    softmaxLayer,    classificationLayer];

net = trainNetwork(train,layers,ops);
%
c = classify(net,test);
l = test.Labels;
sum(c==l)/numel(l)




