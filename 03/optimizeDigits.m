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
%%  Optimization options
ops = trainingOptions('sgdm','InitialLearnRate',0.0005,'MaxEpochs',10,'Plots','training-progress')
% ops = trainingOptions('sgdm','MaxEpochs',10, 'InitialLearnRate',0.0005,...
     %'Plots','training-progress')
%% train Networks
net = trainNetwork(train,layers,ops);
%%
c = classify(net,test);
l = test.Labels;
sum(c==l)/numel(l)




