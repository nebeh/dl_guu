%% Create object imageDatastore
path =fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
imds.countEachLabel
% split each labels, 800 for train and 200 for test
[train, test] = imds.splitEachLabel(800,'randomize');
%% Network
layers = [imageInputLayer([28, 28]), ...
    convolution2dLayer(5,20),reluLayer,    maxPooling2dLayer(2,'Stride',2),...
  %  convolution2dLayer(5,20),reluLayer,    maxPooling2dLayer(2,'Stride',2),...
    fullyConnectedLayer(10),    softmaxLayer,    classificationLayer];
%%
layers = [
    imageInputLayer([28 28])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Augmantation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
augimds = augmentedImageDatastore([28 28],train,'DataAugmentation',imageAugmenter);

%%  Optimization options

ops = trainingOptions('adam',...
    'InitialLearnRate',0.0005,...
      'MaxEpochs',30,...
      'ValidationData',test,...
      'ValidationFrequency',100,...
      'ValidationPatience',10,...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.2,...
      'LearnRateDropPeriod',5,...
    'Plots','training-progress');
%% train Networks
net = trainNetwork(augimds,layers,ops);
%%
c = classify(net,test);
l = test.Labels;
sum(c==l)/numel(l)




