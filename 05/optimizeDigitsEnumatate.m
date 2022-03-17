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
MaxEpochs = [5 10 15];  InitialLearnRate = linspace(0.0001,0.004,5);
tic
for i = 1:numel(MaxEpochs)
    for j = 1:numel(InitialLearnRate)
        ops = trainingOptions('sgdm',...
            'InitialLearnRate',InitialLearnRate(j),...
              'MaxEpochs',MaxEpochs(i),...
              'CheckpointPath','CheckPoint',...
              'ValidationData',test,...
              'ValidationFrequency',100);
        %% train Networks
        net = trainNetwork(train,layers,ops);
        %%
        c = classify(net,test);
        l = test.Labels;
        accur(i,j) = sum(c==l)/numel(l);
    end
end
toc




