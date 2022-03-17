function ObjFcn = BayesOpt(train_DS, val_DS)
 ObjFcn = @valErrorFun;

    function valError = valErrorFun(optVars)
        %% CNN Architecture
       layers = [imageInputLayer([28, 28]),...
    convolution2dLayer(optVars.FilterSize,optVars.NumFilter),reluLayer,maxPooling2dLayer(2,'Stride',2),...
    fullyConnectedLayer(10),softmaxLayer,classificationLayer];    
               
        options = trainingOptions('adam',...
            'ValidationData',val_DS,...
            'InitialLearnRate',optVars.InitialLearnRate,...,
            'Plots','training-progress',...
              'MiniBatchSize', optVars.MiniBatchSize);

        
        %% Train the network
        net = trainNetwork(train_DS, layers, options);
        %% Validation accuracy
        labels= classify(net, val_DS);
        accuracy_training  = sum(labels== val_DS.Labels )./numel(labels);
        %ObjFcn = 1 - accuracy_training; 
         valError = 1 - accuracy_training;
        
    end 
end 



