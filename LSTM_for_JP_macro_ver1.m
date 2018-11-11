%% Import data and set parameters
% This script is based on the URL below.

%clear;
%close all;
cd("/Users/yano/Dropbox/program/matlab/lstm/src")


%================================================
% Initial Setting
%================================================
% Import data

% Choose a data-set flag
flagForProgramTest = "macroeconometrics";
%flagForProgramTest = "chickenpox"; % See the URL below about "chickenpox"
% https://jp.mathworks.com/help/deeplearning/examples/time-series-forecasting-using-deep-learning.html

txt = [];
switch flagForProgramTest
    
    case "macroeconometrics"
        %
        [dataAll,txt,raw] = xlsread("/Users/yano/Dropbox/program/matlab/lstm/src/jp_monthly_data_5vars_with_mbase.xlsx");
        %[dataAll,txt,raw] = xlsread("/Users/yano/Dropbox/program/matlab/lstm/src/jp_monthly_data_5vars_with_money.xlsx");
        % Money stock and commo
        %[dataAll,txt,raw] = xlsread("/Users/yano/Dropbox/program/matlab/lstm/src/jp_monthly_data_4vars_with_money.xlsx");
        % Monetary base and commo
        %[dataAll,txt,raw] = xlsread("/Users/yano/Dropbox/program/matlab/lstm/src/jp_monthly_data_4vars_with_mbase.xlsx");
        % Money and ex
        %[dataAll,txt,raw] = xlsread("/Users/yano/Dropbox/program/matlab/lstm/src/jp_monthly_data_4vars_ex_money.xlsx");
        
        dataAll = dataAll.'; % Matlab assumes that an observation vector is a row vector
        disp(txt);
        
    case "chickenpox"
        dataAll = chickenpox_dataset;
        dataAll = [dataAll{:}];
        
end

[numOfInput, ~] =size(dataAll);
% Set parameters
numOfOutput = numOfInput;
numFeatures = numOfInput;
numResponses = numOfOutput;
numHiddenUnits = 300;
dropoutProb = 0.5;
dataLag = 1;

[~, sampleSize] = size(dataAll);
availableSampleSize = sampleSize - (dataLag - 1);
numOfInput = numOfInput * dataLag;


numTimeStepsTrain = floor(0.90*numel(dataAll(1,:)))
dataTrain = dataAll(:,1:numTimeStepsTrain+1);
dataTest = dataAll(:,numTimeStepsTrain+1:end);
dataAll(:,1:5);
dataTrain(:, 1:5);
dataTest(:, 1:5);

layersSetting = "withoutDropout";
%layersSetting = "withDropout";
%layersSetting = "simpleRegression";


%% Standardize train data and plot them
%figure;
standDataTrain = dataTrain;
for ii = 1:numOfInput
    mu = mean(dataTrain(ii,:));
    sig = std(dataTrain(ii,:));
    
    standDataTrain(ii,:) = (dataTrain(ii,:) - mu) / sig;
    %    subplot(numOfInput,1,ii);
    %    plot(standDataTrain(ii,:));
end
XTrain = standDataTrain(:, 1:end-dataLag);
YTrain = standDataTrain(:, (dataLag+1):end);
XTrain(:, 1:5)
YTrain(:, 1:5)

%% Construct neural net layers

switch layersSetting
    
    case "withoutDropout"
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
    case "withDropout"
        layers = [ ...
            sequenceInputLayer(numFeatures)
            dropoutLayer(dropoutProb)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
    case "simpleRegression"
        layers = [ ...
            sequenceInputLayer(numFeatures)
            fullyConnectedLayer(numHiddenUnits)
            %            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
end

%% Train LSTM network

optionFlag = "changeDefaults";
%optionFlag = "useDefault";

% Set options
switch optionFlag
    case "changeDefaults"
        % Chnage default options
        options = trainingOptions('adam', ...
            'MaxEpochs',200, ...
            'MiniBatchSize', 64, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',50, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',1, ...
            'Plots','training-progress');
        
    case "useDefault"
        % Use detault options
        options = trainingOptions('adam', ...
            'MaxEpochs',200, ...
            'Plots','training-progress');
        
end

% Train network
net = trainNetwork(XTrain,YTrain,layers,options);

%% Standardize test data and plot them
%figure;
muTest = zeros(numOfInput,1);
sigTest = zeros(numOfInput,1);
standDataTest = dataTest;

for ii = 1:numOfInput
    muTest(ii,1) = mean(dataTest(ii,:));
    sigTest(ii,1) = std(dataTest(ii,:));
    
    standDataTest(ii,:) = (dataTest(ii,:) - muTest(ii,1)) / sigTest(ii,1);
    %    subplot(numOfInput,1,ii);
    %    plot(dataTest(ii,:));
end
XTest = standDataTest(:, 1:end-dataLag);
YTest = standDataTest(:, (dataLag+1):end);


%% One-step-ahed prediction
% Predict nad update states
% Update Network State with Observed Values

net = resetState(net);
net = predictAndUpdateState(net,XTrain);

numTimeStepsTest = numel(XTest(1,:));
YPred = zeros(numOfOutput, numTimeStepsTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

for ii = 1:numOfInput
    YPred(ii,:) = sigTest(ii,1)*YPred(ii,:) + muTest(ii,1);
    %    YTest(ii,:) = sigTest(ii,1)*YTest(ii,:) + muTest(ii,1);
end


figure;
YTestOriginal = dataTest(:, (dataLag+1):end);
for ii = 1:numOfInput
    rmse = sqrt(mean((YPred(ii,:)-YTestOriginal(ii,:)).^2));
    subplot(numOfInput,1,ii);
    plot(YTestOriginal(ii,:))
    hold on
    plot(YPred(ii,:),'.-')
    hold off
    legend(["Observed" "Forecast"], 'Location', "southeast")
    
    if ~isempty(txt)
        ylabel(txt(ii))
        title("One-step-ahead forecast")
        disp("RMSE (" + txt(ii) + "): " + rmse)
    end
    
end

%% Multi-step-ahed forecast
% Predict nad update states
% Multi-step-ahed forecast is suppressed now. Delete %{ and %} to activate
% it.
%{
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(:,end));

numTimeStepsTest = numel(XTest(1,:));
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

for ii = 1:numOfInput
    YPred(ii,:) = sigTest(ii,1)*YPred(ii,:) + muTest(ii,1);
    %    YTest(ii,:) = sigTest(ii,1)*YTest(ii,:) + muTest(ii,1);
end


figure;
YTestOriginal = dataTest(:, (dataLag+1):end);
for ii = 1:numOfInput
    
    subplot(numOfInput,1,ii);
    plot(YTestOriginal(ii,:))
    hold on
    plot(YPred(ii,:),'.-')
    hold off
    legend(["Observed" "Forecast"])
    ylabel("Cases")
    if ~isempty(txt)
        rmse = sqrt(mean((YPred(ii,:)-YTestOriginal(ii,:)).^2));
        ylabel(txt(ii))
        title("Multi-step-ahead forecast")
        disp("RMSE (" + txt(ii) + "): " + rmse)
    end
    
end
%}
