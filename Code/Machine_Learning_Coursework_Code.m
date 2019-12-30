%% Machine Learning Coursework 
% Dataset: UCI Wisconsin Breast Cancer (Original)
% Choosen models: Naive Bayes | Random Forest

%% Set random seed for reproducibility of experiment results

rng(2)

%% Import data from data file

% Read data file to text file
fileName = 'breast-cancer-wisconsin.data';
formatSpec = '%n%n%n%n%n%n%n%n%n%n%n%[^\n\r]';
fileID = fopen(fileName, 'r');

% Note: 'TreatAsEmpty' replaces '?' values with 'NaN'
dataArray = textscan(fileID, formatSpec, 'Delimiter', ',', 'TextType', 'string', ...
    'ReturnOnError', false, 'TreatAsEmpty', '?');

fclose(fileID);

% Table for predictors (features) and response (class) variables 
breastCancerDataTable = table(dataArray{2:end-1}, 'VariableNames', ...
    {'ClumpThickness', 'UCellSize', 'UCellShape', 'MarginalAdhesion', ...
    'SECellSize', 'BareNuclei', 'BlandChromatin', 'NormalNuclei', 'Mitoses', ...
    'Class'});

clearvars fileName formatSpec fileID dataArray ans;

%% Delete rows containing 'NaN' (missing values)

% Total amount of 'NaN' values in 'BreastCancerDataTable'
missingValues = sum(ismissing(breastCancerDataTable));

% Note: 'rmmissing' removes rows containing 'NaN'
breastCancerDataTable = rmmissing(breastCancerDataTable);

clearvars ans

%% Basic statistical overview of 'BreastCancerDataTable'

% Matrix containing values of the 9 predictors (features)
predictorMatrix = table2array(breastCancerDataTable(:,1:9));

predictors = {'ClumpThickness', 'UCellSize', 'UCellShape', 'MarginalAdhesion', ...
    'SECellSize', 'BareNuclei', 'BlandChromatin', 'NormalNuclei', 'Mitoses'};

% Calculate Spearman's Rank Correlation between all predictors
% Plot results on a heatmap
rho = corr(predictorMatrix, 'Type', 'Spearman');
figure;
SRCCMatrix = heatmap(rho, 'colormap', colormap('gray'));
boxProp = gca;
xlabel('Predictor');
boxProp.XData = predictors;
ylabel('Predictor');
boxProp.YData = predictors;
title('Spearman''s Rank Correlation Coefficient Matrix');

% Plot boxplots of predictors (values)
figure;
boxplot(table2array(breastCancerDataTable(:,1:9)), 'Symbol', 'ro', ...
    'Colors', 'k', 'Widths',0.4);
% Set face color of boxes
 obj = findobj(gca, 'Tag', 'Box');
 for j = 1:length(obj)
    patch(get(obj(j), 'XData'),get(obj(j), 'YData'), 'r', 'FaceAlpha', 0.5);
 end
xlabel('Predictor')
xticks([1 2 3 4 5 6 7 8 9]);
xticklabels(predictors);
xtickangle(45);
ylabel('Value');
title('Boxplots');

% Calculate frequency of instances that correspond to benign and malignant
BMFrequency = zeros(1, 3);
for i = 1:3
    if i < 3
        idx = breastCancerDataTable.(10) == (i + i);
        BMFrequency(i) = BMFrequency(i) + height(breastCancerDataTable(idx ,:));
    else
        BMFrequency(i) = BMFrequency(i) + sum(BMFrequency(:,1:2));
    end
end
% Store results in a table
BMFrequencyTable = array2table(BMFrequency, 'RowNames', {'Frequency'}, ...
    'VariableNames', {'Benign', 'Malignant', 'Total'});
% Display frequency table
display(BMFrequencyTable);

clearvars predictorMatrix predictors rho SRCCMatrix boxProp obj j i idx ...
    BMFrequency ans

%% Training set and test set allocation

X = breastCancerDataTable(:,1:9);
Y = breastCancerDataTable(:,10);
pt = cvpartition(breastCancerDataTable.Class,'HoldOut',0.25);
trainingAttributes = X(training(pt),:);
testAttributes = X(test(pt),:);
trainingClass = Y(training(pt),:);
testClass = Y(test(pt),:);


%% ------------------------------------------------------------------------
% Naive Bayes experiments
% -------------------------------------------------------------------------

%% Experiment 0 (Elgedawy experiment)

% Feature selection for experiment (all predictors)
trainingAttributesE0 = trainingAttributes;
testAttributesE0 = testAttributes;

% Train Naive Bayes model with training subset
MdlNBE0 = fitcnb(trainingAttributesE0,trainingClass,'DistributionNames', 'kernel');

% Predict classes with training subset
predictedTrainingClassE0 = predict(MdlNBE0, trainingAttributesE0);
% Predict classes with test subset
predictedClassE0 = predict(MdlNBE0, testAttributesE0);

% Call custom function for training set to retrieve:
% confusion matrix | recall | precision | accuracy | f-measure
CMTrainE0 = CMMeasures(trainingClass, predictedTrainingClassE0);

% Call custom function for test set to retrieve:
% confusion matrix | recall | precision | accuracy | f-measure
CMTestE0 = CMMeasures(testClass, predictedClassE0);

%% Experiment 1 (8 predictors & normal | kernel) 

% Feature selection for experiment (8 predictors)
% predictor 3 and predictor 2 are highly correlated, so drop predictor 2
% Note: predictor 2 = 'Uniformity Cell Size'
% Note: predictor 3 = 'Uniformity Cell Shape'
Xfs = removevars(X, {'UCellSize'});
trainingAttributesE1 = Xfs(training(pt),:);
testAttributesE1 = Xfs(test(pt),:);
trainingClassE1 = Y(training(pt),:);
testClassE1 = Y(test(pt),:);

% Train models with 10 fold cross validation
% Distributions: normal | kernel
MdlNBE1 = fitcnb(trainingAttributesE1,trainingClassE1,'KFold',10, ...
    'DistributionNames', 'normal');
MdlNBE1Kernel = fitcnb(trainingAttributesE1,trainingClassE1,'KFold',10, ...
    'DistributionNames', 'kernel');

% Calculate training class predictions for each model (normal | kernel)
predictedTrainClassE1 = zeros(513,10);
predictedTrainClassE1Kernel = zeros(513,10);
% 10 fold cross validation on training set (normal | kernel)
for i = 1:10
    predictedTrainClassE1(:,i) = predict(MdlNBE1.Trained{i},trainingAttributesE1);
    predictedTrainClassE1Kernel(:,i) = predict(MdlNBE1Kernel.Trained{i},trainingAttributesE1);
end

% Call custom function for training set to retrieve (normal | kernel):
% confusion matrix | recall | precision | accuracy | f-measure
CMTrainNormalE1 = CMMeasures(trainingClassE1, predictedTrainClassE1);
CMTrainKernelE1 = CMMeasures(trainingClassE1, predictedTrainClassE1Kernel);

% Calculate test class predictions for each model (normal | kernel)
predictedClassE1 = zeros(170,10);
predictedClassE1Kernel = zeros(170,10);
% 10 fold cross validation on training set (normal | kernel)
for i = 1:10
    predictedClassE1(:,i) = predict(MdlNBE1.Trained{i},testAttributesE1);
    predictedClassE1Kernel(:,i) = predict(MdlNBE1Kernel.Trained{i},testAttributesE1);
end

% Call custom function for test set to retrieve (normal | kernel):
% confusion matrix | recall | precision | accuracy | f-measure
CMTestNormalE1 = CMMeasures(testClassE1, predictedClassE1);
CMTestKernelE1 = CMMeasures(testClassE1, predictedClassE1Kernel);

% plot confusion matrix chart for normal test set
figure;
confusionchart(CMTestNormalE1, unique(predictedClassE1), ...
   'DiagonalColor', 'k', 'OffDiagonalColor', 'w');
title('Naive Bayes Confusion Matrix (Experiment 1 - Normal Distribution - Test Set)');

%% Experiment 2 (altering prior probabilities)
% NOTE: experiment uses previous predictors from previous experiment
% i.e. dropped predictor 2 ('Uniformity Cell Size')
%Look at changing prior probabilities based on real world knowledge

% Retrieve exisiting prior probabilities
defaultPrior = zeros(10,2);
for i = 1:10
    defaultPrior(i,:) = MdlNBE1Kernel.Trained{i}.Prior;
end

% Set prior to 80%/20%
prior = [0.8 0.2];
MdlNBE2 = fitcnb(trainingAttributesE1,trainingClassE1,'KFold', 10, ...
    'DistributionNames', 'kernel', 'Prior', prior);

% Calculate training class predictions for each model
predictedTrainClassE2 = zeros(513,10);
for i = 1:10
    predictedTrainClassE2(:,i) = predict(MdlNBE2.Trained{i},trainingAttributesE1);  
end

% Call custom function for training set to retrieve (altered prior):
% confusion matrix | recall | precision | accuracy | f-measure
CMTrainE2 = CMMeasures(trainingClass, predictedTrainClassE2);

% Calculate test class predictions for the model
predictedClassE2 = zeros(170,10);
for i = 1:10
    predictedClassE2(:,i) = predict(MdlNBE2.Trained{i},testAttributesE1);
end

% Call custom function for test set to retrieve (altered prior):
% confusion matrix | recall | precision | accuracy | f-measure
CMTestE2 = CMMeasures(testClass, predictedClassE2);

clearvars DefaultPrior

%% Organise performance measurements for each experiment into table

% Retrieve each measure for each experiment
[~, ReCallTRE0, PTRE0, AccTRE0, FmTRE0] = CMMeasures(trainingClass, predictedTrainingClassE0);
[~, ReCallTEE0, PTEE0, AccTEE0, FmTEE0] = CMMeasures(testClass, predictedClassE0);
[~, ReCallTRNE1, PTRNE1, AccTRNE1, FmTRNE1] = CMMeasures(trainingClassE1, predictedTrainClassE1);
[~, ReCallTENE1, PTENE1, AccTENE1, FmTENE1] = CMMeasures(testClassE1, predictedClassE1);
[~, ReCallTRKE1, PTRKE1, AccTRKE1, FmTRKE1] = CMMeasures(trainingClassE1, predictedTrainClassE1Kernel);
[~, ReCallTEKE1, PTEKE1, AccTEKE1, FmTEKE1] = CMMeasures(testClassE1, predictedClassE1Kernel);
[~, ReCallTRE2, PTRE2, AccTRE2, FmTRE2] = CMMeasures(trainingClass, predictedTrainClassE2);
[~, ReCallTEE2, PTEE2, AccTEE2, FmTEE2] = CMMeasures(testClass, predictedClassE2);

% Store measures into different column vectors
AccuracyArray = [AccTRE0; AccTEE0; AccTRNE1; AccTENE1; AccTRKE1; AccTEKE1; AccTRE2; ...
    AccTEE2];
FmeasureArray = [FmTRE0; FmTEE0; FmTRNE1; FmTENE1; FmTRKE1; FmTEKE1; FmTRE2; ...
    FmTEE2];
RecallArray = [ReCallTRE0; ReCallTEE0; ReCallTRNE1; ReCallTENE1; ReCallTRKE1; ...
    ReCallTEKE1; ReCallTRE2; ReCallTEE2];
PrecisionArray = [PTRE0; PTEE0; PTRNE1; PTENE1; PTRKE1; PTEKE1; PTRE2; PTEE2];

Set = {'TrainE0'; 'TestE0'; 'TrainNorE1'; 'TestNormE1'; 'TrainKerE1'; ...
    'TestKerE1'; 'TrainE2'; 'TestE2'};

% Create table with complete performance measures for every NB experiment
measureTableNB = table(AccuracyArray, FmeasureArray, RecallArray, ...
    PrecisionArray, 'RowNames', Set, 'VariableNames', ...
    {'Accuracy', 'Fmeasure', 'Recall', 'Precision'});

clearvars AccTRE0 AccTEE0 AccTRNE1 AccTENE1 AccTRKE1 AccTEKE1 AccTRE2 ...
    AccTEE2 FmTRE0 FmTEE0 FmTRNE1 FmTENE1 FmTRKE1 FmTEKE1 FmTRE2 ...
    FmTEE2 ReCallTRE0 ReCallTEE0 ReCallTRNE1 ReCallTENE1 ReCallTRKE1 ...
    ReCallTEKE1 ReCallTRE2 ReCallTEE2 PTRE0 PTEE0 PTRNE1 PTENE1 ...
    PTRKE1 PTEKE1 PTRE2 PTEE2 Set AccuracyArray FmeasureArray RecallArray ...
    PrecisionArray ans


%% ------------------------------------------------------------------------
% Random Forest experiments
% -------------------------------------------------------------------------

%% Experiment 0 (Elgedawy experiment)

% Train treebagger model (Random Forest) with 2000 trees
MdlRFE0 = TreeBagger(2000, trainingAttributes, trainingClass, ...
    'OOBVarImp', 'Off');

% Calculate training class predicitions for model
predictedTrainClassRFE0 = str2double(predict(MdlRFE0,trainingAttributes));
% Calculate training class predicitions for model
predictedTestClassRFE0 = str2double(predict(MdlRFE0,testAttributes));

%% Experiment 1 (grid search)
% Feature selection for experiment (8 predictors)
% predictor 3 and predictor 2 are highly correlated, so drop predictor 2
% Note: predictor 2 = 'Uniformity Cell Size'
% Note: predictor 3 = 'Uniformity Cell Shape'
% Use trainingAttributesE1 | testAttributesE1 | trainingClassE1 | testClassE1
% Perform grid search

% Parameters
numofTrees = [10 50 100 500];
numofFeatures = [1 2 3 4 5 6 7 8];

numofTreesLen = length(numofTrees);
numofFeaturesLen = length(numofFeatures);

gridResultsTrain = zeros(numofTreesLen, numofFeaturesLen);
gridResultsTest = zeros(numofTreesLen, numofFeaturesLen);
gridRunTimeTest = zeros(numofTreesLen, numofFeaturesLen);

% Grid search loop (running over different no of trees & predictors)
for i = 1:numofTreesLen
    for j = 1:numofFeaturesLen
        rng(2);
        tic
        MdlRFG = TreeBagger(numofTrees(i), trainingAttributesE1, trainingClass, ...
            'NumPredictorsToSample', numofFeatures(j), 'OOBVarImp', 'Off');
        gridRunTimeTest(i, j) = toc;
        predictedTrainClassG = str2double(predict(MdlRFG, trainingAttributesE1));
        predictedTestClassG = str2double(predict(MdlRFG, testAttributesE1));
        [~, ~, ~, RFtrainGAccuracy] = CMMeasures(trainingClass, predictedTrainClassG);
        [~, ~, ~, RFtestGAccuracy] = CMMeasures(testClass, predictedTestClassG);
        trainError = (1-RFtrainGAccuracy)*100;
        testError = (1-RFtestGAccuracy)*100;
        gridResultsTrain(i, j) = trainError;
        gridResultsTest(i, j) = testError;
    end
end

% Plot grid search results for accuracy %
figure;hold on
surf(gridResultsTest, 'FaceColor', 'r')
surf(gridResultsTrain, 'FaceColor', 'k')
alpha 0.5
title('Random Forest Hyperparameter Tuning')
xlabel('Number of Predictors')
ylabel('Number of Trees')
yticks([1 2 3 4])
yticklabels({'10', '50', '100', '500'})
zlabel('Classification Error (%)')
legend('Training Set','Test Set')
view(17,22)

% Plot grid search results for run time 
figure;hold on
surf(gridRunTimeTest, 'FaceColor', 'r', 'LineStyle', '-.')
alpha 0.5
title('Random Forest Run Times')
xlabel('Number of Predictors')
ylabel('Number of Trees')
yticks([1 2 3 4])
yticklabels({'10', '50', '100', '500'})
zlabel('Run Time')
legend('Model Run Time')
view(17,22)
hold off

%% Random Forest model with optimal parameters (from grid search)

MdlRFE1 = TreeBagger(500, trainingAttributesE1, trainingClass, ...
    'NumPredictorsToSample', 1, 'OOBVarImp', 'On', 'OOBPred', 'on');

% Calculate training class predicitions for model
predictedTrainClassRFE1 = str2double(predict(MdlRFE1,trainingAttributes));
% Calculate test class predicitions for model
predictedTestClassRFE1 = str2double(predict(MdlRFE1,testAttributes));

% Call custom function for optimal test set to retrieve:
% confusion matrix | recall | precision | accuracy | f-measure
CMTestOptimal = CMMeasures(testClass,predictedTestClassRFE1);

% plot confusion matrix chart for optimal test set
figure;
confusionchart(CMTestOptimal, unique(predictedTestClassRFE1), ...
   'DiagonalColor', 'k', 'OffDiagonalColor', 'w');
title('Random Forest Confusion Matrix (Experiment 1 - Optimal - Test Set)');

%% Calculate predictor importance

% Calculate feature importance (Random Forest)
figure
bar(MdlRFE1.OOBPermutedVarDeltaError, 'r')
alpha 0.5
xlabel('Predictor')
xticks([1 2 3 4 5 6 7 8 9]);
xticklabels({'ClumpThickness', 'UCellShape', 'MarginalAdhesion', ...
    'SECellSize', 'BareNuclei', 'BlandChromatin', 'NormalNuclei', 'Mitoses'});
xtickangle(45);
ylabel('Out-of-Bag Predictor Importance')
title('Out-of-Bag Permuted Predictor Importance Estimates');

%% Organise performance measurements for each experiment into table

% Retrieve each measure for each experiment
[~, ReCallTRE0, PTRE0, AccTRE0, FmTRE0] = CMMeasures(trainingClass, predictedTrainClassRFE0);
[~, ReCallTEE0, PTEE0, AccTEE0, FmTEE0] = CMMeasures(testClass, predictedTestClassRFE0);
[~, ReCallTRNE1, PTRNE1, AccTRNE1, FmTRNE1] = CMMeasures(trainingClassE1, predictedTrainClassRFE1);
[~, ReCallTENE1, PTENE1, AccTENE1, FmTENE1] = CMMeasures(testClassE1, predictedTestClassRFE1);

% Store measures into different column vectors
AccuracyArray = [AccTRE0; AccTEE0; AccTRNE1; AccTENE1];
FmeasureArray = [FmTRE0; FmTEE0; FmTRNE1; FmTENE1];
RecallArray = [ReCallTRE0; ReCallTEE0; ReCallTRNE1; ReCallTENE1];
PrecisionArray = [PTRE0; PTEE0; PTRNE1; PTENE1]; 
Set = {'TrainE0'; 'TestE0'; 'TrainE1'; 'TestE1'};

% Create table with complete performance measures for every NB experiment
measureTableRF = table(AccuracyArray, FmeasureArray, RecallArray, ...
    PrecisionArray, 'RowNames', Set, 'VariableNames', ...
    {'Accuracy', 'Fmeasure', 'Recall', 'Precision'});

clearvars AccTRE0 AccTEE0 AccTRNE1 AccTENE1 FmTRE0 FmTEE0 FmTRNE1 FmTENE1 ...
    ReCallTRE0 ReCallTEE0 ReCallTRNE1 ReCallTENE1 PTRE0 PTEE0 PTRNE1 PTENE1 ...
    AccuracyArray FmeasureArray RecallArray PrecisionArray Set i j


%% ------------------------------------------------------------------------
% ROC visualisation
% -------------------------------------------------------------------------

% Calculte Posterior Probabilities and ROC curve for MdlNBE1
resp = table2array(testClassE1);
resp(resp==4)=1;
resp(resp==2)=0;
resp = logical(resp);
figure;hold on;
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curves for Experiment 1 - Normal NB and Optimal RF')


% Create ROC curves for each cross validation model (Naive Bayes)
for i=1:10
    [~,score_nb] = predict(MdlNBE1.Trained{i},testAttributesE1);
    [Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,2),'true');
    plot(Xnb,Ynb,'r','LineWidth',1);
end

% Create ROC curve for Random Forest
[~,score_rf] = predict(MdlRFE1,testAttributesE1);
[Xrf,Yrf,Trf,AUCrf] = perfcurve(resp,score_rf(:,2),'true');
plot(Xrf, Yrf, 'g' , 'LineWidth', 1.5, 'Color', 'k')
legend('Naive Bayes CV1', 'Naive Bayes CV2', 'Naive Bayes CV3', ...
    'Naive Bayes CV4', 'Naive Bayes CV5', 'Naive Bayes CV6', ...
    'Naive Bayes CV7', 'Naive Bayes CV8', 'Naive Bayes CV9', ...
    'Naive Bayes CV10', 'Random Forest', 'Location', 'Best');
hold off;


%% ------------------------------------------------------------------------
% Custom function
% -------------------------------------------------------------------------

%% Custom function to calculate confusion matrix and performance measures
% Inputs: class labels and predicted class labels
% Outputs: confusion matrix, recall, precision, accuracy and f-measure
function [CM, Recall, Precision, Accuracy, Fmeasure] = CMMeasures(InputClass, InputPredictedClass)
% Determine whether 10 fold cross validation was used
if size(InputPredictedClass, 2) == 1
    % No 10 fold cross validation
    % Calculate confusion matrix
    % | TN | FP |
    % | FN | TP |
    CM = confusionmat(table2array(InputClass), InputPredictedClass);
    % Retrieve TP, TN, FP, FN values
    TN = CM(1 ,1);
    TP = CM(2, 2);
    FN = CM(2, 1);
    FP = CM(1, 2);
else
    %10 fold cross validation
    % Initiate column vectors to store TP, TN, FP, FN values 
    TNA = zeros(1, 10);
    TPA = zeros(1, 10);
    FNA = zeros(1, 10);
    FPA = zeros(1, 10);
    % Loop over each training/test set to retrieve TP, TN, FP, FN values 
    for i = 1:10
        CMK = confusionmat(table2array(InputClass), InputPredictedClass(:,i));
        TNA(i) = CMK(1, 1);
        TPA(i) = CMK(2, 2);
        FNA(i) = CMK(2, 1);
        FPA(i) = CMK(1, 2);
    end
    % Retrieve mean of the 10 TP, TN, FP, FN values
    TN = mean(TNA);
    TP = mean(TPA);
    FN = mean(FNA);
    FP = mean(FPA);
    % Calculate confusion matrix
    % | TN | FP |
    % | FN | TP |
    CM = [round(TN) round(FP); round(FN) round(TP)];
end

% Recall calculation
Recall = TP / (TP + FN);
display(Recall);
% Precision calculation
Precision = TP / (TP + FP);
display(Precision);
% Accuracy calculation
Accuracy = (TP + TN) / (TP + TN + FP + FN);
display(Accuracy);
% F-measure calculation
Fmeasure = (2 * Precision * Recall) / (Precision + Recall);
display(Fmeasure);

end
