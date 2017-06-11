%supervised data learning no correct answer

%% Import data
heartData = readtable('heartDiseaseData.txt');
heartData2 = readtable('heartDiseaseData.csv');
groupData= heartData(:,[ 1 2 end])
stats=heartData(:,[ end ])

labels = heartData.Properties.VariableNames(:,[ 1 2 end])
groupData.Properties.VariableNames = labels
% plot(groupData,'x')
% hold on
cvpt = cvpartition(groupData.HeartDisease,'holdout',0.35);
dataTrain = groupData(training(cvpt),:);
dataTest = groupData(test(cvpt),:);
mdl = fitcknn(dataTrain,'HeartDisease');
predictedGroups = predict(mdl,dataTest);
% plot(dataTest,predictedGroups,'o')
testErr = loss(mdl,dataTest)
%%c = confusionmat(originalData,...
%predictedData)
%Apply the confusionmat function to the predicted data
cm = confusionmat(dataTest.group,predictedGroups)

%%Find the cost matrix of the model.
costMat = mdl.Cost
heartData.HeartDisease = categorical(heartData.HeartDisease);
labelsfull = heartData.Properties.VariableNames([ 	1:end ]); %give labels
%% Split the data into training and test sets
% Create the cvpartition variable, check training iteration size, take 30%
pt = cvpartition(heartData.HeartDisease,'HoldOut',0.3);

% Create the training and test tables
hdTrain = heartData(training(pt),:);
hdTest = heartData(test(pt),:);

%% Create a model and predict the outcome
mdl = fitcknn(hdTrain,'HeartDisease');

%% TODO - TASK 1: Calculate the loss errTrain and errTest
errTrain = resubLoss(mdl);
errTest = loss(mdl,hdTest);
disp(['Training Error: ',num2str(errTrain)])
disp(['Test Error: ',num2str(errTest)])

%% TODO - TASK 2: Calculate falseNeg
predictions = predict(mdl,hdTest);

%confusion matrix
[cm,cl] = confusionmat(hdTest.HeartDisease,predictions);
misClass = cm(cl=='true',cl=='false');
falseNeg = 100*misClass/height(hdTest);
disp(['Percentage of False Negatives: ',num2str(falseNeg),'%'])