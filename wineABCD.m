%% Import data,wineABCD determine the wine quality
redData = readtable('wineabcd.txt');

% Convert quality data to categorical
redData.QCLabel = categorical(redData.QCLabel);

%% TODO - TASK 1: Split the data into training and test sets
% Create the cvpartition variable
pt = cvpartition(redData.QCLabel,'HoldOut',0.3);

% Create the training and test tables
redWineTrain = redData(training(pt),:);
redWineTest = redData(test(pt),:);

disp([num2str(height(redData)),' original observations split into:'])
disp(['   ',num2str(height(redWineTrain)),' training observations'])
disp(['   ',num2str(height(redWineTest)),' test observations'])

%% TODO - TASK 2: Predict quality
% Create classification model
knnModel = fitcknn(redWineTrain,'QCLabel');

% Evaluate quality of classifier
errRate = loss(knnModel,redWineTest);
disp(['The error rate is ',num2str(errRate)])

% Predict the quality for the test data
predictedQuality = predict(knnModel,redWineTest);
confMat = confusionmat(redWineTest.QCLabel,predictedQuality);
disp('The confusion matrix:')
disp(confMat)