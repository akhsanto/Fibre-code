%% Import the data, if you misclassify the poisonous mushroom, the cause would be fatal, test the mushroom
mushrooms = readtable('mushrooms.txt','Format',repmat('%C',1,7));

%% TODO - TASK 1: Split the data into training and test sets
% Create the cvpartition variable, this is the weakness of splitting data
% it would result different values because they take only amount of defined
% dataset for training machine process
cvpt = cvpartition(mushrooms.edibility,'holdout',0.25);

% Create the training and test tables
mushTrain = mushrooms(training(cvpt),:);
mushTest = mushrooms(test(cvpt),:);

%% TODO - TASK 2: Create a k-NN classifier model
%Create a k-NN classifier model
mdl = fitcknn(mushTrain,'edibility');

%% TODO - TASK 3: Predict the loss on the test set and the resubstitution loss
errTrain = resubLoss(mdl);
disp(['The error on the training set is ',num2str(errTrain)])
errTest = loss(mdl,mushTest);
disp(['The error on the test set is ',num2str(errTest)])


%% TODO - TASK 4: Calculate the percentage of poisonous mushrooms that were classified as edible
prediction = predict(mdl,mushTest);
[cm,cl] = confusionmat(mushTest.edibility,prediction);
misClass = cm(cl=='poisonous',cl=='edible');
errPoison = 100*misClass/height(mushTest);
disp(['The percentage of misclassified poisonous mushrooms is ',num2str(errPoison),'%'])
