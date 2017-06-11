%% Import data
bonds = readtable('bond.txt');
% Extract numeric data
bonds = bonds{:,:};

%% TODO - TASK 1: Use k-means to cluster the numeric data into three groups
nc = 3;
distK = 'cosine';
kGrp = kmeans(bonds,nc,'Distance',distK);

%% TODO - TASK 2: Determine the optimal number of clusters
eva = evalclusters(bonds,'kmeans','silhouette','KList',2:7,'Distance',distK);
numClust = eva.OptimalK;
disp(['The optimal number of clusters is ',num2str(numClust)])

  T = readtable('bond.txt')
my_directory = 'F:\Machine Learning with Matlab';  
Pulsdauer = [my_directory filesep 'SomeFileName.txt'];
writetable(T,Pulsdauer);