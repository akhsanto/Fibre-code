%% Import data Wheat Seeds Kernel
data = readtable('seeds.txt');
numData = data{:,1:end-1};
cats = data{:,end};

%% TODO: Cluster the numeric data into three groups
% k-Means
grpK = kmeans(numData,3);

% GMM
gmmModel = fitgmdist(numData,3);
grpGmm = cluster(gmmModel,numData);

% Hierarchical Trees
Z = linkage(numData);
grpTree = cluster(Z,'maxclust',3);


%% Visualization
% PCA
[~,scrs] = pca(numData);

% k-Means
figure(1)
title('k-Means Clustering')
scatter(scrs(:,1),scrs(:,2),4,grpK)

% Cross-tabulation
figure(2)
bar(crosstab(grpK,cats),'stacked')

% GMM
figure(3)
title('GMM Clustering')
scatter(scrs(:,1),scrs(:,2),4,grpGmm)

% Cross-tabulation
figure(4)
bar(crosstab(grpGmm,cats),'stacked')

% Hierarchical Tree
figure(5)
title('Tree Clustering')
scatter(scrs(:,1),scrs(:,2),4,grpTree)

% Cross-tabulation
figure(6)
bar(crosstab(grpTree,cats),'stacked')
