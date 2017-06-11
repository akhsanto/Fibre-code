%You can use bsxfun as shown below. Note that @gt refers to the built-in “greater than” function.
% >> x = [4;5;6]
% >> y = [1 2 3; 4 5 6; 7 8 9];
% >> bsxfun(@gt,x,y)
% ans =
%     1   1   1
%     1   0   0
%     0   0   0

%% Import & initialize data
%%be careful this is the example of unsupervised dataset
data = readtable('basketballDataProcessed.csv');
data.pos = categorical(data.pos);
%% TODO - TASK 1: Calculate the per-game statistics
stats = data{:,7:end};
statsnorm = bsxfun(@rdivide,stats,data.GP);
data{:,7:end} = statsnorm;
% Get numeric columns and normalize them
stats = data{:,[ 5 6 11:end ]}; %get column data 5,6,11 end
labels = data.Properties.VariableNames([ 5 6 11:end ]); %give labels

%% TODO - TASK 2: Shift and scale the variables
%Normalizing Data: zscore, still same coordinate, to simplify amount of
%each value
statsNorm = zscore(stats); %give the score

%% TODO - TASK 1: Use CMD scaling, then plot the pareto chart and a 3-D scatter plot
% Use CMD to scale the observations, multi dimensional scaling, 
% Use the pdist function to calculate the pairwise distances between observations
pd = pdist(statsNorm);
%two dimension
% Xi = mdscale(pd,2); error-becareful
%[configMat,eigenVal] = cmdscale(distMat)
[X,e] = cmdscale(pd);
%two dimension= configMat = mdscale(distMat,numDims)
%the value of X would be similar with scrs, e is score each rows
% Create a pareto chart
fig1 = figure(1);
pareto(e)

% Create a scatter plot of reconstructed coordinates,visualising score vs
% height vs assisted
fig2 = figure(2);
scatter3(X(:,1),X(:,2),X(:,3))

view(110,40)

% Reconstruct coordinates
[pcs,scrs] = pca(statsNorm);

%% TODO: Group data using k-means clustering
% The following command repeats the clustering 5 times and returns the clusters with lowest sumd.
%from centroid in each cluster with the closest one
%sqeuclidean: default value of the Distance measure property in kmeans
grp = kmeans(statsNorm,2,'Replicates',5);
%above is iterative 5 times, 2 cluester at the end
%% View data
%following functions will return the probabilities associated with each cluster for each observation=cluster
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,grp)
view(0,90)
%% TODO - TASK 2: Use PCA, then plot the pareto chart and a 3-D scatter plot
% PCA scoring principal coordinate in 2 axis= score*value; scrs=total score
% each column, pexp=
%PCA=measure how much variance in the data
%CMD=measure how close in the data
[~,scrs,~,~,pexp] = pca(statsNorm);

% Create pareto chart
fig3 = figure(3);
pareto(pexp)

% Create scatter plot of principal component scores
fig4 = figure(4);
scatter3(scrs(:,1),scrs(:,2),scrs(:,3))
view(110,40)

%% TODO TASK 3: Group data using k-means clustering, clustering into 5 group
%&grp=group
% grp = kmeans(X,2);
grp = kmeans(statsNorm,2,'Replicates',5);
%to know value each cluster
silhouette(X,grp)
figure(111)

%% View data
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,grp)  
view(110,40)

%% TODO TASK 4: Group data using GMM
%to find the most means value in clustering
gmModel = fitgmdist(statsNorm,2,'Replicates',5,'RegularizationValue',0.02);
%to decide cluster each data
[grp,~,gprob] = cluster(gmModel,statsNorm);
scatter(X(:,1),X(:,2),10,grp)
%% View data
% Visualize groups
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,grp)
view(110,40)

% Visualize group separation
gpsort = sortrows(gprob,1);
figure
plot(gpsort)
xlabel('Point Ranking')
ylabel('Cluster Membership Probability')
legend('Cluster 1','Cluster 2')

%%TASK 5 %% Group data using GMM
gmModel = fitgmdist(statsNorm,2,'Replicates',5,'RegularizationValue',0.02);
grp = cluster(gmModel,statsNorm);

%% Evaluate clusters
figure(5)
subplot(2,1,1)
%% TODO - TASK 1: Create stacked bar chart of results
%Info: You can use the bar function with the 'stacked' option to visualize the counts.
%we can see which position that have tendency in cluster 1 or cluster 2, so
%we could improve it in the future
bar(crosstab(grp,data.pos),'stacked')
positions = categories(data.pos);
legend(positions)

subplot(2,1,2)
%% TODO - TASK 2: Plot mean values for each observation
parallelcoords(gmModel.mu,'Group',1:2)

%% TODO - TASK 3: Determine the optimal number of groupings
%If you are not certain about how many groups to use, the evalclusters function can help you make the decision.
clustEv = evalclusters(X,'kmeans','silhouette','KList',2:10)
cev = evalclusters(statsNorm,'gmdistribution','DaviesBouldin','KList',2:4)
disp(['The optimal number of groupings is ',num2str(cev.OptimalK)])

% Get guard positions
idx = data.pos == 'G';

% Get numeric columns and normalize them
stats = data{:,[ 5 6 11:end ]};
labels = data.Properties.VariableNames([ 5 6 11:end ]);
%only Guard position
guardstats = zscore(stats(idx,:));

% Get guard positions
idx = data.pos == 'G';

% Get numeric columns and normalize them
stats = data{:,[ 5 6 11:end ]};
labels = data.Properties.VariableNames([ 5 6 11:end ]);
guardstats = zscore(stats(idx,:));

%% TODO: Cluster groups using hierarchical clustering techniques
%Info: The 'ward' method computes the inner squared distance using Ward's minimum variance algorithm.
%Info: You can use the linkage function to encode a tree of hierarchical clusters from a set of observations.
Z = linkage(guardstats,'ward');

% Visualize the hierarchy
figure(11)
dendrogram(Z)

% Make clusters for two groups and three groups
gc2 = cluster(Z,'maxclust',2);
gc3 = cluster(Z,'maxclust',3);

%% Visualize the clusters using parallelcoords
figure(112)
parallelcoords(gmModel.mu,'Group',1:2)
%how it correlates at the end
figure(12)
parallelcoords(X,'Group',grp)
figure(9)
parallelcoords(guardstats,'Group',gc2,'Quantile',0.25)
% labelXTicks(labels);

figure(13)
parallelcoords(guardstats,'Group',gc3,'Quantile',0.25)
% labelXTicks(labels);
