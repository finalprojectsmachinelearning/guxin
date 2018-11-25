function pred_labels=gmm(train_inputs,train_labels,test_inputs)

total_data=[train_inputs;test_inputs];
topics=total_data(:,22:2021);
train_points=size(train_inputs,1);
test_points=size(test_inputs,1);
all_points=size(total_data,1);
labeldimension=size(train_labels,2);

clusterNumber=10;

[COEFF,SCORE1,latent]=pca(topics);
accu=cumsum(latent)./sum(latent);
col=0;

for i=1:size(accu,1)
    col=col+1;
    if accu(i) > 0.99
        break
    end
end

reduced_trainingdata=[train_inputs(:,1:21),SCORE1(1:train_points,1:col)];
reduced_testingdata=[test_inputs(:,1:21),SCORE1(train_points+1:all_points,1:col)];


[Idx,C,distortionWithin,D]=kmeans(reduced_trainingdata,clusterNumber,'start','uniform'); 


S.mu=C;
d=size(reduced_trainingdata,2);
k=clusterNumber;
S.Sigma=ones(1,d,k);
S.ComponentProportion=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1];
options=statset('MaxIter',10000);

gm = gmdistribution.fit(reduced_trainingdata,10,'CovarianceType','diagonal','Regularize',1e-10,'Options',options,'Start',S);
result = posterior(gm, reduced_trainingdata) ;
result_test = posterior(gm, reduced_testingdata) ;

gm2 = gmdistribution.fit(reduced_trainingdata,10,'CovarianceType','diagonal','Regularize',1e-10,'Options',options,'Start','plus');
result2 = posterior(gm2, reduced_trainingdata) ;
result2_test = posterior(gm2, reduced_testingdata) ;

%assign label on result
trainpost_possibility=result2;
testpost_possibility=result2_test;

traincluster=zeros(train_points,1);
testcluster=zeros(test_points,1);
for i=1:train_points
    max_poss=max(trainpost_possibility(i,:));    
    index=find(trainpost_possibility(i,:)==max_poss(1));
    traincluster(i)=index;
    
end
for i=1:test_points
    max_poss_test=max(testpost_possibility(i,:));
    index_test=find(testpost_possibility(i,:)==max_poss_test(1));
    testcluster(i)=index_test;
end

%calculate mean labels for each cluster
for K=1:clusterNumber
    point=find(Idx==K);
    num=size(point,1);    
    LabelsInEachCluster=zeros(num,labeldimension);
    for i=1:num
        LabelsInEachCluster(i,:)=train_labels(point(i),:);
    end
    MeanLabels(K,:)=mean(LabelsInEachCluster);   
end


%assign mean labels to points in each cluster
predict_labels=zeros(train_points,labeldimension);
for i=1:train_points
    ClusterID=Idx(i);
    predict_labels(i,:)=MeanLabels(ClusterID,:);
end

% predict_labels_test=zeros(test_points,labeldimension);
% for i=1:test_points
%     ClusterID_test=testcluster(i);
%     predict_labels_test(i,:)=MeanLabels(ClusterID_test,:);
% end

pred_testlabels=zeros(test_points,labeldimension);
for i=1:test_points
    distance=zeros(clusterNumber,1);
    for j=1:clusterNumber
        distance(j,1)=norm(reduced_testingdata(i,:)-C(j,:));
    end
    minDistance=min(distance);
    clusterid=find(distance==minDistance);
    pred_testlabels(i,:)=MeanLabels(clusterid(1),:);
end


%pred_labels=predict_labels_test;
pred_labels=predict_labels;

end

