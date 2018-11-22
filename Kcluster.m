%Kmenas
%decides K be elbow methods
clusterNumber=10;

distortion=zeros(clusterNumber,1);
for K=1:clusterNumber
    [Idx,C,distortionWithin,D]=kmeans(reduced_X,K);
    distortion(K)=sum(distortionWithin);
end
%choose K=10 for now


%calculate labels for each cluster
labeldimension=size(train_labels,2);
MeanLabels=zeros(clusterNumber,labeldimension);
for K=1:clusterNumber
    index=find(Idx==K);
    num=size(index,1);    
    LabelsInEachCluster=zeros(num,labeldimension);
    for i=1:size(index,1)
        LabelsInEachCluster(i,:)=train_labels(index(i),:);
    end
    MeanLabels(K,:)=mean(LabelsInEachCluster);   
end


%assign mean labels to points in each cluster
N=size(train_inputs,1);
predict_labels=zeros(N,labeldimension);
for i=1:N
    ClusterID=Idx(i);
    predict_labels(i,:)=MeanLabels(ClusterID,:);
end


%cluster test inputs
%assuming test inputs is reduced to the same dimension as train inputs
%So test_input and train_input should do dimention reduction together
%to ensure they have the same dimension???

% predicted_labels=zeros(N,labeldimension);
% for i=1:N
%     distance=zeros(clusterNumber,1);
%     for j=1:clusterNumber
%         distance(j,1)=norm(test_input(i,:)-C(j,:));
%     end
%     minDistance=min(distance);
%     clusterid=find(distance==minDistance);
%     predicted_labels(i,:)=MeanLabels(clusterid,:);
% end
        
    

