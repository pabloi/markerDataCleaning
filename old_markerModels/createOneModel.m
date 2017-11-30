function [D,sD,meanPos,biasPos] = createOneModel(inputData,Nclust)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

if isa(inputData,'char') %Assume it is the name of a .c3d file
    %Load c3d using btk, to be done
elseif isa(inputData,'orientedLabTimeSeries') %Assume it is a timeseries containing 3D data
    X=inputData.getOrientedData;
else %Assume it is a Nx3xM matrix
   X=inputData; 
end
if nargin<2 || isempty(Nclust)
    Nclust=100; %Training Nclust models
end

[N,dim,M]=size(X);
dist=nan(N,N,M);
for i=1:M
    dist(:,:,i)=computeDistanceMatrix(X(:,:,i));
end
dd=reshape(dist,N^2,M);
[p,c,a]=pca(dd');
kk=3;
projectors=p(:,1:kk); % First two PCs
projections=c(:,1:kk);

idx=kmeans(projections,Nclust);
% figure
% hold on
centroids=nan(Nclust,kk);
D=nan(N,N,Nclust);
sD=nan(N,N,Nclust);
meanPos=nan(N,dim,Nclust);
biasPos=nan(N,dim,Nclust);
 for i=1:Nclust
     %plot3(projections(idx==i,1),projections(idx==i,2),projections(idx==i,3),'.')
     %centroids(i,:)=nanmean(projections(idx==i,:));
     [D(:,:,i),sD(:,:,i),meanPos(:,:,i)]= createZeroModel(inputData(:,:,idx==i)); 
 end
 
 
if nargout>3
    %Doxy
else
    biasPos=zeros(N,dim);
 end

end

