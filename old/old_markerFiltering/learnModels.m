%% Load
load C0001MarkerData.mat
data=M{3}; %Just one trial of data

%% Learn distance between markers model
[N,M,d]=size(data);
dataAux1=reshape(data,N,1,M,d);
dataAux2=reshape(data,N,M,1,d);
distances=sqrt(sum((dataAux1-dataAux2).^2,4));
distReduced=zeros(N,0);
for i=1:M
    aux=squeeze(distances(:,i,i+1:end));
    distReduced(:,end+[1:size(aux,2)])=aux;
end
mu=squeeze(nanmean(distReduced));
sigma=squeeze(nanstd(distReduced));
S=squeeze(nancov(distReduced));

%% Detect outliers
dd=dataToDistances(data)';
[outlierIndx]=detectOutliers(dd,mu',S,eye(length(mu)),zeros(size(S)));
