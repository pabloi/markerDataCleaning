% Test distance model:

%% Load data
clearvars

load ./data/C0001MarkerData.mat
data2=M{7};
labels2=labels;

load ./data/LI16_Trial9_expData.mat %processedTrialData
labels=LI16_Trial9_expData.markerData.getLabelPrefix;
data=LI16_Trial9_expData.markerData.getOrientedData(labels);

iL=cellfun(@(x) ~isempty(x),regexp(labels,'^L*'));
iR=cellfun(@(x) ~isempty(x),regexp(labels,'^R*'));
dL=data(:,iL,:);
lL=labels(iL);
dR=data(:,iR,:);
lR=labels(iR);
[~,idx1]=sort(nanmean(dL(:,:,3)),'ascend');
[~,idx2]=sort(nanmean(dR(:,:,3)),'descend');
labels=[lL(idx1) lR(idx2)];
data=cat(2,dL(:,idx1,:),dR(:,idx2,:));

%%
missing=any(isnan(data),3);
figure
miss=missing(:,any(missing));
pp=plot(miss,'o');
aux=labels(any(missing));
for i=1:length(pp)
    set(pp(i),'DisplayName',[aux{i} ' (' num2str(sum(miss(:,i))) ' frames)'])
end
legend(pp)
title('Missing markers')
xlabel('Time (frames)')
set(gca,'YTick',[0 1],'YTickLabel',{'Present','Missing'})

%% Learn a model
d=permute(data,[2,3,1]);
mm = naiveDistances.learn(d,labels);

%%
idx=1:2; %Some frames
dd=d(:,:,idx);
dd(isnan(dd))=0; %Marking missing data to arbitrary locations
posSTD=ones(size(dd,1),size(dd,3));
posSTD(missing(idx,:)')=1e5; %No idea where those markers are!
%%
newDD=mm.reconstruct(dd,posSTD);