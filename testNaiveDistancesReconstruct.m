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
idx=1:500; %Some frames
dd=d(:,:,idx);
%Trust in measurments:
posSTD=1.3*ones(size(dd,1),size(dd,3));

%Removing NaN:
dd(isnan(dd))=0;
posSTD(missing(idx,:)')=1e3; %No idea where those markers are!
%Marking outliers as untrusted:
outBefore=mm.outlierDetectFast(dd);
posSTD(outBefore)=1e3; %No idea where those markers are!
%%
newDD=mm.reconstruct(dd,posSTD);
outAfter=mm.outlierDetectFast(newDD);
%%
figure;
idx=5;
plot3(dd(:,1,idx),dd(:,2,idx),dd(:,3,idx),'o')
text(dd(:,1,idx),dd(:,2,idx),dd(:,3,idx),labels)
view(3)
axis equal
hold on
plot3(newDD(:,1,idx),newDD(:,2,idx),newDD(:,3,idx),'kx','MarkerSize',10,'LineWidth',4)

