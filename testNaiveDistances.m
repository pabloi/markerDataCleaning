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

%% Assess likelihood of individual frames
inds=[16500:17000];
dd=d(:,:,inds);
ll1=mm.naiveScoreMarkers(dd);
ll2=mm.indScoreMarkers(dd);
ll3=mm.medianScoreMarkers(dd);
ll4=mm.rankedScoreMarkers(dd);
ll=ll1;
ll(isnan(ll))=5;

%Compare scoring:
figure()
bad=[311:316];% + 100;
mbad=any(ll(:,bad)<-10,2);%bad markers on those frames
hold on
for i=1:length(mbad)
    if mbad(i)
        p1=plot(inds,ll1(i,:)');
        plot(inds,ll2(i,:)','-.','Color',p1.Color)
        plot(inds,ll3(i,:)','.-','Color',p1.Color)
        plot(inds,ll4(i,:)','--','Color',p1.Color)
    end
end
legend('Naive','Ind','Median','ranked')

figure;
subplot(1,2,1)
hold on
plot(inds,ll')
legend(mm.markerLabels)
axis([300 350 -100 0])
bad=any(ll<-10,1); %bad frames
bad=[311:316];% + 100;
mbad=any(ll(:,bad)<-10,2);%bad markers on those frames
subplot(1,2,2)
hold on
DD=nanmean(d(:,:,bad),3);
plot3(DD(:,1),DD(:,2),DD(:,3),'o','LineWidth',4)
text(DD(:,1),DD(:,2),DD(:,3),mm.markerLabels)
plot3(DD(mbad,1),DD(mbad,2),DD(mbad,3),'o','LineWidth',4)
axis equal
view(3)



