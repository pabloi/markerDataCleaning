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
mm2 = naivePositions.learn(d,labels);
mmZ = naivePositions.learn(d,labels,3);
L1=mm.scoreMarkers(d);
L5=mm.scoreMarkersNaive(d2);
L2=mm.scoreMarkersFast(d2);
L3=mm.scoreMarkersMedian(d2);
L4=mm.scoreMarkersRanked(d2,2);
L6=mm.scoreMarkersRanked(d2,1);
for j=1:6 %Defining threshold as twice the distance between the 10-th and 50th percentile below the 50th percentile
    eval(['aux=prctile(L' num2str(j) ',[5,50],2);']);
    eval(['th' num2str(j) '=aux(:,2)-2*(aux(:,2)-aux(:,1));']);
end
LB1=mm2.scoreMarkers(d);
%% Assess likelihood of individual frames
inds=[16500:18000];
dd=d(:,:,inds);
d2=dd;
d2(7,1,580+[0:50])=d(7,1,580+[0:50])+30; %1mm displacement

ll1=mm.scoreMarkers(d2);
ll5=mm.scoreMarkersNaive(d2);
ll2=mm.scoreMarkersFast(d2);
ll3=mm.scoreMarkersMedian(d2);
ll4=mm.scoreMarkersRanked(d2,2);
ll6=mm.scoreMarkersRanked(d2,1);

th5=-2;
th4=-(3.4)^2/2;
th4=-(3.4)^2/2;
th6=-(3.6)^2/2;
oo5=ll5<th5;
oo4=ll4<th4;
oo6=ll6<th6;
oo1=ll1<-(3.6)^2/2;

% ll5=L2;
% ll=mm2.scoreMarkersNaive(d);
% ll2=mm2.scoreMarkersFast(d);
% ll3=mm2.scoreMarkersMedian(d);
% ll4=mm2.scoreMarkersRanked(d,3);
% ll6=mm2.scoreMarkersRanked(d,1);
%%
figure; 
ph(1)=subplot(4,2,1); imagesc(ll5); caxis([-5 0]); title('Naive Likelihood'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))
ph(2)=subplot(4,2,3); imagesc(ll4); caxis([-5 0]); title('Ranked 3rd'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))
ph(3)=subplot(4,2,5); imagesc(ll6); caxis([-5 0]); title('Ranked 1st'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))
ph(4)=subplot(4,2,7); imagesc(ll1); caxis([-5 0]); title('Sophisticated'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))


ph(5)=subplot(4,2,2); imagesc(oo5); caxis([0 1]); title('Naive outliers'); colorbar;set(gca,'YTickLabels',labels,'YTick',1:length(labels))
ph(6)=subplot(4,2,4); imagesc(oo4); caxis([0 1]); title('Ranked 3rd'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))
ph(7)=subplot(4,2,6); imagesc(oo6); caxis([0 1]); title('Ranked 1st'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))
ph(8)=subplot(4,2,8); imagesc(oo1); caxis([0 1]); title('Sophisticated'); colorbar; set(gca,'YTickLabels',labels,'YTick',1:length(labels))

linkaxes(ph,'xy')
%%
oo1=mm.outlierDetectFast(dd);
oo2=mm.outlierDetect(dd);
oo3=ll5<2*prctile(L',3)';


%% Compare scoring:
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
        plot(inds,ll5(i,:)','x-','Color',p1.Color)
    end
end
legend('Naive','Ind','Median','ranked')

figure;
subplot(1,2,1)
hold on
p1=plot(inds,ll');
%axis([300 350 -100 0])
bad=any(ll<-10,1); %bad frames
bad=[311:316];% + 100;
mbad=any(ll(:,bad)<-10,2);%bad markers on those frames
ooo1=nan(size(oo1));
ooo2=nan(size(oo2));
ooo3=nan(size(oo3));
ooo1(oo1)=1;
ooo2(oo2~=0)=1;
ooo3(oo3~=0)=2;
set(gca,'ColorOrderIndex',1)
plot(inds,ooo1,'o')
set(gca,'ColorOrderIndex',1)
plot(inds,ooo2,'x')
set(gca,'ColorOrderIndex',1)
plot(inds,ooo3,'o')
legend(p1,mm.markerLabels)
subplot(1,2,2)
hold on
DD=nanmean(d(:,:,16570),3);
plot3(DD(:,1),DD(:,2),DD(:,3),'o','LineWidth',4)
text(DD(:,1),DD(:,2),DD(:,3),mm.markerLabels)
%plot3(DD(mbad,1),DD(mbad,2),DD(mbad,3),'o','LineWidth',4)
axis equal
view(3)



