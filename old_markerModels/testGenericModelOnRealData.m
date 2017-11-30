%% Test model
modelName='One';
%modelName='Zero';
eval(['createFunc=@create' modelName 'Model;'])
eval(['likeliFunc=@determineLikelihoodFrom' modelName 'Model;'])
eval(['reconsFunc=@getBestReconsFrom' modelName 'Model;'])

%%
% clear M
% labels=expData.data{3}.markerData.getLabelPrefix;
% for i=1:length(expData.data)
%     if ~isempty(expData.data{i})
%         M{i}=expData.data{i}.markerData.getOrientedData(labels); %Get data in order
%     end
% end
% labels=expData.data{3}.markerData.getLabelPrefix;
% save C0001MarkerData.mat -v7.3 M labels
%%
load C0001MarkerData.mat
X1=M{4};
X1=permute(X1,[2,3,1]);

[D,sD,meanPos] = createFunc(X1);
[lp1,p] = likeliFunc(X1,D,sD);

save(['C0001' modelName 'Model_k' num2str(size(D,3)) '.mat'], 'D', 'sD', 'meanPos', 'lp1', 'p','X1')
%% see model
f=figure;
kk=1:(size(D,3)/5):size(D,3);
switch modelName
    case 'Zero'
        MM=1;
    case 'One'
        MM=5;
end

for k=1:MM
subplot(MM,2,1+(2*(k-1)))
spy(D(:,1:size(X1,1),kk(k))<250 & D(:,1:size(X1,1),kk(k))>0)
if k==MM
set(gca,'XTick',1:18,'XTickLabel',labels,'XTickLabelRotation',90,'YTick',1:18,'YTickLabel',labels)
end
title(['Distances less than 250mm, model ' num2str(kk(k))])
subplot(MM,2,2+(2*(k-1)))
spy(sD(:,1:size(X1,1),kk(k))<10 & sD(:,1:size(X1,1),kk(k))>0)
%imagesc(sD)
title('Distances with std() less than 10mm')
axis equal
axis tight
%colorbar
if k==MM
set(gca,'XTick',1:18,'XTickLabel',labels,'XTickLabelRotation',90,'YTick',1:18,'YTickLabel',labels)
end

end
%saveFig(f,'./',[modelName 'Model_k' num2str(size(D,3))]);

%% Test model on training data:
idxs=3500:3620;
idxs=3500:3620;
idxs=1:size(X1,3);
idxs=330:460;

X1a=X1(:,:,idxs);
for i=1:size(lp1,1)
    %outliers=find(lp1(i,idxs)<-5);
    %X1a(i,:,outliers)=nan;
end
Xbar=nan(size(X1a));
lp2a=nan(size(X1,1),length(idxs));
for i=1:size(X1a,3)
    i
    [Xbar(:,:,i),lp2a(:,i)]=reconsFunc(X1a(:,:,i),D,sD,meanPos); 
end

%save(['C0001' modelName 'ModelReconstructTrainingData_top5_k' num2str(size(D,3)) '.mat'], 'X1a',  'idxs', 'Xbar', 'D', 'sD', 'meanPos', 'lp1', 'p','lp2a')

%% plot
e=squeeze(sqrt(sum((Xbar-X1a).^2,2)));
dd=squeeze(Xbar-X1a);

[lp2a,pp2a] = likeliFunc(Xbar,D,sD);
f=figure('Name','Quantify reconstruction');
subplot(5,2,1)
hold on
plot(lp1')
%plot(find(any(isnan(lp1))),-5,'k.')
xlabel('Frames')
ylabel('Log-like')
legend([labels])

subplot(5,2,3)
hold on
lp3=lp1(:,idxs);
p1=plot(idxs,lp3','Color',[.3,.3,.6]);
p2a=plot(idxs,lp2a','Color',[.6,.3,.3]);
p1=plot(idxs,nanmean(lp3',2),'b','LineWidth',2);
p2a=plot(idxs,nanmean(lp2a',2),'r','LineWidth',2);
xlabel('Frames')
legend([p1(1) p2a(1)],{'Original data','Reconstructed data'})
ylabel('Log-like')

subplot(5,2,5)
hold on
plot(idxs,nanmean(e))
plot(idxs,min(e))
plot(idxs,max(e))
plot(idxs,nanmedian(e))
plot(idxs(squeeze(any(isnan(X1a(:,1,:)),1))),1,'k.')
legend('Mean','Min','Max','Median')
ylabel('RMS error')
xlabel('Frames')

subplot(5,2,1+[6:2:8])
hold on

auxInd=lp3(:)>-4;
e2=e;
e2(~auxInd)=nan;
boxplot(e2')
axis([0 19 0 100])
plot(sqrt(sum(mean(dd,3).^2,2)),'x')
axis([0 19 0 100])
set(gca,'XTick',1:18,'XTickLabel',labels,'XTickLabelRotation',90)
ylabel('Reconstruction error (mm)')
subplot(5,2,2+[0:2:8])
hold on

%plot3(mean(X1a(:,1,:),3),mean(X1a(:,2,:),3),mean(X1a(:,3,:),3),'o')
%text(mean(X1a(:,1,:),3),mean(X1a(:,2,:),3),mean(X1a(:,3,:),3),labels)
%plot3(mean(Xbar(:,1,:),3),mean(Xbar(:,2,:),3),mean(Xbar(:,3,:),3),'x')
[~,frameNo]=min(mean(lp1(:,idxs),1));
%frameNo=5076;
%frameNo=10;
[~,frameNo]=min(mean(lp2a(:,:),1));
p1=plot3(mean(X1(:,1,idxs(frameNo)),3),mean(X1(:,2,idxs(frameNo)),3),mean(X1(:,3,idxs(frameNo)),3),'o');
p2=plot3(mean(Xbar(:,1,frameNo),3),mean(Xbar(:,2,frameNo),3),mean(Xbar(:,3,frameNo),3),'*');
text(mean(Xbar(:,1,frameNo),3),mean(Xbar(:,2,frameNo),3),mean(Xbar(:,3,frameNo),3),labels)
text(mean(Xbar(:,1,frameNo),3),mean(Xbar(:,2,frameNo),3),mean(Xbar(:,3,frameNo),3)+20,num2str(lp2a(:,frameNo)),'Color',p2.Color)
text(mean(X1(:,1,idxs(frameNo)),3),mean(X1(:,2,idxs(frameNo)),3),mean(X1(:,3,idxs(frameNo)),3)-20,num2str(lp1(:,idxs(frameNo))),'Color',p1.Color)
title(['Data for frame #' num2str(idxs(frameNo))])

view(3)
legend('Mean position from actual data','Mean reconstruction of best-fit data')
%saveFig(f,'./',[modelName 'Model_Results_bestThird_k' num2str(size(D,3))]);

%% find bad markers in a testing dataset

X2=M{9};
X2=permute(X2,[2,3,1]);
[lp2] = likeliFunc(X2,D,sD);

% Find outliers
%[i,j]=find(lp1<-5);
[i,j]=find(lp2<-14);
%%
figure
subplot(1,2,1)
hold on
%plot(lp1','b')
plot(lp2','r')
plot(j,lp2(sub2ind(size(lp2),i,j)),'o')
subplot(1,2,2)
hold on
n=5;
jj=j(n);
ii=i(n);
p1=plot3(X2(:,1,jj),X2(:,2,jj),X2(:,3,jj),'o'); % All positions
text(X2(:,1,jj),X2(:,2,jj),X2(:,3,jj),labels,'Color',p1.Color)
%[R,t,meanPos2]=getRotationAndTranslation(meanPos,X2(:,:,jj));
%plot3(meanPos2(:,1),meanPos2(:,2),meanPos2(:,3),'go'); % All positions
%text(meanPos2(:,1),meanPos2(:,2),meanPos2(:,3),labels,'Color','g');
p2=plot3(X2(ii,1,jj),X2(ii,2,jj),X2(ii,3,jj),'x'); %Bad(?) marker
[X2alt] = reconsFunc(X2(:,:,jj),D,sD,meanPos);
p3=plot3(X2alt(:,1),X2alt(:,2),X2alt(:,3),'*');
p4=plot3(X2alt(ii,1),X2alt(ii,2),X2alt(ii,3),'m*');
title(['log-p=' num2str(lp2(ii,jj))])
axis equal
legend('Actual data','Bad data','Estimated data')