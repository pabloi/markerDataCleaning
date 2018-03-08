% Test getting a position from distances

%%
X1=randn(10,3);
knownPositions=X1(1:end-1,:);
targetPos=X1(end,:);
D=computeDistanceMatrix(X1);
knownDistances=D(end,1:end-1)';
weights=[];

%% test different noise levels
n=[0:.01:1];
for i=1:length(n)
x(i,:)=getPositionFromDistances(knownPositions,knownDistances+n(i)*randn(size(knownDistances)),weights);
end

%%
figure 
subplot(2,1,1)
plot3(targetPos(1),targetPos(2),targetPos(3),'o')
hold on
plot3(x(:,1),x(:,2),x(:,3),'.')
plot3(knownPositions(:,1),knownPositions(:,2), knownPositions(:,3),'x')

subplot(2,1,2)
d=sqrt(sum(bsxfun(@minus,x,targetPos).^2,2));
plot(n,d)