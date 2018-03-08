%% test RotationAndTranslation
%% data
X1=randn(10,3);
X1=meanPos;
X2=bsxfun(@plus,X1*[0 1 0; 1 0 0; 0 0 -1] ,[1 2 3]) +.1*randn(size(X1));

%%
[R,t,X1p] = getRotationAndTranslation(X1,X2);
R
t

%% Plot
figure
hold on
plot3(X2(:,1),X2(:,2),X2(:,3),'o')
plot3(X1p(:,1),X1p(:,2),X1p(:,3),'x')
hold off