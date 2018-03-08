function [f,g]=distanceDistanceAll(x,kP,kD,w)
    xx=bsxfun(@minus,x,reshape(kP',1,size(kP,2),size(kP,1))); %M x dim x N
    normXX=sqrt(sum(xx.^2,2)); %M x 1 x N
    f=sum(sum((w'.*(squeeze(normXX)-kD')).^2)); %scalar
    gg1=2*w'.^2.*(squeeze(normXX)-kD'); %M x N
    gg2=bsxfun(@rdivide,xx,normXX); %M x dim x N
    gg=bsxfun(@times,reshape(gg1,size(gg2,1),1,size(gg2,3)),gg2); %M x dim x N
    g=sum(gg,3); %M x dim
    g=g(:);
end