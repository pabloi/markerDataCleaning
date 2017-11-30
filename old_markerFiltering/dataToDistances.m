function [distances] = dataToDistances(data)

[N,M,d]=size(data);
idxMap=zeros(0,2);
for i=1:M
    aux=[i*ones(M-i,1) [i+1:M]'];
    idxMap(end+[1:size(aux,1)],:)=aux;
end

distances=sqrt(sum((data(:,idxMap(:,1),:)-data(:,idxMap(:,2),:)).^2,3));


end

