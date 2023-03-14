function [error1,error2,error5]=disparityEr(GT,test)
% % Normalize and resclae two images between 0 - 255 ( image format)
% GT = uint8(rescale(GT,0,255));
% test = uint8(rescale(test,0,255));
GT = round(GT);
%test = uint16(test);
[h1,w1]=size(GT);
[h2,w2]=size(test);
if(h1~=h2 || w1~=w2)
    
  GT=GT(1:h2,1:w2);
    
end
pixel_sum=h2*w2;
threshold2= 2;
threshold1= 1;
threshold5=5;
count2=0;
count1=0;
count5=0;
for i=1:h2
    for j=1:w2
        
        if(abs(GT(i,j)-test(i,j))>threshold1)
           count1= count1+1;
        end
        if(abs(GT(i,j)-test(i,j))>threshold2)
           count2= count2+1;
        end
        if(abs(GT(i,j)-test(i,j))>threshold5)
           count5= count5+1;
        end
        
    end
    
    
end

error1=count1/pixel_sum;
error2=count2/pixel_sum;
error5=count5/pixel_sum;
end