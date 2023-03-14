GT = dispG;
test = (md);
[h1,w1]=size(GT);
[h2,w2]=size(test);
if(h1~=h2 || w1~=w2)
    
  GT=GT(1:h2,1:w2);
    
end
cnt = 0
for i=1:h2
    for j=1:w2
       if(abs(GT(i,j)-test(i,j)) < 1)
           %fprintf('Disparity at position %i:%i in the output result is equal!\n',i,j);
           cnt = cnt + 1; 
       end
        
        
    end
    
    
end
fprintf(' in total there are %i many equal pixels out of %i, accuracy is = %i\n',cnt,h2*w2,(100 * cnt/(h2*w2)));