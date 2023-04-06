function output=my_median_filter(noisy,only_filtered)
[m,n]= size(noisy);
output=zeros(m,n);
output=single(output);

%% This filter keeps the boundires unfiltered to match the hardware design of it. 

for i=1:m
    for j=1:n
       
        xmin=max(1,i-1); % min x coordinate has to be greater than or equal
        % to one
        xmax=min(m,i+1);
        ymin=max(1,j-1);
        ymax=min(n,j+1);
        % the neighbourhood matrix then will be 
        temp=noisy(xmin:xmax,ymin:ymax);
        % the new intensity of pixel at (i,j) will be median of this matrix:
        
        output(i,j)=median(temp(:));
 
        
        
        
    end
    
end

        output(1,:) = noisy(1,:);
        output(end,:) =  noisy(end,:);
        output(:,1) = noisy(:,1);
        output(:,end) =  noisy(:,end); 



        if(only_filtered == true)

            output = output(2:(end-1),:);
        
        
        end
        
end



