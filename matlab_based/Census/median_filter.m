function output=median_filter(noisy)
[m,n]= size(noisy);
output=zeros(m,n);
output=uint16(output);
% Intensity of pixel in the noisy image is given as noisy(i,j)
% Here we define max and min values of x and y coordinates of any
% pixel can take
% This is done since some pixels in neighborhood will go out of bounds
% some constraints need to be followed
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
        
end



