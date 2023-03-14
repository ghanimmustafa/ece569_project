
function output =census(img,census_size)

img_gray = rgb2gray(img);
[y,x]=size(img_gray);
          
borders = floor(census_size/2); % limit to exclude image borders when filtering
center= floor(census_size/2) + 1;
for(iy = 1+borders : y-borders)
    for(ix = 1+borders : x-borders)
        f=img_gray(iy-borders:iy+borders,ix-borders:ix+borders);
        iix=ix-borders;
        iiy=iy-borders;
   
       
            
           resulted_window= f>f(center,center);
            
            
            %img_out=resulted_window;
   
        result = census_bit_string(resulted_window) ;
        
       
        img_out(iiy,iix)=result ;
     
    end
end

% normalization to range of [0 255]
%a = 0;
%b = 255;
%output=uint8((img_out-min(min(img_out))).*(b-a)./(max(max(img_out))-min(min(img_out))) + a);
output=img_out;
end
