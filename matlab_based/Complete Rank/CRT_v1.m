
function output =CRT(img,rankSize,RGB)
%img = imread(input);
if(RGB)
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
[y,x]=size(img_gray);
%img_out = zeros(y,x,9);
            
borders = floor(rankSize/2); % limit to exclude image borders when filtering
for(iy = 1+borders : y-borders)
    
    for(ix = 1+borders : x-borders)
        f=img_gray(iy-borders:iy+borders,ix-borders:ix+borders);
        
        iix=ix-borders;
        iiy=iy-borders;
        tmp = SortAndIndex(f);
        tmp = reshape(tmp',1,[]);
		img_out(iiy,iix,:)=tmp; % Complete Rank transform, 
    end
    
end
output = img_out - 1; % order starts from zero.
%output = img_out ./ max(max(img_out)) ;
%imshow(normalised_image);

end
