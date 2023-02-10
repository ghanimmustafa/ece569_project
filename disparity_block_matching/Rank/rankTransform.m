
function output =rankTransform(img,rankSize)
%img = imread(input);

img_gray = rgb2gray(img);
[y,x]=size(img_gray);

            
borders = floor(rankSize/2) ; % limit to exclude image borders when filtering
center = borders + 1;
for(iy = 1+borders : y-borders)
    for(ix = 1+borders : x-borders)
        f=img_gray(iy-borders:iy+borders,ix-borders:ix+borders);
        iix=ix-borders;
        iiy=iy-borders;
    img_out(iiy,iix)=sum(sum(f<(f(center,center )))); % Rank transform
    end
end
%output = img_out ./ max(max(img_out)) ;
output = img_out;
%imshow(normalised_image);

end
