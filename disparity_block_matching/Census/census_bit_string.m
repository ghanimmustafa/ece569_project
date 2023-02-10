
function output = census_bit_string(A);

% result = 10011001
[col row] = size(A);

x = zeros(1,row*col-1);
k = length(x);
i = col ;
while i ~= 0  
    j = row;
    while j ~= 0
       if((i ~= floor(row/2)+1) || (j ~= floor(col/2) +1))
          x(k) = A(i,j);   
          k = k - 1;
       end
       j = j - 1;
    end
    i = i - 1;  
end

 y = 0;
 for i = 1 : length(x)
    y = y + (2^(length(x) - i) * x(i)) ;     
end
%x
%y
%count = 0;
%for i=1:length(x)
%    if(x(i) == 1) 
%       count = count +1; 
        
%    end

% end
output = y;