function out = SortAndIndex(in)

[row,col] = size(in);
v=in(:);
s=sort(v,'ascend');
[idx,loc] = ismember(v,s);
out = vec2mat(loc,col)';

end



