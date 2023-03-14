function D = pfmread(filename_pfm)
 
fid = fopen(filename_pfm);
 
fscanf(fid,'%c',[1,3]);
cols = fscanf(fid,'%f',1);
rows = fscanf(fid,'%f',1);
fscanf(fid,'%f',1);
fscanf(fid,'%c',1);
D = fread(fid,[cols,rows],'single');
D(D == Inf) = 0;
D = rot90(D);
fclose(fid);
end

% function pfmwrite(D, filename)
% % assert(size(D, 3) == 1 & (isa(D, 'single') ));
%  
% [rows, cols] = size(D);
% scale = -1.0/ max(max(D));
% fid = fopen(filename, 'wb');
%  
% fprintf(fid, 'Pf\n');
% fprintf(fid, '%d %d\n', cols, rows);
% fprintf(fid, '%f\n', scale);
% %fscanf(fid, '%c', 1);
%  
% fwrite(fid, D(end:-1:1, :)', 'single');
% fclose(fid);
% end