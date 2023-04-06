clc

test=1;


for census_size=7
	for search_size=1
        
          if((search_size==3 && census_size==7) || (search_size==7 && census_size==3))
                fprintf('Not Considered case of census:%i search:%i\n',census_size,search_size);
                continue;
          end
          left = imread('im0.png');
          right =imread('im1.png');
          
          
      
        fprintf(' Considered case of census:%i search:%i\n',census_size,search_size);
		fprintf('Performing census Transform of size:%i*%i...\n',census_size,census_size);
    
		left=census(left,census_size);
		right=census(right,census_size);
%         		save('censusRealValues.mat','left');

%         left = IP_tool(left,census_size);
%         right = IP_tool(right,census_size);

     
        
        out_filenameLeft = sprintf('%iX%i_LeftCensus.png',census_size,census_size);
        out_filenameRight = sprintf('%iX%i_RightCensus.png',census_size,census_size);

		imwrite(mat2gray(left), out_filenameLeft );
		imwrite(mat2gray(right), out_filenameRight);
    

		% ====================================
		%        Basic Block Matching
		% ====================================


		fprintf('Performing basic block matching...\n');

		% Start a timer.
		tic();

		% Convert the images from RGB to grayscale by averaging the three color 
		% channels.
		leftI = mean(left, 3);
		rightI = mean(right, 3);
		
		disparityRange =255;

		% Define the size of the blocks for block matching.
		halfBlockSize = (search_size-1)/2;
		%blockSize = 2 * halfBlockSize + 1;
		blockSize=search_size;

		% Get the image dimensions.
		[imgHeight, imgWidth] = size(leftI);
     %   disparity_map = zeros( 1986,2962,256, 'uint8');
		disparity_map = zeros(size(leftI), 'single');
		% For each row 'm' of pixels in the image...
		for (m = 1 : imgHeight)
			
		% Set min/max row bounds for the template and blocks.
		% e.g., for the first row, minr = 1 and maxr = 4
		minr = max(1, m - halfBlockSize);
		maxr = min(imgHeight, m + halfBlockSize);

		% For each column 'n' of pixels in the image...
		for (n = 1 : imgWidth)
			
			% Set the min/max column bounds for the template.
			% e.g., for the first column, minc = 1 and maxc = 4
			minc = max(1, n - halfBlockSize);
			maxc = min(imgWidth, n + halfBlockSize);
			
			 %  mind = max(-disparityRange, 1 - minc);
			mind = 0; 
			maxd = min(disparityRange, imgWidth - maxc);

			% Select the block from the right image to use as the template.
			template = rightI(minr:maxr, minc:maxc);
			
			% Get the number of blocks in this search.
			numBlocks = maxd - mind + 1;
			
			% Create a vector to hold the block differences.
			blockDiffs = zeros(numBlocks, 1);
		 % motor cycle icin
			% Calculate the difference between the template and each of the blocks.
			for (i = mind : maxd)
			
				% Select the block from the left image at the distance 'i'.
				block = leftI(minr:maxr, (minc + i):(maxc + i));
			
				% Compute the 1-based index of this block into the 'blockDiffs' vector.
				blockIndex = i - mind + 1;
			
				% Take the sum of absolute differences (SAD) between the template
				% and the block and store the resulting value.
% 				blockDiffs(blockIndex, 1) = sum(sum(abs(template - block)));
                xor_tmp = bitxor(template,block);
                
                blockDiffs(blockIndex, 1) = sum(de2bi(xor_tmp));    
              %  disparity_map(m,n,blockIndex)= blockDiffs(blockIndex, 1);
			end
			
			% Sort the SAD values to find the closest match (smallest difference).
			% Discard the sorted vector (the "~" notation), we just want the list
			% of indices.
			[temp, sortedIndeces] = sort(blockDiffs);
			
			% Get the 1-based index of the closest-matching block.
			bestMatchIndex = sortedIndeces(1, 1);
			
			% Convert the 1-based index of this block back into an offset.
			% This is the final disparity value produced by basic block matching.
			d = bestMatchIndex + mind - 1;	
			% Calculate a sub-pixel estimate of the disparity by interpolating.
			% Sub-pixel estimation requires a block to the left and right, so we 
			% skip it if the best matching block is at either edge of the search
			% window.
			
				% Skip sub-pixel estimation and store the initial disparity value.
				disparity_map(m, n) = d;
			
				
			
			
		end

		% Update progress every 10th row.
		if (mod(m, 10) == 0)
			fprintf('  Image row %d / %d (%.0f%%)\n', m, imgHeight, (m / imgHeight) * 100);
		end
			
		end

		% Display compute time.
		elapsed = toc();
		fprintf('Calculating disparity map took %.2f min.\n', elapsed / 60.0);

		% =========================================
		%        Visualize Disparity Map
		% =========================================

		%fprintf('Displaying disparity map...\n');

		%figure(1);
		%imagesc(disparity_map);
		%title('blockmatching result');
		fname = sprintf('DisparityCosts%d x %d.mat', census_size,search_size);
		save(fname,'disparity_map');
        disparity_map_normalized=mat2gray(disparity_map);
    %    fname = sprintf('DisparityCosts_normalized%d x %d.mat', census_size,search_size);
	%	save(fname,'disparity_map_normalized');


% 		% read ground truth disparity image
		dispG=pfmread('disp1GT.pfm');
%       %  dispG=imread('disp5.png');
% 		%figure(2);
% 		%imagesc(dispG);
% 		%title('Ground Truth Image');
% 
% 
		[error1, error2,error5]=disparityEr(dispG,uint8(disparity_map));
		fprintf('Performing Median Filter...\n');
% 
		md=median_filter(disparity_map);
		md_name= sprintf('FiltredDisparityMap%d x %d.mat', census_size,search_size);
		save(md_name,'md');
% 
% 		%figure(3);
% 		%imagesc(md);
		%title('Post-Proccessing blockmatching result');
% 
		[errorm1, errorm2,errorm5]=disparityEr(dispG,uint8(md));
% 
		fprintf('error for Simple block matching Before MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n',...
		error1*100,error2*100,error5*100);
% 
		fprintf('error for Simple block matching After MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n',...
		errorm1*100,errorm2*100,errorm5*100);
% 
% 
% 
% 
		out_textfile = fopen('ErrorResults.txt','a');
%       
		fprintf(out_textfile,'For census_size of:%i x %i and Search size of: %i x %i\n Error is:Before MF  threshold >=1: %f\n threshold >=2:%f\n threshold >=5:%f\n\n',...
 		census_size,census_size,search_size,search_size,error1*100,error2*100,error5*100);
 		fprintf(out_textfile,'For census_size of:%i x %i and Search size of: %ix%i\n Error is:After MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n\n',...
         census_size,census_size,search_size,search_size,errorm1*100,errorm2*100,errorm5*100);    
 		fprintf(out_textfile,'\n\n\n');
 		out_filenameBeforeMD = sprintf( '%i x %i BeforeMedianF.png',census_size,search_size);
         out_filenameAfterMD = sprintf('%i x %i AfterMedianF.png',census_size,search_size);
% 
		imwrite(uint8(disparity_map),out_filenameBeforeMD);
 		imwrite(uint8(md),out_filenameAfterMD );
         test=test+1;
        fprintf('test of number %i is done!\n',test);
	end
end
