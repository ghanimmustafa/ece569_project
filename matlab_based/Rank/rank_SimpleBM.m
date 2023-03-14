clc
clear all
target_folder = '../dataset/Room';
image_folder_name = extractAfter(target_folder,'dataset/');

disparityRange = 255;
% read ground truth disparity image
dispG = double(imread(strcat(target_folder,'/disp0.pgm')));
dispG = uint8(dispG);
%dispG= pfmread(strcat(target_folder,'/disp1.pfm'));
test = 1;
for rank_size= 25
	for search_size= 25
        


        % Read Input images
        left =  uint8(imread(strcat(target_folder,'/left.ppm')));%imread('im0.png');
        right =  uint8(imread(strcat(target_folder,'/right.ppm')));%imread('im1.png');  
          
        fprintf(strcat('processing the image: ',image_folder_name));
        fprintf('\n');
        fprintf(' considered case of  rank:%i search:%i\n',rank_size,search_size);
		fprintf('performing Rank  transform of size:%i*%i...\n',rank_size,rank_size);

		left=rankTransform(left,rank_size);
		right=rankTransform(right,rank_size);
        out_filenameLeft = sprintf(strcat(image_folder_name,'_%iX%i_left_rank_image.png'),rank_size,rank_size);
        out_filenameRight = sprintf(strcat(image_folder_name,'_%iX%i_right_rank_image.png'),rank_size,rank_size);

		imwrite(mat2gray(left),out_filenameLeft);
		imwrite(mat2gray(right), out_filenameRight);


		% ====================================
		%        Basic Block Matching
		% ====================================


		fprintf('Performing basic block matching...\n');

		% Start a timer.
		tic();

		% Convert the images from RGB to grayscale by averaging the three color 
		% channels.
		leftI = mean(left, 3); % does not change anything if rank is already applied
		rightI = mean(right, 3); % does not change anything if rank is already applied
		disparity_map = zeros(size(leftI), 'uint8');

		% Define the size of the blocks for block matching.
		halfBlockSize = (search_size-1)/2;
		%blockSize = 2 * halfBlockSize + 1;
		blockSize=search_size;

		% Get the image dimensions.
		[imgHeight, imgWidth] = size(leftI);

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
			
			mind = max(0, 1 - minc);
			%mind = 0; 
			maxd = min(disparityRange, imgWidth - maxc);

			% Select the block from the right image to use as the template.
			template = rightI(minr:maxr, minc:maxc);
			
			% Get the number of blocks in this search.
			numBlocks = maxd - mind + 1;
			
			% Create a vector to hold the block differences.
			blockDiffs = zeros(numBlocks, 1);
			
			% Calculate the difference between the template and each of the blocks.
			for (i = mind : maxd)
			
				% Select the block from the left image at the distance 'i'.
				block = leftI(minr:maxr, (minc + i):(maxc + i));
			
				% Compute the 1-based index of this block into the 'blockDiffs' vector.
				blockIndex = i - mind + 1;
			
				% Take the sum of absolute differences (SAD) between the template
				% and the block and store the resulting value.
				blockDiffs(blockIndex, 1) = sum(sum(abs(template - block)));
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

		figure(1);
		imagesc(disparity_map);
		title('blockmatching result');
	%	fname = sprintf(strcat(image_folder_name,'rank_DisparityMap_%dx%d.mat'), rank_size,search_size); %sprintf('CRT_DisparityMap%d x %d.mat', rank_size,search_size);
	%	save(fname,'disparity_map');



   
		[error1, error2,error5]=disparityEr(dispG,disparity_map);
		fprintf('Performing Median Filter...\n');

		filtered_DM=median_filter(disparity_map);
		figure(2);
		imagesc(disparity_map);
		title('filtered blockmatching result');        
	%	md_name= sprintf(strcat(image_folder_name,'rank_DisparityMap_MF_%dx%d.mat'), rank_size,search_size);
	%	save(md_name,'filtered_DM');


		[errorm1, errorm2,errorm5]=disparityEr(dispG,filtered_DM);

		fprintf('error for Simple block matching Before MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n',...
		error1*100,error2*100,error5*100);

		fprintf('error for Simple block matching After MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n',...
		errorm1*100,errorm2*100,errorm5*100);

		%fname = sprintf(strcat(image_folder_name,'rank_DisparityMap_%dx%d.mat'), rank_size,search_size); %sprintf('CRT_DisparityMap%d x %d.mat', rank_size,search_size);
		%save(fname,'disparity_map');



		out_textfile = fopen(strcat(image_folder_name,'_rank_ErrorResults.txt'),'a');
      
		fprintf(out_textfile,'For rank size of:%i x %i and Search size of: %i x %i\n Error is:Before MF  threshold >=1: %f\n threshold >=2:%f\n threshold >=5:%f\n\n',...
		rank_size,rank_size,search_size,search_size,error1*100,error2*100,error5*100);
		fprintf(out_textfile,'For rank size of:%i x %i and Search size of: %ix%i\n Error is:After MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n\n',...
        rank_size,rank_size,search_size,search_size,errorm1*100,errorm2*100,errorm5*100);    
		fprintf(out_textfile,'\n\n\n');
        
		out_filenameBeforeMD = sprintf(strcat(image_folder_name,'_%ix%i_rank_DispImg.png'),rank_size,search_size);
        out_filenameAfterMD = sprintf(strcat(image_folder_name,'_%ix%i_rank_DispImg_MF.png'),rank_size,search_size);
		imwrite(uint8(rescale(disparity_map,0,255)),out_filenameBeforeMD);
		imwrite(uint8(rescale(filtered_DM,0,255)),out_filenameAfterMD );
        test=test+1;
        fprintf('test of number %i is done!\n',test);
	end
end
fprintf(out_textfile,'Used Disparity Range = %i',disparityRange);

fclose(out_textfile);

system('sudo chown -R mustafaghanim ./'); % do not worry about this.

