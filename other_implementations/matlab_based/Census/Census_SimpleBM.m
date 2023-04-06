
path_desktop = 'C:\Users\pc\Google Drive\OZU\Image_Depth_Research\spring2020\dataset';
path_laptop = 'E:\My Google Drive\OZU\Image_Depth_Research\spring2020\dataset';
% E:\My Google Drive\OZU\Image_Depth_Research\spring2020\dataset
dataset_path = path_desktop


target_folder1 = strcat(dataset_path,'\Adirondack-perfect');
target_folder2 = strcat(dataset_path,'\Jadeplant-perfect');
target_folder3 = strcat(dataset_path,'\Motorcycle-perfect');
target_folder4 = strcat(dataset_path,'\Piano-perfect');
target_folder5 = strcat(dataset_path,'\Pipes-perfect');
target_folder6 = strcat(dataset_path,'\Playroom-perfect');
target_folder7 = strcat(dataset_path,'\Playtable-perfect');
target_folder8 = strcat(dataset_path,'\Recycle-perfect');
target_folder9 = strcat(dataset_path,'\Shelves-perfect');
target_folder10 = strcat(dataset_path,'\Vintage-perfect');

% Select the image folder(s):
for i = [9,10]
target_folder = eval(strcat('target_folder',int2str(i)))    
image_folder_name = target_folder(length(dataset_path) + 2:end);
test = 1;
preproc_name =['CRT','Rank','Census'];
disparityRange = 255;
% read ground truth disparity image
dispG= pfmread(strcat(target_folder,'/disp1.pfm'));

for rank_size= 7
	for search_size= 1
        

         % Read input images 
        left =  imread(strcat(target_folder,'/im0.png'));%imread('im0.png');
        right =  imread(strcat(target_folder,'/im1.png'));%imread('im1.png'); 
          
        fprintf(strcat('Processing the Image: ',image_folder_name));
        fprintf('\n');
        fprintf(' Considered Case of Census:%i search:%i\n',rank_size,search_size);
		fprintf('Performing Complete Census Transform of size:%i*%i...\n',rank_size,rank_size);

		left=census(left,rank_size);
		right=census(right,rank_size);
        out_filenameLeft = sprintf(strcat(image_folder_name,'_%iX%i_left_Census_image.png'),rank_size,rank_size);
        out_filenameRight = sprintf(strcat(image_folder_name,'_%iX%i_right_Census_image.png'),rank_size,rank_size);

		%imwrite(mat2gray(left),out_filenameLeft);
		%imwrite(mat2gray(right), out_filenameRight);


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
		disparity_map = zeros(size(leftI), 'single');
		

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
			
			 %  mind = max(-disparityRange, 1 - minc);
			mind = 0; 
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
                xor_tmp = bitxor(template,block);
				% Compute the 1-based index of this block into the 'blockDiffs' vector.
				blockIndex = i - mind + 1;
			
				% Take the Hamming distances between the template
				% and the block and store the resulting value.
                blockDiffs(blockIndex, 1) = sum(my_d2b(xor_tmp));
                
				%blockDiffs(blockIndex, 1) = sum(sum(abs(template - block)));
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

		%figure(1);
		%imagesc(disparity_map);
		%title('blockmatching result');
		fname = sprintf(strcat(image_folder_name,'Census_DisparityMap_%dx%d.mat'), rank_size,search_size); %sprintf('CRT_DisparityMap%d x %d.mat', rank_size,search_size);
		save(fname,'disparity_map');





   
		[error1, error2,error5]=disparityEr(dispG,disparity_map);
		fprintf('Performing Median Filter...\n');

		md=median_filter(disparity_map);
		md_name= sprintf(strcat(image_folder_name,'Census_DisparityMap_MF_%dx%d.mat'), rank_size,search_size);
		save(md_name,'md');


		[errorm1, errorm2,errorm5]=disparityEr(dispG,md);

		fprintf('error for Simple block matching Before MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n',...
		error1*100,error2*100,error5*100);

		fprintf('error for Simple block matching After MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n',...
		errorm1*100,errorm2*100,errorm5*100);




		out_textfile = fopen(strcat(image_folder_name,'_Census_ErrorResults.txt'),'a');
        
		fprintf(out_textfile,'For Census size of:%i x %i and Search size of: %i x %i\n Error is:Before MF  threshold >=1: %f\n threshold >=2:%f\n threshold >=5:%f\n\n',...
		rank_size,rank_size,search_size,search_size,error1*100,error2*100,error5*100);
		fprintf(out_textfile,'For Census size of:%i x %i and Search size of: %ix%i\n Error is:After MF  threshold >=1: %f\n and threshold >=2:%f\n and threshold >=5:%f\n\n',...
        rank_size,rank_size,search_size,search_size,errorm1*100,errorm2*100,errorm5*100);    
		fprintf(out_textfile,'\n\n\n');
        
		out_filenameBeforeMD = sprintf(strcat(image_folder_name,'_%ix%i_Census_DispImg.png'),rank_size,search_size);
        out_filenameAfterMD = sprintf(strcat(image_folder_name,'_%ix%i_Census_DispImg_MF.png'),rank_size,search_size);
		imwrite(uint8(rescale(disparity_map,0,255)),out_filenameBeforeMD);
		imwrite(uint8(rescale(md,0,255)),out_filenameAfterMD );
        test=test+1;
        fprintf('test of number %i is done!\n',test);
	end
end
fprintf(out_textfile,'Used Disparity Range = %i',disparityRange);
fclose(out_textfile);
end
