rm lut RGB_2D RGB_3D

nvcc -o RGB_2D ColorToGray_2D.cu
nvcc -o lut colorToGray_LUT_opti.cu
nvcc -o RGB_3D ColorToGray_3D.cu
nvcc -o 3d_lut 3d_lut.cu

./RGB_2D $1
./lut $1
./RGB_3D $1
./3d_lut $1
