clean : StereoMatching
		rm -rf *.o

StereoMatching : StereoMatching.o main.o
					g++ -Wall main.o StereoMatching.o -o StereoMatching `pkg-config --cflags --libs opencv4`


StereoMatching.o : StereoMatching.cpp StereoMatching.h
					g++ -Wall -c StereoMatching.cpp `pkg-config --cflags --libs opencv4`

main.o : main.cpp StereoMatching.h
			g++ -Wall -c main.cpp `pkg-config --cflags --libs opencv4`


StereoMatching_ : StereoMatching_.o main_.o
					g++ -Wall main_.o StereoMatching_.o -o StereoMatching_ `pkg-config --cflags --libs opencv4`


StereoMatching_.o : StereoMatching_.cpp StereoMatching_.h
					g++ -Wall -c StereoMatching_.cpp `pkg-config --cflags --libs opencv4`

main_.o : main_.cpp StereoMatching_.h
			g++ -Wall -c main_.cpp `pkg-config --cflags --libs opencv4`

