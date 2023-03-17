#/bin/sh

make StereoMatching
make StereoMatching_

./StereoMatching ../images/Art/view1.png ../images/Art/view5.png SAD 5 100
./StereoMatching_ ../images/Art/view1.png ../images/Art/view5.png SAD 5 100
