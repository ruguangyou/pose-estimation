# # Before run this script,
# # images should be put in ./cfsd-imageset/left/ and ./cfsd-imageset/right
# cd ./cfsd-imageset/left/
# rename 's/\.jpg$/_left\.jpg/' *.jpg
# # rename 's/\.png$/_left\.png/' *.png
# mv * ..
# cd ../right/
# rename 's/\.jpg$/_right\.jpg/' *.jpg
# # rename 's/\.png$/_right\.png/' *.png
# mv * ..
# cd ..
# rm -r left right
# cd ..

# # Create imagelist.yml
# cd ./imagelist-creator/build/
# ./imagelist_creator ../../imagelist.yml ../../cfsd-imageset/*.jpg
# # ./imagelist_creator ../../imagelist.yml ../../cfsd-imageset/*.png
# cd ../../

# Run stereo calibration
cd ./stereo-calib/build/
./stereo_calib -w=9 -h=6 -s=0.025 ../../imagelist.yml 
mv intrinsics.yml ../../
mv extrinsics.yml ../../
