wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
mv SegmentationClassAug.zip ./VOCdevkit/VOC2012
cd ./VOCdevkit/VOC2012
unzip SegmentationClassAug.zip
rm SegmentationClassAug.zip
cd ImageSets
cd Segmentation
rm *.txt
wget https://raw.githubusercontent.com/xmojiao/deeplab_v2/master/voc2012/list/train_aug.txt
wget https://raw.githubusercontent.com/xmojiao/deeplab_v2/master/voc2012/list/train.txt
wget https://raw.githubusercontent.com/xmojiao/deeplab_v2/master/voc2012/list/val.txt
