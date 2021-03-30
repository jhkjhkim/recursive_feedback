#!/bin/bash

#for q in {0..6} # dataset (total 7 dataset)
#do
  #for w in {0..3} # model, 0:unet, 1:linknet 2:fpn 3:pspnet
  #do
#    for e in {0..9} # random seed
#    do
#      python recursive_feedback.py $q 0 $e 
#    done
#  done
#done


for i in {1..3} # dataset (total 4 dataset)
do
  #for j in {0..3} # model, 0:unet, 1:linknet 2:fpn 3:pspnet
  #do
  #for k in {0..3} # lambda 0..3 total 4
  #do
  for a in {0..6} # decay 0..6 total 7
  do
    for b in {0..1} # combine method 0..1 total 2
    do
      for c in {0..9} # random seed
      do
        python evaluation.py $i 0 0 $a $b $c
      done
    done
  done
done  
#done
