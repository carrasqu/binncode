#!/bin/bash


#touch histogram_entangled_${lay}layers.txt; rm histogram_entangled_${lay}layers.txt; touch histogram_entangled_${lay}layers.txt


#!/bin/bash

noise='0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0'

for (( i=1; i<=10; i++ ))
do  
   cd $i
   
   for j in $noise
   do 

      cp ../loadparamsevaluate.jl .
      echo $j | julia loadparamsevaluate.jl 

   done
   cd ../

done 
