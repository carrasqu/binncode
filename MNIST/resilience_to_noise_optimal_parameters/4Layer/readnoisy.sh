
lay=4


noise='0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0'
#!/bin/bash

for k in $noise
do
touch noisy_histogram_entangled_${lay}layers_p_${k}.txt ; rm noisy_histogram_entangled_${lay}layers_p_${k}.txt; touch noisy_histogram_entangled_${lay}layers_p_${k}.txt

touch noisy_samples_entangled_${lay}layers_p_${k}.txt; rm noisy_samples_entangled_${lay}layers_p_${k}.txt; touch noisy_samples_entangled_${lay}layers_p_${k}.txt

 for (( i=1; i<=10; i++ ))
 do

 cd $i

 paste noisy_histogram_p_$k.txt >> ../noisy_histogram_entangled_${lay}layers_p_${k}.txt
 
 for (( m=1; m<=10; m++ ))
 do
 paste noisy_loss_samples${m}_p_$k.txt >> ../noisy_samples_entangled_${lay}layers_p_${k}.txt 
 done

 cd ../

 done 
done
