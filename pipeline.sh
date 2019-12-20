bash command_representation.sh > result_representation_50.txt
bash command.sh > result_segmentation.txt
echo "validation" >> result_segmentation.txt
bash command_val.sh >> result_segmentation.txt
