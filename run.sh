

#device_set=(0);
#device_order=(0);
color=("red" "green" "yellow" "blue");

num_device=${#device_set[@]};
template_list="data/templates.txt";
template_folder="data/Templates/";
image_name="data/1413-2015-09-04T17-03-33PNG/1413-2015-09-04T17-03-33_%05d.png";
result_filename="results/1413/result_%05d.txt";
#begin_image_num=0;
end_image_num=4; #114048;


#for d in $(seq 0 $(($num_device-1)));
#do
./aceMatchTemplate 0 999 $num_device $template_list $template_folder "$image_name" $result_filename 0 $end_image_num ${color[$d]} &
./aceMatchTemplate 1 999 $num_device $template_list $template_folder "$image_name" $result_filename 1024 $end_image_num ${color[$d]} &
#done

wait
echo "complete"
