

device_set=(0 1);
device_order=(0 1);

num_device=${#device_set[@]};
template_list="templates.txt";
template_folder="Templates/";
image_name="1413-2015-09-04T17-03-33 PNG/1413-2015-09-04T17-03-33_%05d.png";
result_filename="result/1413/result_%05d.txt";
begin_image_num=0;
end_image_num=100; #114048;

for d in $(seq 0 $(($num_device-1)));
do
	./c1413_notbb ${device_set[$d]} ${device_order[$d]} $num_device $template_list $template_folder "$image_name" $result_filename $begin_image_num $end_image_num &
done

wait
echo "complete"
