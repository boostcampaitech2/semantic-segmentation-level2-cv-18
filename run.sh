#!/bin/bash

TRAIN="train.py"
INFER="inference.py"
YAML="./configs/default.yaml"
CSV="./submission/pseudo.csv"
ALL=($TRAIN $INFER $YAML "./src/dataloader.py" "./src/dataset.py" \
		 "./src/transforms.py" "./src/models.py" "./src/losses.py" "./src/schedulers.py" \
		"./src/select.py" "./src/pipeline.py" "./src/utils.py" "./src/make.py" "./src/save.py" \
		"./src/model_hrnet.py")

C_BD="\e[1m"
C_R="\e[31m"
C_Y="\e[33m"
C_G="\e[32m"
C_B="\e[34m"
C_P="\e[35m"
C_RS="\e[0m"

M_FILE="All files exist. Continue..."
M_CT=" files missing! Continue? (Y/N): "
M_ER="Typing error! Please type Y or N: "
ER="Typing error! Please type Y or N or Find: "
M_NUM="Choose number (0-5): "
M_NUM_ER="Typing error! Please type 0 - 5: "
M_ST=("(1): train\n" "(2): train_pseudo\n" "(3): inference\n" \
		"(4): train + inference\n" "(5): train_pseudo + inference\n" \
		"(0): exit\n")

CNT=0
CONTINUE=0
EXIT=0

# set -xv

dot ()
{
	for NUM in {1..55}; do
		echo -en '-'
	done
	echo -e "-\n"
}

interface ()
{
	echo -e "$C_BD			[ SETTINGS ]$C_RS"
	echo -e " ${M_ST[@]}"
}


echo -e "$C_BD\n		< Object Segmentation >$C_RS"

dot

for CHECK in ${ALL[@]}; do
	if test -e $CHECK; then
		echo -e "${C_G}Find '$CHECK'!$C_RS"
	else
		echo -e "${C_R}No '$CHECK'!$C_RS"
		((CNT++))
	fi
done

if [ $CNT == 0 ]; then
	echo -e "\n$C_Y$M_FILE$C_RS"
else
	echo -en "\n$C_Y$CNT$M_CT$C_RS"
	while [ $CONTINUE ]; do
		read YN
		if [ "$YN" = "Y" ]; then
			echo -e "${C_Y}Continue...$C_RS"
			break
		elif [ "$YN" = "N" ]; then
			exit 0
		fi
		echo -en "$C_Y$M_ER$C_RS"
	done
fi

dot
interface

echo -en "$C_Y$M_NUM$C_RS"
while [ $CONTINUE ]; do
	read FLAG
	if [ "$FLAG" = "0" ]; then
		exit 0
	elif [[ "$FLAG" -gt "0" ]] && [[ "$FLAG" -le "5" ]]; then
		break
	fi
	echo -en "$C_Y$M_NUM_ER$C_RS"
done

dot

echo -e "$C_BD ${M_ST[$FLAG-1]}$C_RS"
echo -en "Config file is $C_B$YAML$C_RS. Do you want to change? (Y/N/Find): "
while [ $CONTINUE ]; do
	read YN
	if [ "$YN" = "Y" ]; then
		cp $YAML ./configs/tmp.yaml
		vim ./configs/tmp.yaml
		echo -en "${C_Y}Yaml name: $C_RS"
		read name
		name="./configs/$name"
		mv ./configs/tmp.yaml $name
		YAML=$name
		echo -e "${C_Y}Continue...$C_RS"
		break
	elif [ "$YN" = "Find" ]; then
		ls configs/
		echo -en "${C_Y}Yaml name: $C_RS"
		read name
		name="./configs/$name"
		YAML=$name
		echo -e "${C_Y}Continue...$C_RS"
		break
	elif [ "$YN" = "N" ]; then
		break
	fi
	echo -en "$C_Y$ER$C_RS"
done
echo -e "\nConfig file: ${C_B}$YAML$C_RS"
cat $YAML

echo -e "\n${C_G}Start ${M_ST[$FLAG-1]}$C_RS"
if [ "$FLAG" -eq "1" ]; then
	python $TRAIN --yaml $YAML
elif [ "$FLAG" -eq "2" ]; then
	python $TRAIN --yaml $YAML --pseudo 1 --ps_csv $CSV
elif [ "$FLAG" -eq "3" ]; then
	python $INFER --yaml $YAML
elif [ "$FLAG" -eq "4" ]; then
	python $TRAIN --yaml $YAML
	dot
	python $INFER --yaml $YAML
else
	python $TRAIN --yaml $YAML --pseudo 1 --ps_csv $CSV
	dot
	python $INFER --yaml $YAML
fi
echo -e "${C_G}Finish ${M_ST[$FLAG-1]}$C_RS"


