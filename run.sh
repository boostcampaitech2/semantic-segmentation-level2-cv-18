#!/bin/bash

TRAIN="train.py"
INFER="inference.py"
YAML="default.yaml"
ALL=($TRAIN $COMPARE $YAML)

C_BD="\e[1m"
C_R="\e[31m"
C_Y="\e[33m"
C_G="\e[32m"
C_B="\e[34m"
C_P="\e[35m"
C_RS="\e[0m"

M_FILE="All files exist. Continue..."
M_CT=" files missing! Continue? (Y, [ENTER]/N): "
M_ER="Typing error! Please type Y, [ENTER] or N: "
M_NUM="Choose number (0-9): "
M_NUM_ER="Typing error! Please type 0 - 9: "
M_ST=("(1): train\n" "(2): train_pseudo\n" "(3): compare\n" \
		"(4): inference\n" "(5): inference_tta\n" \
		"(6): train + inference\n" "(7): train + inference_tta\n" \
		"(8): train + train_pseudo + inference_tta\n" \
		"(9): Reproduce the final result\n" "(0): exit\n")

CNT=0
CONTINUE=0
EXIT=0

# set -xv

dot ()
{
	for NUM in {1..45}; do
		echo -en '-'
		sleep 0.002
	done
	echo -e "-\n"
}

interface ()
{
	echo -e "$C_BD	      [ SETTINGS ]$C_RS"
	echo -e " ${M_ST[@]}"
}

phase ()
{
	if test -e $1; then
		echo -e "${C_G}Start '$1'!$C_RS"
		echo -e "${C_G}Finish '$1'!$C_RS"
	else
		echo -e "${C_R}No '$1'!$C_RS"
	fi
}

echo -e "$C_BD\n	< Object Segmentation >$C_RS"

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
		if [ "$YN" = "Y" ] || [ -z "$YN" ]; then
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
	elif [[ "$FLAG" -gt "0" ]] && [[ "$FLAG" -le "9" ]]; then
		break
	fi
	echo -en "$C_Y$M_NUM_ER$C_RS"
done

dot

echo -e "$C_BD ${M_ST[$FLAG-1]}$C_RS"
if [ "$FLAG" -le "5" ]; then
	phase ${ALL[$FLAG-1]}
elif [ "$FLAG" -le "7" ]; then
	phase ${ALL[0]}
	dot
	phase ${ALL[$FLAG-3]}
else
	phase ${ALL[0]}
	dot
	phase ${ALL[1]}
	dot
	phase ${ALL[4]}
fi


