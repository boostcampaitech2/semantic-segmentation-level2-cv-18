# prepare dataset
wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000078/data/data.zip
mv data.zip rawdata.zip
unzip rawdata.zip
rm rawdata.zip

# prepare libraries
pip install pytorch_toolbelt
pip install madgrad

python pipeline.py --cfg-yaml ./config/default.yaml
