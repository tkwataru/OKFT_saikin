set OUT=./Segmentation_w10_20180223
set MODEL=result_w10_20180223\model_latest
@rem set MODEL=result_w10\model_epoch-0990
set GPU=0
@rem set GPU=-1
set VLIST=../data\train_list_20180223.txt

python inferenceSaikin.py %VLIST% -R=%OUT% -m=%MODEL% -g=%GPU%
@rem python inferenceIHMseg.py %VLIST% -R=%OUT% -m=%MODEL% -g=%GPU% -d=1

