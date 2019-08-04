set BCHSIZE=20
@rem set EPOCH=10
@rem set EPOCH=100
set EPOCH=1000
@rem set EPOCH=1
set PARA=8
set GPU=0
set TLIST=../data\train_list_20180223.txt
set VLIST=../data\train_list_20180223.txt
set OUT=result_w10_20180223
set RESUME=result_w10\snapshot_latest

python saikinTrain.py %TLIST% %VLIST% -o=%OUT% -B=%BCHSIZE% -b=10 -e=%EPOCH% -g=%GPU% -j=%PARA%
@rem python saikinTrain.py %TLIST% %VLIST% -o=%OUT% -B=%BCHSIZE% -b=10 -e=%EPOCH% -g=%GPU% -j=%PARA% --resume=%RESUME%


exit /b
