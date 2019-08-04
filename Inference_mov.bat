set OUT=./Segmentation_w10_20180223
set MODEL=result_w10_20180223\model_latest
@rem set MODEL=result\model_epoch-0010
@rem set MODEL=result\model_epoch-0100
@rem set MODEL=result\model_epoch-1000
set GPU=0
@rem set GPU=-1
@rem set VIDEO=../data\E-coli\E-coli-1.mp4
@rem set VIDEO=../data\E-coli\E-coli-2.mp4
@rem set VIDEO=../data\E-coli\E-coli-3.mp4
set VIDEO=../data\E-coli\E-coli-4.mp4

python inferenceSaikin_mov.py %VIDEO% -R=%OUT% -m=%MODEL% -g=%GPU%
@rem python inferenceIHMseg.py %VIDEO% -R=%OUT% -m=%MODEL% -g=%GPU% -d=1

