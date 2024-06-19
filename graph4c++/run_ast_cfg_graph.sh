#python getCFG.py --rootpath /ext0/knhdu/CodeRAG/data/Filtered_/codes/ --astpath ./astpath/ --cfgpath ./cfgpath/
python getCFG.py --rootpath /ext0/knhdu/CodeRAG/data/transcode/codes/ --astpath ./astpath/ --cfgpath ./cfgpath/

python call_graphGen.py --writepath ./graphs/ --astpath ./astpath/ --cfgpath ./cfgpath/ --picky 0