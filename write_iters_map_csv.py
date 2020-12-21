from multiprocessing import Pool
import glob
import os
import csv

def run():
    map10 = []
    iters = []
    for name in glob.glob("*.h5"):
        nlist = name.split("_")
        
        map10.append(nlist[0])
        temp = nlist[2].split(".")
        iters.append(temp[0])

    print(map10)
    print(iters)

    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(map10, iters))
        
run()
