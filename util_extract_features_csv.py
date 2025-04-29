from extractor import Extractor
import os
import csv
import math
"""
This program is an example implimentation of the extractor 
and writes out a csv file with named with the parameters below
"""

frame_size = 512
hop_size = math.floor(frame_size/2)
sample_rate = 22500
audio_folder = './mp3corpus'
out_folder = 'mp3Output'



def main():
    
    first_time = True
    s1 = Extractor(sample_rate, frame_size, hop_size)
    for filename in os.listdir(audio_folder):
        
        path = audio_folder + '/' + filename
        windowData = s1.extract(path)

        # Write out the features to csv
        windowData['file_name'] = filename
        print(filename)
        fname = 'features_'+str(sample_rate)+'Hz_'+str(frame_size)+'.csv'
        if first_time:
            with open(fname, 'w', newline='') as csvfile:
                fieldnames = windowData.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(windowData)
            first_time = False
        else:
            with open(fname, 'a', newline='') as csvfile:
                fieldnames = windowData.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(windowData)
        windowData = None




if __name__ == '__main__':
    main()
