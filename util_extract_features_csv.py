from extractor import Extractor
import os
import csv

"""
This program is an example implimentation of the segmenter. 
We pull segment information then create a label file for audacity to visualize the background/foreground assignments.
"""

window_duration = 1
sample_rate = 22500
frame_size = 1024
hop_size = 512
audio_folder = './mp3corpus'
out_folder = 'mp3Output'

def main():
    s1 = Extractor(window_duration, sample_rate, frame_size, hop_size)
    data = []
    for filename in os.listdir(audio_folder):
        path = audio_folder + '/' + filename
        windowData = s1.extract(path)
        
        # print('NEW FILE')
        # print(windowData)
        windowData['file_name'] = filename
        data.append(windowData)

    with open('features_1024_1sec.csv', 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)



if __name__ == '__main__':
    main()
