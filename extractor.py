from essentia_engine import EssentiaEngine
import numpy as np
from essentia.standard import MonoLoader, FrameGenerator, PoolAggregator
import essentia
import math

debug = True 

class Extractor:
    """
    The segmenter uses the Essentia engine to extract features from an audio file.
    The input to the segmenter is a path to an audio file and the output is segment information. 
    """

    # initialize
    def __init__(self, dur, sr, fs, hs):

        self.window_duration = dur # analysis window length in seconds
        self.sample_rate = sr  # sample rate
        self.frame_size = fs  # samples in each frame
        self.hop_size = hs
        self.window_size = int(self.sample_rate * self.window_duration)
        self.adjusted_window = (self.window_size // self.frame_size) * self.frame_size

        # the essentia engine make sure that the features were extracted under the same conditions as the training data
        self.engine = EssentiaEngine(
            self.sample_rate, self.frame_size, self.hop_size)

    # run the segmentation
    def extract(self, afile):
        # extract the regions
        segments = self.extract_regions(afile)


        return segments 

    # use the bf classifier to extract background, foreground, bafoground regions
    # returns # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    def extract_regions(self, afile):

        # instantiate the loading algorithm
        loader = MonoLoader(filename=afile, sampleRate=self.sample_rate)
        # perform the loading
        audio = loader()

        # create pool for storage and aggregation
        pool = essentia.Pool()

        # frame counter used to detect end of window
        accumulator = {}
        windowCount = 0

        # calculate the length of analysis frames
        frame_duration = float(self.frame_size / 2)/float(self.sample_rate)

        # number frames in a window
        numFrames_window = int(self.window_duration / frame_duration)
        if debug:
            print(numFrames_window, ' frames in a window')
            print('frame duration: ', frame_duration)
            print('audio len: ', len(audio))
            print('number frames total: ', len(audio)/self.frame_size)
            print('window size: ', self.window_size)
            print('frame adjusted window size: ', self.adjusted_window)

        # translate type naming convention from csv to database

        processed = []  # storage for the classified segments
        windowCount = 0
        for window in FrameGenerator(audio, frameSize=self.adjusted_window, hopSize=self.adjusted_window, startFromZero=True, lastFrameToEndOfFile=True):
            # extract all features
            pool = self.engine.extractor(window)
            aggrigated_pool = PoolAggregator(defaultStats=['mean', 'stdev', 'skew', 'dmean', 'dvar', 'dmean2', 'dvar2'])(pool)

            # compute mean and variance of the frames using the pool aggregator, assign to dict in same order as training
            # narrow everything down to select features
            features_dict = {}
            descriptor_names = aggrigated_pool.descriptorNames()

            # unpack features in lists
            for descriptor in descriptor_names:
                # little to no values in these features, ignore
                if('tonal' in descriptor or 'rhythm' in descriptor):
                    continue
                value = aggrigated_pool[descriptor]
                # unpack arrays
                if (str(type(value)) == "<class 'numpy.ndarray'>"):
                    for idx, subVal in enumerate(value):
                        features_dict[descriptor + '.' + str(idx)] = subVal
                    continue
                # ignore strings
                elif(isinstance(value, str)):
                    pass
                # add singular values
                else:
                    features_dict[descriptor] = value

            # reset counter and clear pool
            pool.clear()
            aggrigated_pool.clear()

            # prepare dictionary for filtering
            vector = np.array(list(features_dict.values()))
            fnames = np.array(list(features_dict.keys()))

            # remove NAN values, this can happen on segments of short length
            vector = np.nan_to_num(vector)

            # create clean dictionary for the database
            features_filtered = {}
            for idx, val in enumerate(vector):
                features_filtered[fnames[idx]] = val
            for key in features_filtered.keys():
                if key not in accumulator.keys():
                    accumulator[key] = features_filtered[key]
                else:
                    accumulator[key] += features_filtered[key]

            windowCount += 1

            #processed.append({'features': features_filtered})
        output = self.avg_dict_items(accumulator, windowCount)
        return output

    

    def finalize_regions(self, processed):
        region_data = []
        num = len(processed)
        for i in processed:
            temp = {}
            temp['features'] = self.avg_dict_items(i['features'], num)
            region_data.append(temp)
        return region_data

    def avg_dict_items(self, D, a):
        result = {}
        for key in D.keys():
            D[key]/a
            result[key] = D[key]/a
        return result

    def sum_feature_dicts(self, Da, Db):
        result = {}
        for key in Da.keys():
            A = Da[key]
            B = Db[key]
            result[key] = A+B  # [a + b for (a,b) in zip(A,B)]
        return result
