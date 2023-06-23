import wave
import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.stats import mode
import warnings
import os
from sklearn.cluster import SpectralClustering, KMeans
from itertools import combinations
import math
from bpm import beat_detection

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

BILLBOARD100_2021 = ['levitating feat dababy','save your tears remix','blinding lights','mood','good 4 u', 'kiss me more', 'leave the door open', 'drivers license',
'montero', 'peaches', 'butter', 'stay', 'deja vu', 'positions', 'bad habits', 'heat waves', 'without you', 'forever after all',
'go crazy', 'astronaut in the ocean', '34+35', 'what you know bout love', "my ex's best friend", "industry baby", "therefore i am",
"up", "fancy like", "dakiti", "best friend", "rapstar", "heartbreak anniversary", "for the night", "calling my phone", "beautiful mistakes",
"holy", "on me", "you broke me first", "traitor", "back in blood", "i hope", "dynamite", "wockesha", "you right", "beat box",
"laugh now cry later", "need to know", "wants and needs","way 2 sexy","telepatia","whoopty","lemonade","good days","starting over",
"body","willow","bang!","better together","you're mines still",'every chance i get','essence','chasing after you', 'the good ones',
'leave before you love me','glad you exist','lonely','beggin','streets','whats next','famous friends','lil bit','thot shit','late at night',
'kings & queens','anyone','track star','time today','cry baby','all i want for christmas is you','no more parties',"what's your country song",
'one too many', 'arcade', 'yonaguni', 'good time', "if i didn't love you", 'knife talk', 'pov', 'just the way', 'take my breath',
"we're good", 'hell of a view', "rockin' around the christmas tree", 'put your records on', 'happier than ever', 'single saturday night',
'things a man oughta know','throat baby', 'tombstone', 'drinkin beer. talkin god. amen.','todo de ti']

BILLBOARD100_2022 = ['heat waves','stay','super gremlin','abcdefu','ghost',"we don't talk about bruno",'enemy','thats what i want','woman',
'easy on me','big energy','bad habits','shivers','cold heart pnau remix','need to know','levitating feat dababy','save your tears remix',
'til you cant','pushin p','one right now','industry baby','what happened to virgil','i hate u','hrs and hrs','sweetest pie','mamiii',
'good 4 u','ahhh ha','light switch','doin this','you right','fingers crossed','buy dirt','bam bam','surface pressure','fancy like',
'blick blick','sand in my boots','love nwantiti','never say never','aa','boyfriend','thinkin with my dick','drunk','beers on me',
'knife talk','broadway girls','shes all i wanna be','numb little bug','nobody like u','the motto','slow down summer','23','peru',
'the family madrigal','handsomer','sometimes','to be loved by you','heart on fire','circles around this town','computer murderers',
'petty too','do we have a problem','to the moon','me or sum','nail tech','flower shops','no interviews','never wanted to be that girl',
'banking on me', 'barbarian', 'beautiful lies', 'bones', 'by your side', 'city of gods', 'closer', 'comeback as a country boy', 'dos orugitas',
'freaky deaky', 'ghost story', 'give heaven some hell', 'golden child', 'half of my hometown', 'high', 'i love you so', "i'm tired", 'idgaf',
'if i was a cowboy', 'maybe', 'money so big', 'over', 'p power', 'pressure', 'rumors', 'she likes it', 'smokin out the window',
'smoking & thinking', "tom's diner", 'waiting on a miracle', 'what else can i do']

SPOTIFY2018 = ['gods plan','shape of you','sad','rockstar','psycho','in my feelings','better now','i like it','one kiss','idgaf','friends','havana','lucid dreams',
'nice for what','girls like you','the middle','all the stars','no tears left to cry','x','moonlight','look alive','these days','te bote remix',
'mine','youngblood','new rules','love lies','meant to be','jocelyn flores','perfect','taste','solo','i fall apart','nevermind',
'echame la culpa','eastside','never be the same','wolves','changes','in my mind','river','dura','sicko mode','thunder','jackie chan','me niego',
'finesse','back to you','let you down','call out my name','ric flair drip','happier','too good at goodbyes','freaky friday','believer',
'fefe','rise','body','sin pijama','xo tour lif3','2002','nonstop','fuck love','in my blood','silence',
'1, 2, 3', 'be alright','candy paint','congratulations','corazon','criminal','dejala que vuelva','downtown',
'dusk till dawn','everybody dies in their nightmares','feel it still','flames','god is a woman',
'him & i', 'humble', 'i like me better', 'i miss you', 'let me go', 'lovely', 'no brainer', 'perfect duet',
'plug walk','pray for me','promises','rewrite the stars','siguelo bailando','stir fry','taki taki',
'this is me','vaina loca','walk it talk it','what lovers do','wolfine','yes indeed','young dumb and broke']

SPOTIFY2016 = ['closer','love yourself','one dance','starboy','hello','panda','hurts so good','cheap thrills','work','i know what you did last summer',
'pillowtalk','sorry','cant stop the feeling','into you','never forget you','wherever i go','in the name of love','treat you better',
'all my friends','shout out to my ex','needed me','cold water','we dont talk anymore','sit still look pretty','roses','starving',
'can i be him','dangerous woman','ophelia','history','hymn for the weekend','this is what you came for','all time low','side to side',
'close','i took a pill in ibiza','dont','let me love you','same old love','this town','when we were young','me, myself & i','good grief',
'work from home','here','send my love (to your new lover)','hide away','exchange','too good','dangerously','7 years',
'1955','alarm','all in my head','cake by the ocean','controlla','cool girl','dont let me down','fast car','fresh eyes','gold',
'hands to myself','heathens','history olivia holt','holy','hotline bling','hotter than hell','i hate u i love u','just hold on',
'just like fire','lost boy','love me now','low life','luv','middle','my house','my way','never be like you','no money','no',
'one call away','papercuts','perfect strangers','pink + white','ride','say it','say it2','sex','sexual','somebody else','sorry beyonce',
'sucker for pain','sweet lovin','tears','the sound','this ones for you','when you love someone','white iverson','you & me']

ROCK_HITS = [
'ocean avenue','hysteria','boulevard of broken dreams','cant stop','what ive done','chop suey','seven nation army','mr brightside',
'kryptonite','sugar were going down','misery business','dani california','holiday','last resort','in the end','i miss you',
'californication','youre gonna go far kid','toxicity', 'dance dance','the middle','the kill','i write sins not tragedies','take a look around',
'bring me to life','savior','numb','best of you','the diary of jane','take me out','welcome to the black parade','i hate everything about you',
'sex on fire','im not okay','like a stone','the pretender','when you were young','face down','its been awhile','island in the sun',
'she hates me','uprising hq','yellow','smooth criminal','american idiot','the anthem','thnks fr th mmrs','paralyzer','all my life',
'the reason','no one knows','in too deep','supermassive black hole','first date','youth of the nation','rollin','complicated',
'gives you hell','fake it','joker and the thief','miss murder','higher','im just a kid','beverly hills',
'dear maria count me in','use somebody','through glass','wish you were here','last nite','steady as she goes','are you gonna be my girl',
'want you bad','welcome home','reptilia','hate to say i told you so','jerk it out','i bet you look good on the dancefloor','float on',
'your touch','a punk','you know youre right','rock n roll train','obstacle 1','banquet','hey there delilah','little sister','change in the house of flies',
'dig','woman','the bitter end','judith','there there','the taste of ink','everyday is exactly the same','cute without the e','fell in love with a girl',
'stop crying your heart out','beautiful day','how you remind me'
]

def get_all_filenames(dir):
    '''
    Produces a list of all of the filenames in a director
    '''
    lst = []
    for filename in os.listdir(dir):
        lst.append(dir + "/" + filename)
    return lst

def get_actual_bpm_giant_steps(song):
    '''
    Produces the actual BPM of a song from the Giant Steps Dataset by reading from
    Giant Steps Annotations
    '''
    sng = song.split('/')
    filename = sng[-1]
    filename = filename.split('.')
    filename = filename[0] + '.' + filename[1]
    with open(f"giantsteps-tempo-dataset-master/annotations/tempo/{filename}.bpm", "r") as f:
        l = f.readline()
        l = l.strip()
    return float(l)

def remove_outliers(data,deviations=2):
    '''
    Given an array of numbers (data), removes any element outside of inputted # of deviations of the mean.
    '''
    data = np.array(data)
    avg = np.mean(data)
    stdev = np.std(data)
    diff_from_avg = np.abs(data - avg)
    indices_to_keep = np.where(diff_from_avg <= (deviations * stdev))
    return data[indices_to_keep]

def exists_cluster_pair(beats, n_clusters):
    '''
    beats: dict mapping cluster number to avg tempo in that cluster
    n_clusters: total # of clusters
    '''
    ground_list = list(range(n_clusters))
    pairs = list(combinations(ground_list, 2))
    best_pairs = []
    lowest_diff = math.inf
    for i, j in pairs:
        diff_1 = abs(beats[i] * 2 - beats[j])
        diff_2 = abs(beats[j] * 2 - beats[i])
        min_diff = min([diff_1,diff_2])
        if min_diff <= 1 and min_diff <= lowest_diff:
            best_pairs.append((i,j))
            lowest_diff = min_diff
    return best_pairs

def pick_cluster(beats, stdevs, n_clusters, clusters):
    '''
    Cluster picking algorithm.
    beats: array of best beats (ALREADY CONVERTED TO BPM)
    stdevs: array of best stdevs, indexed same as beats
    n_clusters: number of clusters
    clusters: SpectralClustering object with n_clusters number of clusters
    '''
    beats = np.array(beats)
    stdevs = np.array(stdevs)
    clusters = np.array(clusters.labels_) #array where index i tells you which cluster beat window i falls in
    avg_beats = np.zeros(n_clusters)
    avg_stds = np.zeros(n_clusters) #dict to store cluster number to avg stdev
    cluster_sizes = np.zeros(n_clusters) #dict to store cluster number to cluster size
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_sizes[i] = len(cluster_indices)
        avg_stds[i] = np.mean(stdevs[cluster_indices])
        avg_beats[i] = np.mean(beats[cluster_indices])
    pair_check = exists_cluster_pair(avg_beats, n_clusters)
    if np.argmin(avg_stds) == np.argmax(cluster_sizes) and cluster_sizes.max() > 5: #ideal scenario where the cluster that minimizes the stdev is the largest
        print("used clustering 1")
        arr = remove_outliers(beats[np.where(clusters == np.argmax(cluster_sizes))[0]], 1)
        return arr.mean()
    elif pair_check:
        print("used clustering 2")
        total = set()
        for i, j in pair_check:
            total.add(i)
            total.add(j)
        biggest_cand = -math.inf
        to_output = -1
        ## cluster size heuristic
        for elem in total:
            if cluster_sizes[elem] > biggest_cand:
                to_output = elem
                biggest_cand = cluster_sizes[elem]
        ## stdev minimize heuristic
        # smallest_cand = math.inf
        # to_output = -1
        # for elem in total:
        #     if avg_stds[elem] < smallest_cand:
        #         to_output = elem
        #         smallest_cand = cluster_sizes[elem]

        return avg_beats[to_output]
    else:
        return -1

def beat_detection_sc(arg,dist_input=43, n_clusters = 4, std_cutoff = 25, window_size=5):
    '''
    Given a file name of a .wav file and a minimum distance between peaks, returns an estimated BPM of the song.
    '''
    #if len(arg) != 2:
    #   print('Usage: python3 bpm.py filename')
    #   return
    #arg=arg[1]
    #filepath = f'Audio Files/Rock Hits/{arg}.wav'
    warnings.filterwarnings('ignore')
    filepath = arg

    #Number of bytes to read from the file at once (should be a power of 2)
    CHUNK = 1024

    #Opening file using wave module
    wf = wave.open(filepath, 'rb')
    channels = wf.getnchannels()
    # if channels != 2:
    #     print('Error: This program is designed for stereo audio, not mono or music with more than 2 channels.')
    #     return 0

    sampling_rate = wf.getframerate()
    if sampling_rate not in [44100,48000]:
        print('Error: This program is made for music with 44100Hz or 48000Hz sampling rate.')
        return 0

    #Audio Range Information (In Hz) source: https://www.cuidevices.com/blog/understanding-audio-frequency-range-in-audio-design
    audio_ranges = {
        'sub-bass': [16,60],
        'bass': [60,250],
        'lower midrange': [250,500],
        'midrange': [500,2000],
        'higher midrange': [2000,4000],
        'presence': [4000,6000],
        'brilliance': [6000,20000]
    }

    #Setting up Numpy array of frequencies indices
    frequencies = np.arange(CHUNK/2) * sampling_rate / CHUNK
    sub_bass_indices = np.where((frequencies > audio_ranges['sub-bass'][0]) & (frequencies < audio_ranges['sub-bass'][1]))
    bass_indices = np.where((frequencies > audio_ranges['bass'][0]) & (frequencies < audio_ranges['bass'][1]))
    lower_midrange_indices = np.where((frequencies > audio_ranges['lower midrange'][0]) & (frequencies < audio_ranges['lower midrange'][1]))
    midrange_indices = np.where((frequencies > audio_ranges['midrange'][0]) & (frequencies < audio_ranges['midrange'][1]))
    higher_midrange_indices = np.where((frequencies > audio_ranges['higher midrange'][0]) & (frequencies < audio_ranges['higher midrange'][1]))
    presence_indices = np.where((frequencies > audio_ranges['presence'][0]) & (frequencies < audio_ranges['presence'][1]))
    brilliance_indices = np.where((frequencies > audio_ranges['brilliance'][0]) & (frequencies < audio_ranges['brilliance'][1]))

    #Setting up output dict
    output = {
        'left sub-bass': [],
        'left bass': [],
        'left lower midrange': [],
        'left midrange': [],
        'left higher midrange': [],
        'left presence': [],
        'left brilliance': [],
        'right sub-bass': [],
        'right bass': [],
        'right lower midrange': [],
        'right midrange': [],
        'right higher midrange': [],
        'right presence': [],
        'right brilliance': [],
    }

    # Start reading through file using Wave module
    data = wf.readframes(CHUNK)

    #Initialize counter for counting how many times
    counter=0

    #Array of the first half of the data because we want to throw out the second half of the FFT output
    first_half = np.arange(0,CHUNK/2,dtype='int')

    while len(data) > 0:
        #Reads a chunk of data
        data = wf.readframes(CHUNK)
        #Converts data into an array of integers
        raw = np.frombuffer(data, dtype='<i2')
        #Left channel is every other element of the array starting from the 0th index
        #Right channel is every other element of the array starting from the 1st index
        if channels == 2:
            left = raw[::2]
            right = raw[1::2]
        elif channels == 1:
            left = raw[:]
            right = raw[:]
        else:
            print("error: can only do stereo and mono")
            return -1
        #Calculates the Fourier coefficients of the left channel
        left_hat = np.fft.fft(left,CHUNK)
        right_hat = np.fft.fft(right,1024)
        #Calculates the magnitudes of each complex number to get Power Spectral Density
        left_mags = ( np.conj(left_hat) * left_hat ) / 1024
        right_mags = ( np.conj(right_hat) * right_hat ) / 1024
        #Just keeps first half of the data because FFT also returns negative coefficients
        left_mags = left_mags[first_half]
        right_mags = right_mags[first_half]
        #Takes average of the power level within each frequency group
        left_sub_bass = np.mean(left_mags[sub_bass_indices])
        output['left sub-bass'].append(left_sub_bass)
        left_bass = np.mean(left_mags[bass_indices])
        output['left bass'].append(left_bass)
        left_lower_midrange = np.mean(left_mags[lower_midrange_indices])
        output['left lower midrange'].append(left_lower_midrange)
        left_midrange = np.mean(left_mags[midrange_indices])
        output['left midrange'].append(left_midrange)
        left_higher_midrange = np.mean(left_mags[higher_midrange_indices])
        output['left higher midrange'].append(left_higher_midrange)
        left_presence = np.mean(left_mags[presence_indices])
        output['left presence'].append(left_presence)
        left_brilliance = np.mean(left_mags[brilliance_indices])
        output['left brilliance'].append(left_brilliance)
        right_sub_bass = np.mean(right_mags[sub_bass_indices])
        output['right sub-bass'].append(right_sub_bass)
        right_bass = np.mean(right_mags[bass_indices])
        output['right bass'].append(right_bass)
        right_lower_midrange = np.mean(right_mags[lower_midrange_indices])
        output['right lower midrange'].append(right_lower_midrange)
        right_midrange = np.mean(right_mags[midrange_indices])
        output['right midrange'].append(right_midrange)
        right_higher_midrange = np.mean(right_mags[higher_midrange_indices])
        output['right higher midrange'].append(right_higher_midrange)
        right_presence = np.mean(right_mags[presence_indices])
        output['right presence'].append(right_presence)
        right_brilliance = np.mean(right_mags[brilliance_indices])
        output['right brilliance'].append(right_brilliance)
        counter+=1

    #Exporting Data to Pandas DataFrame for ease of plotting
    for key,value in output.items():
        output[key] = np.array(value,dtype='float')
    export = pd.DataFrame.from_dict(output)
    #Converting from samples to seconds using the sampling rate
    song_length = (CHUNK * counter) / sampling_rate
    time_step = np.arange(0,song_length,song_length/(counter))
    #For some song lengths, have to remove the last element of the time step
    try:
        export = export.set_index(time_step)
    except ValueError:
        time_step = np.array(time_step[:-1])
        export = export.set_index(time_step)
    export = export.rename_axis('time')

    #Plotting Frequency Data on MatplotLib
    #dist is the minimum distance between peaks
    dist=dist_input
    #Window size is the size of the moving window of the peaks
    #window_size = 5
    #Master differences will be array of the average distance between the peaks in each moving window
    master_differences = []
    #Master standard deviation will be array of the standard deviation of the distance between the peaks in each moving window
    master_std_dev = []

    #Setting up plot information
    fig,axs = plt.subplots(1,1)
    # fig.suptitle(arg + ' left channel', fontsize=16)

    # plt.sca(axs[0])
    #Plotting the power level of left channel sub-bass against time
    # plt.plot(export.index,export['left sub-bass'],color='red')
    #Calculating the envelope of the sub-bass and plotting it against time
    analytic_signal = hilbert(export['left sub-bass'])
    amplitude_envelope = np.abs(analytic_signal)
    sub_bass_diffs = []
    sub_bass_windows = []
    sub_bass_stds = []
    #Finding peaks in amplitude envelope using SciPy find_peaks function, returns the indexes of the peaks
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    #Moving window looping through the peaks
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        #Finds difference between the peaks and appends to master arrays
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        sub_bass_windows.append(window)
        sub_bass_diffs.append(avg_diff)
        sub_bass_stds.append(std_dev)
    sub_bass_stds = pd.Series(sub_bass_stds)
    good_indices = np.array(sub_bass_stds.nsmallest(int(0.1 * len(sub_bass_stds))).index,dtype='int')
    sub_bass_diffs = np.array(sub_bass_diffs)
    sub_bass_output = sub_bass_diffs[good_indices]
    #Plots peaks as x's on the graph
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange')
    # plt.plot(export.index,amplitude_envelope)
    # axs[0].title.set_text('Left Sub-Bass')

    # plt.sca(axs[1])
    # plt.plot(export.index,export['left bass'],color='red')
    analytic_signal = hilbert(export['left bass'])
    amplitude_envelope = np.abs(analytic_signal)
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    bass_diffs = []
    bass_windows = []
    bass_stds = []
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        bass_windows.append(window)
        bass_diffs.append(avg_diff)
        bass_stds.append(std_dev)
    bass_stds = pd.Series(bass_stds)
    good_indices = np.array(bass_stds.nsmallest(int(0.1 * len(bass_stds))).index,dtype='int')
    bass_diffs = np.array(bass_diffs)
    bass_output = bass_diffs[good_indices]
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange')
    # plt.plot(export.index,amplitude_envelope)
    # axs[1].title.set_text('Left Bass')

    # plt.sca(axs[2])
    # plt.plot(export.index,export['left lower midrange'],color='red')
    analytic_signal = hilbert(export['left lower midrange'])
    amplitude_envelope = np.abs(analytic_signal)
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    lower_mid_diffs = []
    lower_mid_windows = []
    lower_mid_stds = []
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        lower_mid_windows.append(window)
        lower_mid_diffs.append(avg_diff)
        lower_mid_stds.append(std_dev)
    lower_mid_stds = pd.Series(lower_mid_stds)
    good_indices = np.array(lower_mid_stds.nsmallest(int(0.1 * len(lower_mid_stds))).index,dtype='int')
    lower_mid_diffs = np.array(lower_mid_diffs)
    lower_mid_output = lower_mid_diffs[good_indices]
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange')
    # plt.plot(export.index,amplitude_envelope)
    # axs[2].title.set_text('Left Lower Midrange')

    # plt.sca(axs[3])
    # plt.plot(export.index,export['left midrange'],color='red')
    analytic_signal = hilbert(export['left midrange'])
    amplitude_envelope = np.abs(analytic_signal)
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    mid_diffs = []
    mid_windows = []
    mid_stds = []
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        mid_windows.append(window)
        mid_diffs.append(avg_diff)
        mid_stds.append(std_dev)
    mid_stds = pd.Series(mid_stds)
    good_indices = np.array(mid_stds.nsmallest(int(0.1 * len(mid_stds))).index,dtype='int')
    mid_diffs = np.array(mid_diffs)
    mid_output = mid_diffs[good_indices]
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange')
    # plt.plot(export.index,amplitude_envelope)
    # axs[3].title.set_text('Left Midrange')

    # plt.sca(axs[4])
    # plt.plot(export.index,export['left higher midrange'],color='red')
    analytic_signal = hilbert(export['left higher midrange'])
    amplitude_envelope = np.abs(analytic_signal)
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    higher_mid_diffs = []
    higher_mid_windows = []
    higher_mid_stds = []
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        higher_mid_windows.append(window)
        higher_mid_diffs.append(avg_diff)
        higher_mid_stds.append(std_dev)
    higher_mid_stds = pd.Series(higher_mid_stds)
    good_indices = np.array(higher_mid_stds.nsmallest(int(0.1 * len(higher_mid_stds))).index,dtype='int')
    higher_mid_diffs = np.array(higher_mid_diffs)
    higher_mid_output = higher_mid_diffs[good_indices]
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange')
    # plt.plot(export.index,amplitude_envelope)
    # axs[4].title.set_text('Left Higher Midrange')

    # plt.sca(axs[5])
    # plt.plot(export.index,export['left presence'],color='red')
    analytic_signal = hilbert(export['left presence'])
    amplitude_envelope = np.abs(analytic_signal)
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    presence_diffs = []
    presence_windows = []
    presence_stds = []
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        presence_windows.append(window)
        presence_diffs.append(avg_diff)
        presence_stds.append(std_dev)
    presence_stds = pd.Series(presence_stds)
    good_indices = np.array(presence_stds.nsmallest(int(0.1 * len(presence_stds))).index,dtype='int')
    presence_diffs = np.array(presence_diffs)
    presence_output = presence_diffs[good_indices]
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange')
    # plt.plot(export.index,amplitude_envelope)
    # axs[5].title.set_text('Left Presence')

    # plt.sca(axs[6])
    #plt.plot(export.index,export['left brilliance'],color='red',label='Power Level')
    analytic_signal = hilbert(export['left brilliance'])
    amplitude_envelope = np.abs(analytic_signal)
    brilliance_diffs = []
    brilliance_windows = []
    brilliance_stds = []
    peaks = find_peaks(np.array(amplitude_envelope),distance=dist,height=np.mean(amplitude_envelope))[0]
    for i in np.arange(0,len(peaks) - window_size):
        window = np.array(peaks[i:i+window_size+1])
        differences = np.diff(window)
        avg_diff = np.mean(differences)
        std_dev = np.std(differences)
        master_differences.append(avg_diff)
        master_std_dev.append(std_dev)
        brilliance_windows.append(window)
        brilliance_diffs.append(avg_diff)
        brilliance_stds.append(std_dev)
    brilliance_stds = pd.Series(brilliance_stds)
    good_indices = np.array(brilliance_stds.nsmallest(int(0.1 * len(brilliance_stds))).index,dtype='int')
    brilliance_diffs = np.array(brilliance_diffs)
    brilliance_output = brilliance_diffs[good_indices]
    # plt.plot(export.index[peaks],amplitude_envelope[peaks],'x',color='orange',label='Peaks')
    # plt.plot(export.index,amplitude_envelope,label="Envelope")
    axs.title.set_text('Left Brilliance')

    #plt.xlabel('Time (s)')
    #plt.ylabel('Power Level (V^2)/Hz')
    #plt.legend()
    #plt.show()

    #Prints average diffs with the lowest standard deviations within each audio group (This is not used in the final result and is just to see the data)
    # print('sub_bass',sub_bass_output)
    # print('bass',bass_output)
    # print('lower mid',lower_mid_output)
    # print('mid',mid_output)
    # print('higher mid',higher_mid_output)
    # print('presence',presence_output)
    # print('brilliance',brilliance_output)

    #Converting master diff and master std dev into numpy arrays
    master_differences = np.array(master_differences)
    #print(list(master_differences))
    #return (60*2) / (np.mean(master_differences) * (song_length/counter))
    master_std_dev = np.array(master_std_dev)
    #print(list(master_std_dev))
    #Removing windows where the averge distance is exactly equal to the minimum distance, as these are not likely to be beats
    not_dist_indices = np.where(master_differences != dist)[0]
    master_differences = master_differences[not_dist_indices]
    master_std_dev = master_std_dev[not_dist_indices]
    master_std_dev = pd.Series(master_std_dev)

    #While loop reduces the results down to just the <std_cutoff> windows with the smallest standard deviation (and thus the most regularity between the peaks)
    indices_length = 0
    cutoff = 0.001
    while indices_length < std_cutoff:
        good_indices = np.array(master_std_dev.nsmallest(int(cutoff * len(master_std_dev))).index,dtype='int')
        indices_length = len(good_indices)
        cutoff+=0.001
    best_beats = master_differences[good_indices]
    best_beats = (60 / (np.array(best_beats) * (song_length/counter)))
    best_stds = master_std_dev[good_indices]
    sc = SpectralClustering(n_clusters=n_clusters).fit(best_beats.reshape(-1,1))
    n_clusters = len(np.unique(np.array(sc.labels_))) # in case the model uses less than n_clusters clusters (if we have a lot of really good guesses)
    guess = pick_cluster(best_beats, best_stds, n_clusters, sc) #returns -1 if clustering fails
    if guess != -1:
        return guess * 2
    else:
        return beat_detection(arg, dist_input)

if __name__ == '__main__':
    #Removes the Numpy Complex number warning
    warnings.filterwarnings('ignore')
    print(beat_detection(argv[1],44))
