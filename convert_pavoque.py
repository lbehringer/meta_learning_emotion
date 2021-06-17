import re
import numpy as np
import soundfile as sf

"""convert yaml into lines that contain only start (in 1/100s, 6 digits), stop (in 1/100s, 6 digits), and text
--> works for files up to 9999s long, i.e. 166min
return the filename of the file that was written"""
def get_pavoque_transcript(yaml):
    start = str()
    end = str()
    text = str()
    transcript_filename = 'transcript_' + yaml.replace(".yaml", ".txt")
    # with open(transcript_filename, 'r+', encoding='utf-8') as clear_file:
    #     clear_file.truncate(0)
    with open(yaml, 'r', encoding='utf-8') as f:
        yaml_lines = f.readlines()
        for line in yaml_lines:
            if re.search("^\s*text", line):
                text = line.lstrip("  text: ")
            if re.search("^\s*start", line):
                start = line.lstrip("  start: ")
                start = int(float(start)*100)
                start = str(start).zfill(6)
            if re.search("^\s*end", line):
                end = line.lstrip("  end: ")
                end = int(float(end)*100)
                end = str(end).zfill(6)
                if start != "" and end != "" and text != "":
                    transcript_line = start + "_" + end + "\t" + text
                    # print(transcript_line)
                    with open(transcript_filename, 'a', encoding='utf-8') as out_file:
                        out_file.write(transcript_line)
                start, end, text = str(), str(), str()
        return transcript_filename


"""takes as input a transcript file and the audio file; 
sf.read the audio into data and samplerate, then iterate over the transcript lines and in each line apply 
create_segment """
def segment_pavoque_audio(transcript, in_file):
    data, samplerate = sf.read(in_file)

    with open(transcript, 'r', encoding='utf-8') as f:
        for line in f:
            start = int(re.findall("\d{6}", line)[0])
            stop = int(re.findall("\d{6}", line)[1])
            create_segment(data, samplerate, start, stop)


"""takes as input data and samplerate (which were both read by sf.read(audio_file), start time (in 1/100s), 
and stop time (in 1/100s);
converts time into frames (e.g., 2.34s will be taken as input 234 --> /100 = 2.34, then * samplerate to get frame
outputs audio file with the following name format: <starttime_stoptime.fileformat>, e.g. <000234_000567.wav>"""
def create_segment(data, samplerate, start, stop):
    startframe = int(start/100*samplerate)
    stopframe = int(stop/100*samplerate)
    data = data[startframe:stopframe]
    out_file = str(start).zfill(6) + "_" + str(stop).zfill(6) + ".wav"
    sf.write(out_file, data, samplerate)