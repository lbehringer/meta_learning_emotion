"""First argument is the yaml file, second argument the audio file
example command: python run_pavoque_segmentation.py pavoque-angry.yaml pavoque-angry.flac
"""


import convert_pavoque
import sys

yaml_file = sys.argv[1]
audio_file = sys.argv[2]
transcript_file = convert_pavoque.get_pavoque_transcript(yaml_file)
convert_pavoque.segment_pavoque_audio(transcript_file, audio_file)
