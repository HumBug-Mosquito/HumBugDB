import librosa
import soundfile as sf
import pandas as pd
import os
import math

def write_audio_for_df(data_csv_path, dir_in, dir_out, min_duration=0):
    '''Extract wave files from dataframe df. 
    The sample rate at this stage of the processing is preserved. The written output writes to a single folder, dir
    out, with each label producing its own wave file named after its unique label id. This allows simple cross-referencing
    back to the database. Provide the option to skip labels which are shorter than min_duration (in seconds)'''
    # The written output preserves the folder structure
    # that is kept in the database path field upon writing, but replaces the root directory dir_in with dir_out. Each
    # filename is kept, but the clip number is appended to the file, giving one unique file per label.'''
    df = pd.read_csv(data_csv_path)
    n_files_out = 0
    n_files_skipped = 0
    
    for row_idx_series in df.iterrows():
        row = row_idx_series[1]
        signal, rate = librosa.load(os.path.join(dir_in, row['path'][1:]),
                      offset = row['fine_start_time'],
                      duration = row['fine_end_time'] - row['fine_start_time'], sr = None)
        if row['fine_end_time'] - row['fine_start_time'] > min_duration:
            # output_file = os.path.join(dir_data_root, row['path']).replace('.wav', '_clip_' + str(i) + '.wav'))
            output_file = os.path.join(dir_out, str(row['id']) + '.wav') 
            n_files_out+=1
            sf.write(output_file, signal, rate, 'PCM_24') # Writes to PCM wave format for faster i/o for further processing 
            if n_files_out % 100 == 0:
                print('Files successfully written:', n_files_out, ' out of', len(df))
        else:
            n_files_skipped+=1  
    print('Files successfully written:', n_files_out, '. Files skipped due to min_duration:', n_files_skipped)