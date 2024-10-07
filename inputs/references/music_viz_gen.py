
import os
import numpy as np
from glob import glob
import librosa
from moviepy.editor import AudioFileClip, ImageSequenceClip
import cv2

## Util functions
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def load_audio_files(directory_path="audio"):
def load_audio_files(directory_path="test"):
    return glob(f"{directory_path}/*.mp3")

def normalize_and_scale(data, new_min=0, new_max=255):
    assert np.ndim(data) >= 0, f"The object {data} is not array-like."
    try:
        assert np.std(data) != 0, f"Cannot normalize a zero variance array"
    except AssertionError:
        return np.zeros(data.shape)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    scaled_data = new_min + (new_max - new_min) * normalized_data
    return np.uint8(scaled_data)

# TODO assertions
def normalize_and_scale_features(audio_features):
    normalized_features = []
    for feature in audio_features:
        if np.std(feature) != 0:
            normalized = (feature - feature.min()) / (feature.max() - feature.min())
            scaled = np.uint8(255 * normalized)
        else:
            scaled = np.zeros_like(feature, dtype=np.uint8)
        normalized_features.append(scaled)
    return normalized_features

def create_waveform_feature(y, n_points, height=25):
    # Resample the audio signal to n_points
    resampled = np.interp(np.linspace(0, len(y), n_points), np.arange(len(y)), y)
    
    # Normalize the resampled signal to [-1, 1]
    normalized = resampled / np.max(np.abs(resampled))
    
    # Create a 2D array where each column is the waveform
    waveform_feature = np.tile(normalized, (height, 1))
    
    return waveform_feature

## Vid gen functions
def generate_frame(audio_feature_2d, img_idx, img_width=700, img_height=300, 
                   line_mode="mask", stretch_height=False, row_repeat=1):
    assert np.ndim(audio_feature_2d) == 2, "expecting 2D audio features"
    assert not (stretch_height and row_repeat > 1), "stretch height and row repeats are mutually exclusive"
    assert line_mode in [None, "cover", "mask"], "line_mode must be None, 'cover', or 'mask'"
    
    if not stretch_height and row_repeat == 1:
        img_height = audio_feature_2d.shape[0]
    elif row_repeat > 1:
        img_height = audio_feature_2d.shape[0] * row_repeat
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    n_spect_cols = audio_feature_2d.shape[1]
    
    # Calculate the start and end indices for the audio feature slice
    start_idx = max(0, img_idx - img_width // 2)
    end_idx = min(n_spect_cols, start_idx + img_width)
    
    # Create a slice of the audio feature
    feature_slice = audio_feature_2d[:, start_idx:end_idx]
    if row_repeat > 1:
        feature_slice = np.repeat(feature_slice, row_repeat, axis=0)
    else:
        feature_slice = feature_slice
    
    # Resize the audio feature slice vertically to fit the image height and normalize to image
    feature_img_repr = cv2.resize(feature_slice, (feature_slice.shape[1], img_height), interpolation=cv2.INTER_LINEAR)
    # feature_img_repr = normalize_and_scale(feature_slice, 0, 255)
    
    # Calculate the position to place the audio feature slice in the image
    left_padding = max(0, (img_width // 2) - img_idx)
    
    # Place the resized audio feature slice in the image
    img_slice_width = min(img_width - left_padding, feature_img_repr.shape[1])
    img[:, left_padding:left_padding+img_slice_width] = np.repeat(feature_img_repr[:, :img_slice_width, np.newaxis], 3, axis=2)

    # Add green line to indicate current time
    if line_mode:
        line_position = img_width // 2
        line_width = 3
        
        if line_mode == "cover":
            # Draw a solid green line
            cv2.line(img, (line_position, 0), (line_position, img_height), (0, 255, 0), line_width)
        
        elif line_mode == "mask":
            # Create a mask for the green channel
            mask = np.zeros_like(img)
            mask[:, line_position-line_width//2:line_position+line_width//2+1, 1] = 1
            
            # Apply the mask: keep only the green channel where the mask is non-zero
            img[:,:,0] = img[:,:,0] * (1 - mask[:,:,1])
            img[:,:,2] = img[:,:,2] * (1 - mask[:,:,1])

    return img


def create_stacked_video(audio_features, feature_names, y, sr, audio_file, output_file, 
                         fps=45, line_mode="mask", row_repeat=None, separator_width=0):
    assert len(audio_features) > 0, "At least one audio feature is required"
    num_columns = audio_features[0].shape[1]
    assert all(feature.shape[1] == num_columns for feature in audio_features), \
        "All audio features must have the same number of columns (indicating the temporal dimension)"
    
    duration = len(y) / sr
    num_frames = int(duration * fps)
    
    if row_repeat is None:
        row_repeat = [1] * len(audio_features)
    assert len(audio_features) == len(row_repeat), "Number of audio features and row repeats must match"
    
    # Normalize and scale all features before generating frames
    normalized_features = normalize_and_scale_features(audio_features)

    frames = []
    for i in range(num_frames):
        frame_time = i / fps
        frame_idx = int(frame_time * audio_features[0].shape[1] / duration)
        
        # Generate frames for each audio feature
        feature_frames = []
        for j, (feature, repeat) in enumerate(zip(normalized_features, row_repeat)):
            frame = generate_frame(feature, frame_idx, line_mode=line_mode, row_repeat=repeat)
            
            # Add separator before each feature (including the first one)
            if separator_width > 0:
                separator = np.zeros((separator_width, frame.shape[1], 3), dtype=np.uint8)
                feature_frames.append(separator)
            
            feature_frames.append(frame)
        
        # Add separator at the end
        if separator_width > 0:
            separator = np.zeros((separator_width, frame.shape[1], 3), dtype=np.uint8)
            feature_frames.append(separator)
        
        # Stack frames vertically
        stacked_frame = np.vstack(feature_frames)
        frames.append(stacked_frame)

    video_clip = ImageSequenceClip(frames, fps=fps)
    audio_clip = AudioFileClip(audio_file)
    video_clip = video_clip.set_audio(audio_clip)
    create_directory(os.path.dirname(output_file))
    video_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')



## main
audio_files = load_audio_files()
if not audio_files:
    print("No audio files found in the specified directory.")

for audio_file in audio_files:
    y, sr = librosa.load(audio_file)
    
    base_name = os.path.splitext(os.path.basename(audio_file))[0].lower()
    print(f"Audio file base name: {base_name}")
    
    # Compute various audio features and print their shapes
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # delta_mel = librosa.feature.delta(mel_spec_db)
    
    # Compute waveform feature
    n_points = mel_spec.shape[1]  # Match the number of time points in mel spectrogram
    waveform_feature = create_waveform_feature(y, n_points, height=1)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    delta_spectral_contrast = librosa.feature.delta(spectral_contrast)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    delta_spectral_centroid = librosa.feature.delta(spectral_centroid)
    delta_spectral_bandwidth = librosa.feature.delta(spectral_bandwidth)
    delta_spectral_rolloff = librosa.feature.delta(spectral_rolloff)
    delta_zero_crossing_rate = librosa.feature.delta(zero_crossing_rate)
    
    rms = librosa.feature.rms(y=y)
    delta_rms = librosa.feature.delta(rms)
    delta2_rms = librosa.feature.delta(rms, order=2)


    # Prepare audio features and their names for visualization
    audio_features = [
        mel_spec_db, chroma, delta_mfcc, delta2_mfcc, 
        spectral_contrast, delta_spectral_contrast,
        spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, 
        delta_spectral_centroid, delta_spectral_bandwidth, delta_spectral_rolloff, delta_zero_crossing_rate, 
        waveform_feature, rms, delta_rms, delta2_rms
    ]
    
    feature_names = [
        # don't need to track rn
        # "Mel Spectrogram", "Chroma", "Delta MFCC", "Delta2 MFCC", "Spectral Contrast",
        # "Spectral Centroid", "Spectral Bandwidth", "Spectral Rolloff",
        # "Zero Crossing Rate", "RMS", "Delta RMS", "Delta2 RMS"
    ]
    
    # Set row repeat for each feature (adjust as needed)
    row_repeat = [
        1, 6, 6, 6, 
        10, 10, 
        10, 10, 10, 10, 
        10, 10, 10, 10, 
        25, 25, 25, 25
    ]

    output_file = f'vids/{base_name}/output_combined_features_video.mp4'
    if not os.path.exists(output_file):
        create_stacked_video(audio_features, feature_names, y, sr, audio_file, output_file, row_repeat=row_repeat)
        