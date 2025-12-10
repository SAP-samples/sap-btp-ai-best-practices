"""
Migrate files from App_prototype to main directories
"""
import shutil
from pathlib import Path

# Define paths
base_path = Path(r"c:\Users\I344088\_Python_\_BTP_AI_COE_\AI CORE Video template")
prototype_path = base_path / "App_prototype"
video_src = prototype_path / "Video"
audio_src = prototype_path / "Audio"
video_dst = base_path / "Video"
audio_dst = base_path / "Audio"

# Create destination directories
video_dst.mkdir(exist_ok=True)
audio_dst.mkdir(exist_ok=True)

print("Starting file migration...")
print(f"Source: {prototype_path}")
print(f"Destination: {base_path}")
print()

# Copy video files
video_count = 0
if video_src.exists():
    for video_file in video_src.glob("*.mp4"):
        dst_file = video_dst / video_file.name
        if not dst_file.exists():
            print(f"Copying video: {video_file.name}")
            shutil.copy2(video_file, dst_file)
            video_count += 1
        else:
            print(f"Skipping (exists): {video_file.name}")

    for video_file in video_src.glob("*.avi"):
        dst_file = video_dst / video_file.name
        if not dst_file.exists():
            print(f"Copying video: {video_file.name}")
            shutil.copy2(video_file, dst_file)
            video_count += 1

    for video_file in video_src.glob("*.mov"):
        dst_file = video_dst / video_file.name
        if not dst_file.exists():
            print(f"Copying video: {video_file.name}")
            shutil.copy2(video_file, dst_file)
            video_count += 1

# Copy audio files
audio_count = 0
if audio_src.exists():
    for audio_file in audio_src.glob("*.mp3"):
        dst_file = audio_dst / audio_file.name
        if not dst_file.exists():
            print(f"Copying audio: {audio_file.name}")
            shutil.copy2(audio_file, dst_file)
            audio_count += 1
        else:
            print(f"Skipping (exists): {audio_file.name}")

    for audio_file in audio_src.glob("*.wav"):
        dst_file = audio_dst / audio_file.name
        if not dst_file.exists():
            print(f"Copying audio: {audio_file.name}")
            shutil.copy2(audio_file, dst_file)
            audio_count += 1

    for audio_file in audio_src.glob("*.ogg"):
        dst_file = audio_dst / audio_file.name
        if not dst_file.exists():
            print(f"Copying audio: {audio_file.name}")
            shutil.copy2(audio_file, dst_file)
            audio_count += 1

    for audio_file in audio_src.glob("*.flac"):
        dst_file = audio_dst / audio_file.name
        if not dst_file.exists():
            print(f"Copying audio: {audio_file.name}")
            shutil.copy2(audio_file, dst_file)
            audio_count += 1

print()
print("=" * 50)
print(f"Migration complete!")
print(f"Copied {video_count} video files to Video/")
print(f"Copied {audio_count} audio files to Audio/")
print("=" * 50)
print()
print("You can now safely delete the App_prototype directory.")
