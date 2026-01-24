import sys
import os
from pathlib import Path

def convert_with_pydub(input_file, output_file):
    """Convert using pydub (requires FFmpeg)"""
    try:
        from pydub import AudioSegment
        
        print(f"Loading {input_file}...")
        audio = AudioSegment.from_file(input_file)
        
        print(f"Converting to WAV...")
        audio.export(output_file, format='wav')
        
        print(f"✅ Successfully converted!")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Size:   {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        return True
        
    except ImportError:
        print("❌ pydub not installed. Installing...")
        os.system("pip install pydub")
        return False
    except Exception as e:
        print(f"❌ Error with pydub: {e}")
        return False

def convert_with_soundfile(input_file, output_file):
    """Convert using soundfile and librosa"""
    try:
        import soundfile as sf
        import librosa
        
        print(f"Loading {input_file} with librosa...")
        audio, sr = librosa.load(input_file, sr=None, mono=False)
        
        print(f"Saving as WAV...")
        sf.write(output_file, audio.T if audio.ndim > 1 else audio, sr)
        
        print(f"✅ Successfully converted!")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Sample Rate: {sr} Hz")
        print(f"   Size:   {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error with librosa/soundfile: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_audio.py <input_file> [output_file]")
        print("\nExample:")
        print("  python convert_audio.py recording.m4a")
        print("  python convert_audio.py recording.m4a output.wav")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Generate output filename
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.wav'))
    
    print("="*50)
    print("Audio Format Converter")
    print("="*50)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()
    
    # Check if output file already exists
    if os.path.exists(output_file):
        response = input(f"⚠️  {output_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Check FFmpeg
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("⚠️  FFmpeg not detected. Some formats may not work.")
        print("   For M4A files, please install FFmpeg first.")
        print("   See INSTALL_FFMPEG.md for instructions.\n")
    

    print("Attempting conversion...")
    
    # Method 1: Try librosa/soundfile first (works without FFmpeg for some formats)
    if convert_with_soundfile(input_file, output_file):
        return
    
    # Method 2: Try pydub (requires FFmpeg for M4A)
    if has_ffmpeg:
        if convert_with_pydub(input_file, output_file):
            return
    
    # If all methods failed
    print("\n" + "="*50)
    print("Conversion failed with all methods.")
    print("\nAlternative options:")
    print("1. Install FFmpeg (see INSTALL_FFMPEG.md)")
    print("2. Use online converter:")
    print("   - https://cloudconvert.com/m4a-to-wav")
    print("   - https://convertio.co/m4a-wav/")
    print("3. Use audio editing software (Audacity, etc.)")
    print("="*50)
    sys.exit(1)

if __name__ == "__main__":
    main()
