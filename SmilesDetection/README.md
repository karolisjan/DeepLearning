# Smiles Detection

End-to-end CNN-based detection of smiles in a video stream in real-time.

<p  align='center'>
    <img href='Result' src='media/rock.gif'></img>
</p>

## Usage

[detect.py](detect.py) - script for detecting smiles in video files or streams in real-time

```
usage: detect.py [-h] -c CASCADE_PATH -m MODEL_PATH [-v VIDEO_PATH]
                 [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -c CASCADE_PATH, --cascade_path CASCADE_PATH
                        path to the face haar cascade
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to a trained CNN smile model
  -v VIDEO_PATH, --video_path VIDEO_PATH
                        path to a video file (optional)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to save the output to (optional)
```

## Data 

> Daniel Hromada. SMILEsmileD. https://github.com/hromi/SMILEsmileD

