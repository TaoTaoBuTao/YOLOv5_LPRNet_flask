#!/usr/bin/env python
# coding=utf-8

from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices

cameras = list_video_devices()
print(dict(cameras))
# return: {0: 'Intel(R) RealSense(TM) 3D Camera (Front F200) RGB', 1: 'NewTek NDI Video', 2: 'Intel(R) RealSense(TM) 3D Camera Virtual Driver', 3: 'Intel(R) RealSense(TM) 3D Camera (Front F200) Depth', 4: 'OBS-Camera', 5: 'OBS-Camera2', 6: 'OBS-Camera3', 7: 'OBS-Camera4', 8: 'OBS Virtual Camera'}

audios = list_audio_devices()
print(dict(audios))
# return:  {0: '麦克风阵列 (Creative VF0800)', 1: 'OBS-Audio', 2: '线路 (NewTek NDI Audio)'}
