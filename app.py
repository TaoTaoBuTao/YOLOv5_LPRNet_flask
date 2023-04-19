from importlib import import_module
import os
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

app = Flask(__name__)

NAME = ""
FILE_FLAG = False
CAMERA_FLAG = False


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def video_gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_start')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if FILE_FLAG:
        return Response(video_gen(Camera(NAME, True)),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif CAMERA_FLAG:
        print('video_start')
        return Response(video_gen(Camera("0", False)),  # 选择你的摄像头ID
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(mimetype='multipart/x-mixed-replace; boundary=frame')
        # pass


@app.route('/video', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    upload_path = './static/uploads'
    if not os.path.exists(upload_path):
        os.mkdir(upload_path)
    upload_file_path = os.path.join(basepath, upload_path, (f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    f.save(upload_file_path)
    global NAME, FILE_FLAG, CAMERA_FLAG
    NAME = upload_file_path
    FILE_FLAG = True
    CAMERA_FLAG = False
    return redirect(url_for('index'))


@app.route('/camera_stop', methods=['POST'])
def stop():
    # return Response(mimetype='multipart/x-mixed-replace; boundary=frame')
    global CAMERA_FLAG, FILE_FLAG
    CAMERA_FLAG = False
    FILE_FLAG = False
    # return Response(mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(video_gen(Camera("1", False)),  # 选择你的摄像头ID
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/download', methods=['POST'])
def download():
    print('下载')
    file_path = 'inference/output'
    file_name = os.path.basename(NAME)
    print(os.path.join(file_path, file_name))
    # if NAME == "0":
    #     return send_from_directory(file_path, 'camera_prediction.avi', as_attachment=True)
    # else:
    # return send_from_directory(file_path, file_name, as_attachment=True)
    return send_from_directory(file_path, os.listdir(file_path)[0], as_attachment=True)


@app.route('/camera', methods=['POST'])
def camera_get():
    global CAMERA_FLAG, FILE_FLAG
    CAMERA_FLAG = True
    FILE_FLAG = False
    return redirect(url_for('index'))
    # return redirect('/')


if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True, port=5001)
