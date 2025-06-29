# Impor library'
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import queue
import time 

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Buat queue untuk event deteksi
event_queue = queue.Queue()

# Muat model YOLOv8
try:
    model = YOLO('best.pt')
    print("Model 'best.pt' berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

def generate_frames():
    """Membuka webcam, melakukan deteksi, dan menaruh event di queue dengan logika waktu."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Kamera tidak dapat diakses.")
        return

    fire_detected_time = None
    fire_alarm_triggered = False

    print("Kamera berhasil diakses. Memulai streaming...")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()

            detected_classes = results[0].boxes.cls
            is_fire_detected_now = 0 in detected_classes 
            
            if is_fire_detected_now:
                # Jika api terdeteksi 
                if fire_detected_time is None:
                    fire_detected_time = time.time()
                    print("Api terdeteksi pertama kali, memulai timer...")
                else:
                    # Jika api masih terus terdeteksi, cek durasinya
                    duration = time.time() - fire_detected_time
                    print(f"Api masih terdeteksi, durasi: {duration:.2f} detik")
                    # Jika durasi sudah lebih dari 5 detik DAN alarm belum berbunyi
                    if duration >= 5 and not fire_alarm_triggered:
                        print("ALARM API DIAKTIFKAN! (Durasi > 5 detik)")
                        event_queue.put('fire')
                        fire_alarm_triggered = True # Set penanda agar alarm tidak berbunyi terus-menerus
            else:
                # Jika api TIDAK terdeteksi di frame ini, reset semuanya
                if fire_detected_time is not None:
                    print("Api tidak lagi terdeteksi, timer direset.")
                fire_detected_time = None
                fire_alarm_triggered = False

            if 1 in detected_classes and not is_fire_detected_now:
                event_queue.put('smoke')

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_events')
def detection_events():
    def event_stream():
        while True:
            event_type = event_queue.get()
            yield f"data: {event_type}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)