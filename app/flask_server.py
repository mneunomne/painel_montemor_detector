from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
import numpy as np
import cv2

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='static')

FLASK_SERVER_IP="0.0.0.0"
FLASK_SERVER_PORT=3333

socketio = SocketIO(app)

# socketio.run(app)

video_output = None
cropped_output = None
avaraged_output = None
live_output = None
capture_avarage_frames = False

current_segment_index = 0

@app.route('/movement_end/<int:direction>')
def on_movement_end(direction):
    global ready_to_read
    ready_to_read = True
    print('movement end', direction)
    return Response('done', mimetype='text/plain')



if __name__ == '__main__':
    print(f"Starting Flask server on {FLASK_SERVER_IP}:{FLASK_SERVER_PORT}")
    
    # Initialize any required variables
    ready_to_read = False
    
    # Run the Flask-SocketIO app
    socketio.run(
        app, 
        host=FLASK_SERVER_IP, 
        port=FLASK_SERVER_PORT, 
        debug=True,
        allow_unsafe_werkzeug=True  # Add this if you get warnings about unsafe werkzeug
    )