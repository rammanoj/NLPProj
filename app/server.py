import http.server
import socketserver
import json
<<<<<<< HEAD
import pickle

PORT = 8081

model_file = "fill this in!"
trained_model = pickle.load(model_file)

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8000')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

=======

PORT = 8081

class MyHandler(http.server.BaseHTTPRequestHandler):
>>>>>>> 9609b04 (added more presentation stuff)
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = post_data.decode('utf-8')
        post_data_dict = json.loads(post_data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
<<<<<<< HEAD
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:8000')
=======
>>>>>>> 9609b04 (added more presentation stuff)
        self.end_headers()

        response_data = {'message': 'Received POST request', 'data': post_data_dict}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def do_GET(self):
        if self.path == '/food':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Here's your food!")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"404 - Not Found")

<<<<<<< HEAD

=======
>>>>>>> 9609b04 (added more presentation stuff)
Handler = MyHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Server started at port", PORT)
    httpd.serve_forever()
