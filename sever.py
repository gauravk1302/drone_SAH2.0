from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/set_coords', methods=['POST'])
def set_coords():
    data = request.json
    with open("coords.json", "w") as f:
        json.dump(data, f)
    return "Coordinates received", 200

if __name__ == '__main__':
    app.run(debug=True)
