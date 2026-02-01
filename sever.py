from flask import Flask, request, jsonify
from main import run_drone_simulation  # wrap your logic in a function

app = Flask(__name__)

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    start = data['start']
    end = data['end']
    result = run_drone_simulation(start, end)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
