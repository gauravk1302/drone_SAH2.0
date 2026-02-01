from flask import Flask, render_template, request
from main import run_drone_simulation

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start = (float(request.form["start_lat"]), float(request.form["start_lon"]))
        end = (float(request.form["end_lat"]), float(request.form["end_lon"]))
        result = run_drone_simulation(start, end)
        return render_template("result.html", result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
