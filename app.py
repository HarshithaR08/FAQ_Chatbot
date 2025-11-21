# Developer: Prudhvi
# Added this script as a backend logic that uses FLASK 

from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS 
from ui.rag_cli import generate_answer  # assumes this function exists

 

app = Flask(__name__, template_folder="frontend_template")

CORS(app)

 

@app.route("/")

def index():

    return render_template("index.html")

 

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # generate_answer returns: (answer_text, primary_source, intent)
        answer_text, primary_source, intent = generate_answer(question, top_k=4)

        return jsonify({
            "answer": answer_text,
            "source": primary_source,
            "intent": intent,
        })
    except Exception as e:
        import traceback
        print("\n[BACKEND ERROR in /chat]")
        traceback.print_exc()
        return jsonify({"error": f"Backend error: {type(e).__name__}"}), 500


 

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)

 