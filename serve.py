import flask
from flask import Flask, request
import json
import utils
from utils import beam_summarize

global model
model = None
app = Flask(__name__)

@app.route("/", methods=["GET"])
def _hello_world():
	return "Hello world"

@app.route("/summarize", methods=["POST"])
def _summarize():
	data = {"success": False}
	if request.form['text']:
		summary_text = beam_summarize(model, request.form['text'])
		data["summary"] = summary_text
		data["success"] = True
	return json.dumps(data, ensure_ascii=False)

# @app.route("/pretrained", methods=["POST"])
# def _pretrained():
# 	data = {"success": False}
# 	if request.form['text']:
# 		inputs = tokenizer([request.form['text']], max_length=1024, return_tensors='pt')
# 		summary_ids = pretrained_model.generate(inputs["input_ids"], num_beams=4, max_length=142, early_stopping=True)
# 		summary_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
# 						summary_ids]
# 		data["summary"] = summary_text
# 		data["success"] = True
# 	return json.dumps(data, ensure_ascii=False)

if __name__ == "__main__":
	model = utils._load_model()
	# pretrained_model = utils._load_pretrained_model()
	tokenizer = utils._load_tokenizer()
	print("App run!")
	# app.run(debug=False, host='127.0.0.1', threaded=False)
	app.run(debug=False)
