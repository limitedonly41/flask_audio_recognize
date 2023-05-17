# Importing required libs
from flask import Flask, render_template, request
# from model import predict_result



from faster_whisper import WhisperModel

model_size = "small"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# model = WhisperModel(model_size)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")



# Instantiating flask app
app = Flask(__name__)
# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            audio = request.files['file'].stream
            # pred = predict_result(audio)
            segments, info = model.transcribe(audio, beam_size=5)

            # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            ans = []
            for segment in segments:
                # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                ans.append("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            result = '\n'.join(map(str, ans))
            result += '\n\n'
            return render_template("result.html", predictions=result)

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)