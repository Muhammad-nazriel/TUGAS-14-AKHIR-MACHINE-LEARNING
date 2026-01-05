from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model/xgb_model.pkl', 'rb'))
EXPECTED_FEATURES = 30

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        raw = request.form.get('features', '')

        try:
            values = [float(x.strip()) for x in raw.split(',') if x.strip() != '']

            if len(values) != EXPECTED_FEATURES:
                error = f"Jumlah fitur harus {EXPECTED_FEATURES}, saat ini {len(values)}."
            else:
                values = np.array(values).reshape(1, -1)
                prediction = int(model.predict(values)[0])

        except ValueError:
            error = "Input harus berupa angka numerik yang dipisahkan dengan koma."

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
