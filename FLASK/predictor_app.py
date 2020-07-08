import flask
from flask import request
import pickle
import numpy as np

# Initiate

app = flask.Flask(__name__)

with open("KCT_final_fix_model.pkl","rb") as f:
    logr = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

hr_features = ['What is the Kick s goal? ($) ',
               'How long it will take? (days) ',
               'Number of Reward Tiers? (# of gifts)',
               'Will be this your first Kick?',
              'Staff Pick',
               'publishing',
               'journalism',
               'music',
               'crafts',
               'food',
               'technology',
               'photography',
               'theater',
               'dance',
               'games',
               'comics',
               'art',
               'fashion',
               'design',
               'film & video']

@app.route("/", methods=["POST", "GET"])
def predict():


    print(request.args)

    x_input = []
    x_input_sp = []
    for i in range(len(hr_features[:3])):
        # f_value = 0
        f_value = int(
            request.args.get(hr_features[i], "0")
            )
        x_input.append(f_value)
        x_input_sp.append(f_value)

    # FIRST PROJECT?
    key_val = hr_features[3]
    first_project = int(request.args.get(key_val,"1"))
    x_input.append(first_project)
    x_input_sp.append(first_project)

    x_input.append(0)
    x_input_sp.append(1)

    # 0's categories
    x_input.extend([0] * len(hr_features[5:]))
    x_input_sp.extend([0] * len(hr_features[5:]))

    # GET CATEGORY
    cat = request.args.get('category','publishing')
    cat_idx = hr_features.index(cat)
    x_input[cat_idx] = 1
    x_input_sp[cat_idx] = 1

    x_input_std = scaler.transform(np.array(x_input).reshape((1, -1)))
    x_input_sp_std = scaler.transform(np.array(x_input_sp).reshape((1, -1)))

    pred_probs = logr.predict_proba(x_input_std)
    pred_probs_sp = logr.predict_proba(x_input_sp_std)

    success_proba = str(round(pred_probs[0][1] * 100, 1)) + "%"
    success_proba_sp = str(round(pred_probs_sp[0][1] * 100, 1)) + "%"

    # RESPONSE
    return flask.render_template('predictor.html',
    feature_names=hr_features,
    categories = hr_features[5:],
    x_input=x_input,
    prediction=success_proba,
    prediction_sp =success_proba_sp
    )


app.run(debug=True)


