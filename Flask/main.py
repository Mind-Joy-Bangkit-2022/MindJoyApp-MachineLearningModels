from flask import Flask, jsonify, request
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow_hub as tfhub
app = Flask(__name__)

emotion_labels = {0: "Happy", 1: "Neutral", 2: "Sad"}
mental_health_labels = {0: "Butuh Penanganan", 1: "Tidak Butuh Penanganan"}


def predictMentalHealth(data):
    # load model
    model = load_model("mental-health-03_v3.h5", custom_objects={
        'KerasLayer': tfhub.KerasLayer})
    print(data)
    predictions = model.predict(data)
    predicted_class_indices = np.where(predictions < 0.5, 0, 1)
    if predicted_class_indices == 0:
        result = "Butuh Penanganan"
    else:
        result = "Tidak Butuh Penanganan"
    return result


def processEmotion(IMG_PATH):
    # load model
    model = load_model("emotion_classification_01.h5", custom_objects={
        'KerasLayer': tfhub.KerasLayer})

    img = image.load_img(IMG_PATH, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    # because on train and test image is normalized, on image predict supposed to be too.
    images /= 255
    # the value is not always 1 and 0 because of probabilities
    classes = model.predict(images, 64)
    # use to check prediction that have higher probabilities
    predicted_class_indices = np.argmax(classes)
    value = "empty"
    if predicted_class_indices == 0:
        value = 'Happy'
    elif predicted_class_indices == 1:
        value = 'Neutral'
    else:
        value = 'Sad'
    return value


@app.route("/")
def main():
    return "Application is working"


@app.route("/mentalhealth", methods=["POST", "GET"])
def mentalhHealthReq():
    items = ['Gender', 'Are you above 30 years of age?', 'How are you feeling today?',
             'Is your sadness momentarily or has it been constant for a long time?',
             'At what time of the day are you extremely low?',
             'How frequently have you had little pleasure or interest in the activities you usually enjoy?',
             'How confident you have been feeling in your capabilities recently.',
             'Describe how ‘supported’ you feel by others around you – your friends, family, or otherwise.',
             'How frequently have you been doing things that mean something to you or your life?',
             'How easy is it for you to take medical leave for a mental health condition?',
             'How often do you make use of substance abuse(e.g. smoking, alcohol)?',
             'How many hours do you spend per day on watching mobile phone, laptop, computer, television, etc.?',
             'If sad, how likely are you to take an appointment with a psychologist or a counsellor for your current mental state?',
             'How often do you get offended or angry or start crying ?',
             'How likely do you feel yourself vulnerable or lonely?',
             'How comfortable are you in talking about your mental health?',
             'Prediction']

    for item in items:
        if item not in request.form:
            msg = item + " is empty"
            return jsonify({"error": msg})

    gender = int(request.form.get('Gender'))
    age = int(request.form.get('Are you above 30 years of age?'))
    feeling = int(request.form.get('How are you feeling today?'))
    sadness = int(request.form.get('Is your sadness momentarily or has it been constant for a long time?'))
    time = int(request.form.get('At what time of the day are you extremely low?'))
    interest = int(request.form.get('How frequently have you had little pleasure or interest in the activities you usually enjoy?'))
    confident = int(request.form.get('How confident you have been feeling in your capabilities recently.'))
    supported = int(request.form.get('Describe how ‘supported’ you feel by others around you – your friends, family, or otherwise.'))
    things = int(request.form.get('How frequently have you been doing things that mean something to you or your life?'))
    medical = int(request.form.get('How easy is it for you to take medical leave for a mental health condition?'))
    substance = int(request.form.get('How often do you make use of substance abuse(e.g. smoking, alcohol)?'))
    hours = int(request.form.get('How many hours do you spend per day on watching mobile phone, laptop, computer, television, etc.?'))
    appointment = int(request.form.get('If sad, how likely are you to take an appointment with a psychologist or a counsellor for your current mental state?'))
    offended = int(request.form.get('How often do you get offended or angry or start crying ?'))
    vulnerable = int(request.form.get('How likely do you feel yourself vulnerable or lonely?'))
    comfortable = int(request.form.get('How comfortable are you in talking about your mental health?'))
    prediction = gender+age+feeling+sadness+time+interest+confident+supported+things+medical+substance+hours+appointment+offended+vulnerable+comfortable


    #     data_dummy = [age, "female" if gender == "male" else "female",
    #                   0 if feeling == 1 else 1,
    #                   0 if sadness == 1 else 1,
    #                   0 if feeling == 1 else 1,
    #                   0 if activities_interest == 1 else 1,
    #                   0 if confident == 1 else 1,
    #                   0 if supported == 1 else 1,
    #                   0 if doing_thing == 1 else 1,
    #                   0 if medical == 1 else 1,
    #                   0 if substance_abuse == 1 else 1,
    #                   0 if using_gadget == 1 else 1,
    #                   0 if appoinment == 1 else 1,'
    data_dummy = []
    #gender
    if gender == 0:
        data_dummy.append("Male")
    elif gender == 1:
        data_dummy.append("Female")
    else:
        data_dummy.append("Prefer not to say")
    # age
    if age == 0:
        data_dummy.append("No")
    else:
        data_dummy.append("Yes")
    # feeling
    if feeling == 0:
        data_dummy.append("Fine")
    elif feeling == 1:
        data_dummy.append("Good")
    elif feeling == 2:
        data_dummy.append("Sad")
    else:
        data_dummy.append("Depressed")
    # sadness
    if sadness == 0:
        data_dummy.append("Not sad")
    elif sadness == 1:
        data_dummy.append("For some time")
    elif sadness == 2:
        data_dummy.append("Significant time")
    else:
        data_dummy.append("Long time")
    # time
    if time == 0:
        data_dummy.append("Morning")
    elif time == 1:
        data_dummy.append("Afternoon")
    else:
        data_dummy.append("Evening")
    # activities_interest
    if interest == 0:
        data_dummy.append("Never")
    elif interest == 1:
        data_dummy.append("Sometimes")
    elif interest == 2:
        data_dummy.append("Often")
    else:
        data_dummy.append("Very often")
    # confident
    if confident == 1:
        data_dummy.append(1)
    elif confident == 2:
        data_dummy.append(2)
    elif confident == 3:
        data_dummy.append(3)
    elif confident == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)
    # supported
    if supported == 0:
        data_dummy.append("Highly supportive")
    elif supported == 1:
        data_dummy.append("Satisfactory")
    elif supported == 2:
        data_dummy.append("Little bit")
    else:
        data_dummy.append("Not at all")
    # doing_thing
    if things == 0:
        data_dummy.append("Very Often")
    elif things == 1:
        data_dummy.append("Often")
    elif things == 2:
        data_dummy.append("Sometimes")
    else:
        data_dummy.append("Never")
    # medical
    if medical == 0:
        data_dummy.append("Very easy")
    elif medical == 1:
        data_dummy.append("Easy")
    elif medical == 2:
        data_dummy.append("Not so easy")
    else:
        data_dummy.append("Difficult")
    # substance_abuse
    if substance == 0:
        data_dummy.append("Never")
    elif substance == 1:
        data_dummy.append("Sometimes")
    elif substance == 2:
        data_dummy.append("Often")
    else:
        data_dummy.append("Very Often")
    # using_gadget
    if hours == 0:
        data_dummy.append("1-2 hours")
    elif hours == 1:
        data_dummy.append("2-5 hours")
    elif hours == 2:
        data_dummy.append("5-10 hours")
    else:
        data_dummy.append("More than 10 hours")
    # appointment
    if appointment == 1:
        data_dummy.append(1)
    elif appointment == 2:
        data_dummy.append(2)
    elif appointment == 3:
        data_dummy.append(3)
    elif appointment == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)
    # offended
    if offended == 0:
        data_dummy.append("Never")
    elif offended == 1:
        data_dummy.append("Sometimes")
    elif offended == 2:
        data_dummy.append("Often")
    else:
        data_dummy.append("Very often")
    # vulnerable
    if vulnerable == 1:
        data_dummy.append(1)
    elif vulnerable == 2:
        data_dummy.append(2)
    elif vulnerable == 3:
        data_dummy.append(3)
    elif vulnerable == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)
    # comfortable
    if comfortable == 1:
        data_dummy.append(1)
    elif comfortable == 2:
        data_dummy.append(2)
    elif comfortable == 3:
        data_dummy.append(3)
    elif comfortable == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)
    #Prediction
    if prediction > 35:
        data_dummy.append(0)
    else:
        data_dummy.append(1)

    data = [data_dummy]

    data_df = pd.DataFrame(data=data, columns=items)

    features_cat = pd.get_dummies(data_df[items].astype('category'))
    features = pd.concat([data_df, features_cat], axis=1)
    features = features.drop(columns=items).loc[[0], :]

    resp = predictMentalHealth(features)

    return jsonify({"result": resp})


@app.route("/emotion", methods=["POST", 'GET'])
def emotionReq():
    if 'img' not in request.files:
        return jsonify({"error": "Image is empty"})
    data = request.files["img"]
    data.save("img.jpg")
    resp = processEmotion("img.jpg")

    return jsonify({"result": resp})


if __name__ == '__main__':
    app.run(debug=True)
