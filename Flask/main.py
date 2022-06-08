from flask import Flask, jsonify, request

# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model

app = Flask(__name__)

emotion_labels = {0: "Happy", 1: "Neutral", 2: "Sad"}
mental_health_labels = {0: "Butuh Penanganan", 1: "Tidak Butuh Penanganan"}


def predictMentalHealth(data):
    # load model
    model = load_model("mental-health-03.h5", custom_objects={
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
    model = load_model("./ML-API/.h5", custom_objects={
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
             'How comfortable are you in talking about your mental health?']

    items_cat = ['Are you above 30 years of age?', 'How are you feeling today?',
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
                 'How comfortable are you in talking about your mental health?']

    for item in items:
        if item not in request.form:
            msg = item + " is empty"
            return jsonify({"error": msg})

    gender = str(request.form['Gender'])
    age = int(request.form['Are you above 30 years of age?'])
    # age = (age - 1) / (96 - 1)
    feeling = int(request.form['How are you feeling today?'])
    sadness = int(request.form['Is your sadness momentarily or has it been constant for a long time?'])
    time = int(request.form['At what time of the day are you extremely low?'])
    activities_interest = int(
        request.form['How frequently have you had little pleasure or interest in the activities you usually enjoy?'])
    confident = int(request.form['How confident you have been feeling in your capabilities recently.'])
    supported = int(
        request.form['Describe how ‘supported’ you feel by others around you – your friends, family, or otherwise.'])
    doing_thing = int(
        request.form['How frequently have you been doing things that mean something to you or your life?'])
    medical = int(request.form['How easy is it for you to take medical leave for a mental health condition?'])
    substance_abuse = int(request.form['How often do you make use of substance abuse(e.g. smoking, alcohol)?'])
    using_gadget = int(request.form[
                           'How many hours do you spend per day on watching mobile phone, laptop, computer, television, etc.?'])
    appoinment = int(request.form[
                         'If sad, how likely are you to take an appointment with a psychologist or a counsellor for your current mental state?'])
    get_offended = int(request.form['How often do you get offended or angry or start crying ?'])
    vulnerable_lonely = int(request.form['How likely do you feel yourself vulnerable or lonely?'])
    comfort = int(request.form['How comfortable are you in talking about your mental health?'])

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

    # gender
    if gender == 0:
        data_dummy.append("Male")
    elif gender == 1:
        data_dummy.append("Female")
    else:
        data_dummy.append("Prefer not to say")
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
    if activities_interest == 0:
        data_dummy.append("Never")
    elif activities_interest == 1:
        data_dummy.append("Sometimes")
    elif activities_interest == 2:
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
    if doing_thing == 0:
        data_dummy.append("Very Often")
    elif doing_thing == 1:
        data_dummy.append("Often")
    elif doing_thing == 2:
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
    if substance_abuse == 0:
        data_dummy.append("Never")
    elif substance_abuse == 1:
        data_dummy.append("Sometimes")
    elif substance_abuse == 2:
        data_dummy.append("Often")
    else:
        data_dummy.append("Very Often")
    # using_gadget
    if using_gadget == 0:
        data_dummy.append("1-2 hours")
    elif using_gadget == 1:
        data_dummy.append("2-5 hours")
    elif using_gadget == 2:
        data_dummy.append("5-10 hours")
    else:
        data_dummy.append("More than 10 hours")
    # appointment
    if appoinment == 1:
        data_dummy.append(1)
    elif appoinment == 2:
        data_dummy.append(2)
    elif appoinment == 3:
        data_dummy.append(3)
    elif appoinment == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)
    # get_offended
    if get_offended == 0:
        data_dummy.append("Never")
    elif get_offended == 1:
        data_dummy.append("Sometimes")
    elif get_offended == 2:
        data_dummy.append("Often")
    else:
        data_dummy.append("Very often")
    # vulnerable_lonely
    if vulnerable_lonely == 1:
        data_dummy.append(1)
    elif vulnerable_lonely == 2:
        data_dummy.append(2)
    elif vulnerable_lonely == 3:
        data_dummy.append(3)
    elif vulnerable_lonely == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)
    # comfort
    if comfort == 1:
        data_dummy.append(1)
    elif comfort == 2:
        data_dummy.append(2)
    elif comfort == 3:
        data_dummy.append(3)
    elif comfort == 4:
        data_dummy.append(4)
    else:
        data_dummy.append(5)

    data = [[age, gender, feeling, sadness, time, activities_interest, confident, supported, doing_thing, medical,
             substance_abuse, using_gadget, appoinment, get_offended, vulnerable_lonely, comfort], data_dummy]

    data_df = pd.DataFrame(data=data, columns=items)

    features_cat = pd.get_dummies(data_df[items_cat].astype('category'))
    features = pd.concat([data_df, features_cat], axis=1)
    features = features.drop(columns=items_cat).loc[[0], :]

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
