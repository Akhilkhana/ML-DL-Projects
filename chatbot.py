from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections

app = Flask(__name__)

pairs = [
    [r"(.*)my name is (.*)", ["Hello %2, How are you today ?"]],
    [r"(.*)help(.*)", ["I can help you"]],
    [r"(.*) your name ?",
        ["My name is thecleverprogrammer, but you can just call me robot."]],
    [r"how are you (.*) ?", ["I'm doing very well", "I am great!"]],
    [r"sorry (.*)", ["Its alright", "Its OK, never mind that"]],
    [r"i'm (.*) (good|well|okay|ok)", ["Nice to hear that", "Alright, great!"]],
    [r"(hi|hey|hello|hola|holla)(.*)", ["Hello", "Hey there"]],
    [r"what (.*) want ?", ["Make me an offer I can't refuse"]],
    [r"(.*)created(.*)",
        ["prakash created me using Python's NLTK library", "top secret ;)"]],
    [r"(.*) (location|city) ?", ["hyderabad, India"]],
    [r"(.*)raining in (.*)", ["No rain in the past 4 days here in %2",
                              "In %2 there is a 50% chance of rain"]],
    [r"how (.*) health (.*)", ["Health is very important, but I am a computer."]],
    [r"(.*)(sports|game|sport)(.*)", ["I'm a very big fan of Cricket"]],
    [r"who (.*) (Cricketer|Batsman)?", ["Virat Kohli"]],
    [r"quit", ["Bye for now. See you soon :)", "It was nice talking to you."]],
    [r"(.*)", ["Our customer service will reach you"]]
]

chatbot = Chat(pairs, reflections)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = chatbot.respond(user_input)
    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(debug=True)
