from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections

app = Flask(__name__)

pairs = [
    [r"(.*)my name is (.*)", ["Hello %2, how are you today?"]],
    [r"(.*)help(.*)", ["I can help you! What do you need?"]],
    [r"(.*) your name ?",
     ["My name is Chatbot. How can I assist you?"]],
    [r"how are you (.*) ?", ["I'm doing very well, thanks!", "I am great!"]],
    [r"sorry (.*)", ["It's alright!", "No worries, never mind."]],
    [r"i'm (.*) (good|well|okay|ok)", ["Nice to hear that!", "Alright, great!"]],
    [r"(hi|hey|hello|hola|holla)(.*)", ["Hello!", "Hey there! How can I help?"]],
    [r"what (.*) want ?", ["Make me an offer I can't refuse."]],
    [r"(.*)created(.*)",
     ["I was created using Python's NLTK library.", "That's a bit of a secret ;)"]],
    [r"(.*) (location|city) ?", ["I'm a virtual assistant — I exist everywhere!"]],
    [r"(.*)raining in (.*)", ["No rain in the past 4 days in %2",
                              "In %2 there is a 50% chance of rain"]],
    [r"how (.*) health (.*)", ["Health is very important, but I'm just a bot!"]],
    [r"(.*)(sports|game|sport)(.*)", ["I'm a big fan of cricket!"]],
    [r"who (.*) (Cricketer|Batsman)?", ["Virat Kohli — what a player!"]],
    [r"quit", ["Bye for now. See you soon :)", "It was nice talking to you!"]],
    [r"(.*)", ["I'm not sure I understand. Could you rephrase that?",
               "Our support team will reach you shortly."]]
]

chatbot = Chat(pairs, reflections)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = chatbot.respond(user_input) or "Sorry, I didn't understand that."
    return jsonify({"reply": response})


if __name__ == "__main__":
    # Set debug=False for production; use debug=True only during local development
    app.run(debug=False)
