import os
import logging
from flask import Flask, request
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from ai_bot import OnboardingMessage
import ai_bot
import ssl as ssl_lib
import certifi
import threading
import requests

ssl_context = ssl_lib.create_default_context(cafile=certifi.where())

# Initialize a Flask app to host the events adapter
app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(os.environ["SLACK_SIGNING_SECRET"], "/slack/events", app)

# Initialize a Web API client
slack_web_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

# For simplicity we'll store our app data in-memory with the following data structure.
# onboarding_tutorials_sent = {"channel": {"user_id": OnboardingMessage}}
onboarding_tutorials_sent = {}

# Load model once and for all (using load data function)
data = ai_bot.LoadingData()
cat_to_tag,tag_to_response = data.cat_to_tag,data.tag_to_response
_clean_text,_tensorize = data._clean_text,data._tensorize

X_shape,Y_shape = data.X,data.Y # Need data for the shaped to be used

model_obj_1 = ai_bot.ModelFcnn()
X,Y = model_obj_1.get_input_array(X_shape,_clean_text,_tensorize),Y_shape
model_obj_1._build(X,Y)
model_obj_1.model.load_weights("model/FCNN.h5")

model_obj_2 = ai_bot.ModelLstm()
X,Y = model_obj_2.get_input_array(X_shape,_clean_text,_tensorize),Y_shape
model_obj_2._build(X,Y,EMBEDDING_DIM = 128)
model_obj_2.model.load_weights("model/LSTM.h5")

model_obj_3 = ai_bot.ModelRnnlm()
X,Y = model_obj_3.get_input_array(X_shape,_clean_text,_tensorize),Y_shape
model_obj_3._build(X,Y)
model_obj_3.model.load_weights("model/RNNLM.h5")

model_obj_4 = ai_bot.ModelBert(512,Y_shape.shape[1])
model_obj_4._build()
model_obj_4.model.load_weights("model/Bert.h5")

model_objects = [model_obj_1,model_obj_2,model_obj_3,model_obj_4]

def start_onboarding(user_id: str, channel: str):
    # Create a new onboarding tutorial.
    onboarding_tutorial = OnboardingMessage(channel)

    # Get the onboarding message payload
    message = onboarding_tutorial.get_message_payload()

    # Post the onboarding message in Slack
    response = slack_web_client.chat_postMessage(**message)

    # Capture the timestamp of the message we've just posted so
    # we can use it to update the message after a user
    # has completed an onboarding task.
    onboarding_tutorial.timestamp = response["ts"]

    # Store the message sent in onboarding_tutorials_sent
    if channel not in onboarding_tutorials_sent:
        onboarding_tutorials_sent[channel] = {}
    onboarding_tutorials_sent[channel][user_id] = onboarding_tutorial


# ================ Team Join Event =============== #
# When the user first joins a team, the type of the event will be 'team_join'.
# Here we'll link the onboarding_message callback to the 'team_join' event.
@slack_events_adapter.on("team_join")
def onboarding_message(payload):
    """Create and send an onboarding welcome message to new users. Save the
    time stamp of this message so we can update this message in the future.
    """
    event = payload.get("event", {})

    # Get the id of the Slack user associated with the incoming event
    user_id = event.get("user", {}).get("id")

    # Open a DM with the new user.
    response = slack_web_client.im_open(user=user_id)
    channel = response["channel"]["id"]

    # Post the onboarding message.
    start_onboarding(user_id, channel)


# ============= Reaction Added Events ============= #
# When a users adds an emoji reaction to the onboarding message,
# the type of the event will be 'reaction_added'.
# Here we'll link the update_emoji callback to the 'reaction_added' event.
@slack_events_adapter.on("reaction_added")
def update_emoji(payload):
    """Update the onboarding welcome message after receiving a "reaction_added"
    event from Slack. Update timestamp for welcome message as well.
    """
    event = payload.get("event", {})

    channel_id = event.get("item", {}).get("channel")
    user_id = event.get("user")

    if channel_id not in onboarding_tutorials_sent:
        return

    # Get the original tutorial sent.
    onboarding_tutorial = onboarding_tutorials_sent[channel_id][user_id]

    # Mark the reaction task as completed.
    onboarding_tutorial.reaction_task_completed = True

    # Get the new message payload
    message = onboarding_tutorial.get_message_payload()

    # Post the updated message in Slack
    updated_message = slack_web_client.chat_update(**message)

    # Update the timestamp saved on the onboarding tutorial object
    onboarding_tutorial.timestamp = updated_message["ts"]


# =============== Pin Added Events ================ #
# When a users pins a message the type of the event will be 'pin_added'.
# Here we'll link the update_pin callback to the 'pin_added' event.
@slack_events_adapter.on("pin_added")
def update_pin(payload):
    """Update the onboarding welcome message after receiving a "pin_added"
    event from Slack. Update timestamp for welcome message as well.
    """
    event = payload.get("event", {})

    channel_id = event.get("channel_id")
    user_id = event.get("user")

    # Get the original tutorial sent.
    onboarding_tutorial = onboarding_tutorials_sent[channel_id][user_id]

    # Mark the pin task as completed.
    onboarding_tutorial.pin_task_completed = True

    # Get the new message payload
    message = onboarding_tutorial.get_message_payload()

    # Post the updated message in Slack
    updated_message = slack_web_client.chat_update(**message)

    # Update the timestamp saved on the onboarding tutorial object
    onboarding_tutorial.timestamp = updated_message["ts"]


# ============== Message Events ============= #
# When a user sends a DM or a message in channel, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@slack_events_adapter.on("message")
def message(payload):
    """Display the onboarding welcome message after receiving a message
    that contains "start".
    """
    event = payload.get("event", {})

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")


    if text and text.lower() == "start":
        return start_onboarding(user_id, channel_id)


# ============== App_mention Events ============= #
# When a user sends a message with mention "@amexbot", the event type will be 'app_mention'.
# Here we'll link the message callback to the 'app_mention' event.
@slack_events_adapter.on("app_mention")
def slash_response(payload):                
    """endpoint for receiving all slash command requests from Slack"""

    # get the full request from Slack
    slack_request = request.form
    print(slack_request)
    # starting a new thread for doing the actual processing    
    x = threading.Thread(
            target=ai_message,
            args=(slack_request,payload,)
        )
    x.start()

    ## respond to Slack with quick message
    # and end the main thread for this request
    return "Let me think.... Please wait :robot_face:"
def ai_message(slack_request,payload):
    """Respond using the ML model when the bot is mentioned with "@amexbot"
    using the text
    """
    event = payload.get("event", {})

    channel_id = event.get("channel")
    text = event.get("text")

    # Create a message structure with an intent prediction.
    ai_chatbot = ai_bot.SlackMessage(channel_id, text)

    # Get the response predicted
    message = ai_chatbot.get_message_payload(model_objects,_clean_text,_tensorize,cat_to_tag,tag_to_response)
    
    # Post the AI message in Slack
    response = slack_web_client.chat_postMessage(**message)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    app.run(port=3000)