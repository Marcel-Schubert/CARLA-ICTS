import json
import requests as requests
import time
import sys

sys.path.append("/workspace/data/CARLA-ICTS")


def telegram_bot_sendtext(bot_message):
    try:
        config = json.load(open('./NotificationService/config.json'))
        BOT_TOKEN = config['BOT_TOKEN']
        CHAT_ID = config['CHAT_ID']
        send_text = 'https://api.telegram.org/bot'\
                    + BOT_TOKEN + '/sendMessage?chat_id='\
                    + CHAT_ID + '&parse_mode=Markdown&text='\
                    + bot_message
        response = requests.get(send_text)
        print(bot_message)
        return response.json()
    except Exception as e:
        print(f"Error in sending message: {e}")
        return None


if __name__ == '__main__':
    telegram_bot_sendtext("Training started")
    time.sleep(10)
    telegram_bot_sendtext("Training finished")