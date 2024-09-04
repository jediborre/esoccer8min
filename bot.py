import os
import telebot
import logging
from dotenv import load_dotenv
from .model import get_models, predict_game

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID').split(',')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
models = get_models()


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    msj = message.text
    if ' v ' in msj:
        home, away = msj.split(' v ')
        results = predict_game(
            models, {
                'home_player': home,
                'away_player': away,
            })
        
        bot.reply_to(message, "You said: " + message.text)


bot.polling()
